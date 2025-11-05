from flask import Flask, render_template, jsonify
from hdfs import InsecureClient
from hdfs.util import HdfsError
import pandas as pd
import json
import os
import threading
import time
from datetime import datetime

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable static file caching

# -------------------------------
# 1) HDFS + Local Cache Config
# -------------------------------
HDFS_URL = "http://localhost:9870"            # WebHDFS endpoint
HDFS_PRED_PATH = "/user/akash/predictions"    # Spark sink path
LOCAL_PRED_DIR = "cache/predictions"          # local cache for downloaded JSONs
MAX_FILES = 200                               # cap number of files per fetch
REFRESH_SECONDS = 5                           # faster refresh to see updates

hdfs_client = InsecureClient(HDFS_URL, user='akash')

# -------------------------------
# 2) Helpers
# -------------------------------
def _ensure_local_dir():
    os.makedirs(LOCAL_PRED_DIR, exist_ok=True)

def _clear_local_dir():
    _ensure_local_dir()
    for name in os.listdir(LOCAL_PRED_DIR):
        p = os.path.join(LOCAL_PRED_DIR, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass

def _list_hdfs_json_files():
    """List JSON data files under HDFS_PRED_PATH, skipping metadata and hidden files."""
    try:
        entries = hdfs_client.list(HDFS_PRED_PATH, status=True)
    except HdfsError as e:
        print(f"WARN: HDFS list failed: {e}")
        return []

    files = []
    for ent in entries:
        name = None
        etype = None
        mtime = 0
        status = None

        # (name, status) tuples from hdfs lib
        if isinstance(ent, tuple) and len(ent) == 2:
            name, status = ent
            if isinstance(status, dict):
                etype = status.get("type")
                mtime = status.get("modificationTime", 0)
        # dict form (some clients)
        elif isinstance(ent, dict):
            name = ent.get("pathSuffix") or ent.get("name") or ""
            etype = ent.get("type")
            mtime = ent.get("modificationTime", 0)
        # plain name; fetch status
        elif isinstance(ent, str):
            name = ent
            try:
                status = hdfs_client.status(f"{HDFS_PRED_PATH.rstrip('/')}/{name}", strict=False)
                if isinstance(status, dict):
                    etype = status.get("type")
                    mtime = status.get("modificationTime", 0)
            except HdfsError:
                pass

        if not name:
            continue
        if etype and etype != "FILE":
            continue
        if name.startswith("_") or name.startswith("."):
            continue
        if not name.lower().endswith(".json"):
            continue

        hdfs_path = f"{HDFS_PRED_PATH.rstrip('/')}/{name.lstrip('/')}"
        files.append((mtime, hdfs_path, name))

    files.sort(key=lambda x: x[0])
    if len(files) > MAX_FILES:
        files = files[-MAX_FILES:]
    return files

def _download_and_parse_single_json(hdfs_path: str, local_name: str) -> pd.DataFrame:
    """Download a single JSON object file from HDFS and return a one-row DataFrame."""
    local_path = os.path.join(LOCAL_PRED_DIR, local_name)
    try:
        hdfs_client.download(hdfs_path, local_path, overwrite=True)
    except HdfsError as e:
        print(f"WARN: download failed for {hdfs_path}: {e}")
        return pd.DataFrame()

    # Each file contains exactly one JSON object
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return pd.DataFrame([obj])
        # If it isn't a dict, discard
        os.remove(local_path)
        return pd.DataFrame()
    except Exception:
        try:
            os.remove(local_path)
        except Exception:
            pass
        return pd.DataFrame()

def fetch_latest_predictions() -> pd.DataFrame:
    """Fetch latest prediction records from HDFS (single-object JSON files)."""
    try:
        _clear_local_dir()
        files = _list_hdfs_json_files()
        if not files:
            return pd.DataFrame()

        frames = []
        for _, hdfs_path, name in files:
            df_part = _download_and_parse_single_json(hdfs_path, name)
            if not df_part.empty:
                frames.append(df_part)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Sort by Timestamp if present
        if "Timestamp" in df.columns:
            # Robust parse for ordering; keep original strings for plotting
            try:
                df["_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                df = df.sort_values("_ts").drop(columns=["_ts"])
            except Exception:
                df = df.sort_values("Timestamp")

        return df
    except Exception as e:
        print(f"WARN: Could not fetch HDFS data {e}")
        return pd.DataFrame()

# -------------------------------
# 3) In-memory cache for records (no chart)
# -------------------------------
predictions_cache = []  # list of dicts with at least Timestamp and Predicted Load (kW)
predictions_cache_lock = threading.Lock()
MAX_CACHE_ROWS = 1000

# -------------------------------
# 4) Alerts
# -------------------------------
ALERT_THRESHOLD = 10.0
alert_state = {"active": False, "last_value": 0.0, "threshold": ALERT_THRESHOLD, "updated_at": None}

def check_for_alert(df: pd.DataFrame):
    if df.empty or "Predicted Load (kW)" not in df.columns:
        return
    latest = df.iloc[-1]
    try:
        load = float(latest.get("Predicted Load (kW)", 0.0))
    except Exception:
        load = 0.0

    alert_state["last_value"] = load
    alert_state["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if load > ALERT_THRESHOLD:
        if not alert_state["active"]:
            print(f"ALERT: Predicted load {load:.2f} kW exceeds threshold {ALERT_THRESHOLD}")
        alert_state["active"] = True
    else:
        alert_state["active"] = False

# -------------------------------
# 5) Background updater
# -------------------------------
def background_updater():
    while True:
        df = fetch_latest_predictions()
        if not df.empty:
            # Update cache with latest rows
            essential_cols = ["Timestamp", "Predicted Load (kW)"]
            present_cols = [c for c in essential_cols if c in df.columns]
            df_view = df[present_cols] if present_cols else pd.DataFrame()

            with predictions_cache_lock:
                for rec in df_view.to_dict(orient="records"):
                    predictions_cache.append(rec)
                # Deduplicate by Timestamp, keeping newest
                if predictions_cache and present_cols and "Timestamp" in present_cols:
                    seen = set()
                    dedup = []
                    for item in reversed(predictions_cache):
                        ts = item.get("Timestamp")
                        if ts in seen:
                            continue
                        seen.add(ts)
                        dedup.append(item)
                    predictions_cache.clear()
                    predictions_cache.extend(reversed(dedup))
                # Trim cache
                if len(predictions_cache) > MAX_CACHE_ROWS:
                    del predictions_cache[0:len(predictions_cache) - MAX_CACHE_ROWS]

            # Check alerts on current batch
            check_for_alert(df)
        time.sleep(REFRESH_SECONDS)

_updater_started = False

@app.before_request
def start_background_updater():
    global _updater_started
    if not _updater_started:
        print("Starting background updater...")
        t = threading.Thread(target=background_updater, daemon=True)
        t.start()
        _updater_started = True

# Disable client/proxy caching so page updates are visible
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# -------------------------------
# 6) Routes
# -------------------------------
@app.route("/")
def index():
    # Provide latest 200 entries (newest last) to template
    with predictions_cache_lock:
        latest = predictions_cache[-200:]
    return render_template("index.html", alert=alert_state, records=latest)

@app.route("/api/alert")
def api_alert():
    return jsonify(alert_state)

@app.get("/api/predictions")
def api_predictions():
    with predictions_cache_lock:
        return jsonify(predictions_cache[-500:])

# -------------------------------
# 7) Entry point
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)