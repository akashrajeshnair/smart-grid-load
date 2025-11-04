from flask import Flask, render_template, jsonify
from hdfs import InsecureClient
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import shutil
import os
import threading
import time

app = Flask(__name__)

# -------------------------------
# 1. HDFS Config
# -------------------------------
HDFS_URL = "http://localhost:9870"       # adjust if different
HDFS_PRED_PATH = "/user/akash/predictions"
LOCAL_PRED_FILE = "latest_predictions.parquet"

hdfs_client = InsecureClient(HDFS_URL, user='akash')

# -------------------------------
# 2. Fetch and Cache Latest Predictions
# -------------------------------
def fetch_latest_predictions():
    try:
        if os.path.exists(LOCAL_PRED_FILE):
            shutil.rmtree(LOCAL_PRED_FILE)
        hdfs_client.download(HDFS_PRED_PATH, LOCAL_PRED_FILE, overwrite=True)
        dataset = ds.dataset(LOCAL_PRED_FILE, format="parquet")
        table = dataset.to_table()
        df = table.to_pandas()
        return df
    except Exception as e:
        print(f"WARN: Could not fetch HDFS data {e}")
        return pd.DataFrame()

# -------------------------------
# 3. Plotting Function
# -------------------------------
def plot_predictions(df):
    if df.empty:
        return None
    try:
        plt.figure(figsize=(8,4))
        df = df.tail(50)  # last 50 entries
        plt.plot(df["Timestamp"], df["Predicted Load (kW)"], label="Predicted Load", color='orange')
        if "Actual Load (kW)" in df.columns:
            plt.plot(df["Timestamp"], df["Actual Load (kW)"], label="Actual Load", color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Timestamp")
        plt.ylabel("Load (kW)")
        plt.legend()
        plt.tight_layout()
        plot_path = "static/plot.png"
        os.makedirs("static", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"[ERROR] Plot failed: {e}")
        return None

# -------------------------------
# 4. Alert System
# -------------------------------
ALERT_THRESHOLD = 250.0  # change this threshold as appropriate
alert_state = {"active": False, "last_value": 0.0}

def check_for_alert(df):
    if df.empty:
        return
    latest = df.iloc[-1]
    load = latest.get("Predicted Load (kW)", 0.0)
    if load > ALERT_THRESHOLD and not alert_state["active"]:
        alert_state["active"] = True
        alert_state["last_value"] = load
        print(f"⚠️ ALERT: Predicted load {load:.2f} kW exceeds threshold {ALERT_THRESHOLD}")
    elif load <= ALERT_THRESHOLD:
        alert_state["active"] = False

# -------------------------------
# 5. Background Updater
# -------------------------------
def background_updater():
    while True:
        df = fetch_latest_predictions()
        if not df.empty:
            check_for_alert(df)
            plot_predictions(df)
        time.sleep(20)  # refresh every 20 s


@app.before_request
def start_background_updater():
    """Start the background updater once when the first request is handled.

    This prevents the updater from starting at import time (which can cause
    duplicate threads when Flask's reloader or other process managers import
    the module). The thread is daemonized so it won't block process shutdown.
    """
    print("🔁 Starting dashboard background updater thread")
    t = threading.Thread(target=background_updater, daemon=True)
    t.start()

# -------------------------------
# 6. Flask Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html", alert=alert_state)

@app.route("/api/alert")
def get_alert():
    return jsonify(alert_state)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
