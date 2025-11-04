#!/usr/bin/env bash
# Smart Grid Dashboard Runner
# ---------------------------
# Starts the Flask dashboard inside the virtual environment

# Exit on error
set -e

# Project root (change this if your folder path is different)
PROJECT_DIR="$HOME/Projects/smart-grid-load"

# Virtual environment path
VENV_DIR="$PROJECT_DIR/smartgrid_env"

# Flask app path
APP_FILE="$PROJECT_DIR/dashboard/app.py"

# Log file
LOG_FILE="$PROJECT_DIR/dashboard/dashboard.log"

# Activate the virtual environment
if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
else
  echo "❌ Virtual environment not found at $VENV_DIR"
  exit 1
fi

# Check if Flask is installed
if ! python -c "import flask" &>/dev/null; then
  echo "⚠️ Flask not installed. Installing dependencies..."
  pip install flask hdfs pandas matplotlib pyarrow
fi

# Run the Flask dashboard
echo "🚀 Starting Flask dashboard..."
cd "$PROJECT_DIR/dashboard"

# Run Flask in background, log output
nohup python "$APP_FILE" > "$LOG_FILE" 2>&1 &

# Print the process info
PID=$!
echo "✅ Dashboard running (PID: $PID)"
echo "📄 Logs: $LOG_FILE"
echo "🌐 Access: http://127.0.0.1:5000/"
