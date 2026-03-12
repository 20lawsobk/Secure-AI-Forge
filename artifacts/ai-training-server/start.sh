#!/bin/bash
# Start the AI Training Server
set -e

SERVER_DIR="$(dirname "$0")"
cd "$SERVER_DIR"

# Install Python deps if needed
if ! python3 -c "import fastapi, uvicorn, psycopg2" 2>/dev/null; then
    echo "[Start] Installing Python dependencies..."
    pip install fastapi uvicorn psycopg2-binary numpy torch --quiet
fi

# Start the server
echo "[Start] Starting MaxBooster AI Training Server..."
python3 server.py
