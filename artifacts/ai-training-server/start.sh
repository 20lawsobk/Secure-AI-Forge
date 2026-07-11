#!/bin/bash
# Start the AI Training Server
set -e

# Ensure Python dependencies are installed via uv
cd /home/runner/workspace
if ! python3 -c "import torch, fastapi, uvicorn, psycopg2" 2>/dev/null; then
    echo "[Start] Installing Python dependencies via uv..."
    uv sync --no-dev
fi

SERVER_DIR="$(dirname "$0")"
cd "$SERVER_DIR"

# Start the server
echo "[Start] Starting MaxBooster AI Training Server..."
python3 server.py
