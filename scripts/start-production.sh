#!/bin/bash
set -e

echo "[Deploy] Starting MaxBooster AI Training Server (Python)..."
MODEL_API_PORT=9878 python3 artifacts/ai-training-server/server.py &

echo "[Deploy] Waiting for AI Training Server to be ready on port 9878..."
until curl -sf http://localhost:9878/health > /dev/null 2>&1; do
  sleep 1
done
echo "[Deploy] AI Training Server is ready."

echo "[Deploy] Starting API Server (Node.js)..."
exec node artifacts/api-server/dist/index.cjs
