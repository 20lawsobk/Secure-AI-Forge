#!/usr/bin/env bash
# One-time environment setup for MaxBooster (MaxCore AI)
# Run this after cloning or importing the project into a new Replit environment.
# Requires: Node.js 22+, Python 3.11+, pnpm, ffmpeg (all provided by .replit modules)

set -euo pipefail

echo "==> Installing Node.js dependencies..."
pnpm install

echo "==> Installing Python dependencies..."
pip install \
  fastapi \
  uvicorn \
  numpy \
  pillow \
  psycopg2-binary \
  pydantic \
  scikit-learn \
  scipy \
  soundfile \
  librosa \
  "torch>=2.12.1" \
  --index-url https://package-firewall.replit.local/pypi/simple/ \
  --quiet

echo "==> Pushing database schema..."
pnpm --filter @workspace/db push-force

echo ""
echo "Setup complete. Start the app with the 'Start application' workflow,"
echo "or manually: PORT=8080 MODEL_API_PORT=9878 pnpm --filter @workspace/api-server run dev &"
echo "            PORT=5000 BASE_PATH=/ API_PORT=8080 pnpm --filter @workspace/ai-dashboard run dev"
