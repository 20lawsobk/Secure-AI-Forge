#!/usr/bin/env bash
# One-time environment setup for MaxBooster (MaxCore AI)
# Run this after cloning or importing the project into a new Replit environment.
# Requires: Node.js 22+, Python 3.11+, pnpm, uv, ffmpeg (all provided by .replit modules)

set -euo pipefail

echo "==> Installing Node.js dependencies..."
pnpm install

echo "==> Installing Python dependencies (via uv)..."
cd artifacts/ai-training-server
uv sync --no-dev
cd ../..

echo "==> Pushing database schema..."
echo "" | pnpm --filter @workspace/db run push-force

echo ""
echo "Setup complete. Start the app with the 'Start application' workflow,"
echo "or manually:"
echo "  PORT=8080 MODEL_API_PORT=9878 pnpm --filter @workspace/api-server run dev &"
echo "  PORT=5000 BASE_PATH=/ API_PORT=8080 pnpm --filter @workspace/ai-dashboard run dev"
