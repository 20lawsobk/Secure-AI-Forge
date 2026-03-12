"""Entrypoint to start the AI model FastAPI server"""
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("AI_MODEL_PORT", "9878"))
    uvicorn.run(
        "ai_model.api.app:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
    )
