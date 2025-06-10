# main.py

import os
from fastapi import FastAPI
import uvicorn

from routes import router as rag_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Service",
        description="API for document ingestion and RAG-powered chat",
        version="1.0.0",
    )

    # Mount your router
    app.include_router(rag_router)

    return app

app = create_app()

@app.get("/")
async def Welcome():
    """
    Welcome endpoint to verify the API is running.
    """
    return {"message": "Welcome to the RAG API! Use /rag/ingest to upload documents or /rag/chat to start chatting."}


if __name__ == "__main__":
    # Host and port can be overridden via environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
    )

