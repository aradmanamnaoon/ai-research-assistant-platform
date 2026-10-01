"""FastAPI application factory for the AI Research Assistant Platform.

Run locally::

    uvicorn app.api.main:app --reload

or ``python -m app.api.main`` / the Makefile targets.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health
from app.config import get_settings
from app.logging_config import configure_logging, get_logger

logger = get_logger("api")


def create_app() -> FastAPI:
    """Build the FastAPI app with all routes and middleware."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Multi-agent research assistant built with LangGraph, LangChain, "
            "ChromaDB RAG and LangSmith observability.\n\n"
            "* `POST /api/v1/chat` — ask a question (full pipeline)\n"
            "* `POST /api/v1/chat/stream` — same pipeline, SSE progress events\n"
            "* `POST /api/v1/documents/upload` — add documents to the knowledge base\n"
            "* `GET /docs` — interactive OpenAPI documentation"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(chat.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
    )