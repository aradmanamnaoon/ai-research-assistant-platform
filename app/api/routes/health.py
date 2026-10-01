"""Health and introspection routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_document_service
from app.services.document_service import DocumentService
from app.api.schemas import HealthResponse
from app.config import get_settings as _get_settings
from app.tools.registry import get_tool_catalog

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(documents: DocumentService = Depends(get_document_service)) -> HealthResponse:
    """Liveness/readiness probe with runtime configuration summary."""
    settings = _get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        llm_provider=settings.llm_provider,
        model_name=settings.model_name,
        documents_indexed=documents.store.count(),
        langsmith_enabled=settings.langsmith_enabled,
    )


@router.get("/tools")
def list_tools() -> dict:
    """Expose the agent tool catalog for introspection."""
    return {"tools": get_tool_catalog()}