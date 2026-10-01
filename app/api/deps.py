"""FastAPI dependency providers (lazy singletons).

Services are built on first use so that importing the app is cheap for
tests, which may override dependencies entirely.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.document_service import DocumentService
from app.services.research_service import ResearchService


@lru_cache
def get_research_service() -> ResearchService:
    """Return the process-wide research service."""
    return ResearchService()


@lru_cache
def get_document_service() -> DocumentService:
    """Return the process-wide document service."""
    return DocumentService()


__all__ = ["get_research_service", "get_document_service", "get_settings"]