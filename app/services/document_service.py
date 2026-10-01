"""Document service: CRUD over the RAG document library."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from langsmith import traceable

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.retrieval.ingest import ingest_text, load_file
from app.retrieval.vector_store import VectorStore

logger = get_logger("services.documents")


class DocumentService:
    """Manage uploaded documents: ingest, list, delete, search preview."""

    def __init__(
        self,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Create the service.

        Args:
            store: Optional pre-built vector store.
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self.store = store or VectorStore(settings=self.settings)

    @traceable(name="document_ingest")
    def ingest_upload(self, file_name: str, content: bytes) -> dict[str, Any]:
        """Ingest an uploaded file's bytes.

        Args:
            file_name: Original file name (determines parsing).
            content: File bytes.

        Returns:
            ``{"source", "chunks", "total_chunks"}`` on success.

        Raises:
            ValueError: On unsupported types or oversized files.
        """
        safe_name = Path(file_name).name  # strip any path components
        suffix = Path(safe_name).suffix.lower()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / safe_name
            tmp_path.write_bytes(content)
            documents = load_file(tmp_path)

        ids = ingest_text(
            documents[0].page_content,
            source=safe_name,
            metadata=documents[0].metadata,
            settings=self.settings,
            store=self.store,
        )
        logger.info("Ingested upload %s (%d chunks)", safe_name, len(ids))
        return {
            "source": safe_name,
            "chunks": len(ids),
            "total_chunks": self.store.count(),
        }

    @traceable(name="document_ingest_text")
    def ingest_text(self, text: str, source: str) -> dict[str, Any]:
        """Ingest raw text under a given source name."""
        ids = ingest_text(text, source=source, settings=self.settings, store=self.store)
        return {"source": source, "chunks": len(ids), "total_chunks": self.store.count()}

    def list_documents(self) -> list[dict[str, Any]]:
        """List ingested sources with chunk counts."""
        return self.store.list_sources()

    def delete_document(self, source: str) -> int:
        """Delete all chunks of a document. Returns the count deleted."""
        deleted = self.store.delete_by_source(source)
        logger.info("Deleted %d chunks for source %r", deleted, source)
        return deleted

    def search_preview(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Run a raw semantic search (debug/inspection endpoint)."""
        return self.store.search(query, top_k=top_k)