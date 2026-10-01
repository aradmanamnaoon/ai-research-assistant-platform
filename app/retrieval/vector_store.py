"""Vector store wrapper around ChromaDB with local ONNX embeddings.

Uses Chroma's bundled default embedding function (all-MiniLM-L6-v2 via
onnxruntime), so document ingestion and retrieval work fully offline with
no external embedding API. The embedding backend is pluggable: pass a
``embedding_function`` to use sentence-transformers / OpenAI / etc.
"""

from __future__ import annotations

import threading
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger("retrieval.vector_store")

_COLLECTION_LOCK = threading.Lock()


class VectorStore:
    """Thin, typed wrapper over a persistent ChromaDB collection.

    Attributes:
        collection_name: Name of the underlying Chroma collection.
        top_k: Default number of neighbors returned by searches.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Create/open the persistent collection.

        Args:
            settings: Optional settings override.
            collection_name: Optional collection name override.
        """
        settings = settings or get_settings()
        self.collection_name = collection_name or settings.collection_name
        settings.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.top_k = settings.retrieval_top_k

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add LangChain ``Document`` objects and return their ids.

        Empty documents are skipped. Ids are deterministic per (source,
        chunk index) so re-ingesting the same file is idempotent.
        """
        ids: list[str] = []
        docs, metadatas, documents_texts = [], [], []
        for doc in documents:
            if not doc.page_content.strip():
                continue
            source = str(doc.metadata.get("source", "unknown"))
            index = int(doc.metadata.get("chunk_index", 0))
            metadata = {
                k: v
                for k, v in doc.metadata.items()
                if isinstance(v, (str, int, float, bool))
                and k not in ("chunk_index", "source")
            }
            doc_id = f"{source}::chunk-{index}"
            metadata["source"] = source
            metadata["chunk_index"] = index
            metadatas.append(metadata)
            documents_texts.append(doc.page_content)
            ids.append(doc_id)

        if not ids:
            return []

        with _COLLECTION_LOCK:
            # Upsert makes re-ingestion of the same source idempotent.
            self._collection.upsert(
                ids=ids,
                documents=documents_texts,
                metadatas=metadatas,
            )
        logger.info("Upserted %d chunks into '%s'", len(ids), self.collection_name)
        return ids

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search returning normalized result dicts.

        Args:
            query: Natural-language query.
            top_k: Override for the number of results.
            where: Optional Chroma metadata filter.

        Returns:
            List of ``{"text", "metadata", "distance", "score"}`` dicts,
            ordered by relevance (score = 1 - cosine distance).
        """
        top_k = top_k or self.top_k
        with _COLLECTION_LOCK:
            result = self._collection.query(
                query_texts=[query],
                n_results=max(1, min(top_k, self._collection.count() or 1)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        hits: list[dict[str, Any]] = []
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                    "score": max(0.0, 1.0 - float(distance)),
                }
            )
        return hits

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks belonging to a source file. Returns count."""
        with _COLLECTION_LOCK:
            before = self._collection.count()
            self._collection.delete(where={"source": {"$eq": source}})
            return before - self._collection.count()

    def list_sources(self) -> list[dict[str, Any]]:
        """List distinct sources with their chunk counts."""
        with _COLLECTION_LOCK:
            data = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in data.get("metadatas") or []:
            source = str(metadata.get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return [
            {"source": source, "chunks": n}
            for source, n in sorted(counts.items())
        ]

    def reset(self) -> None:
        """Drop every document in the collection (admin/testing helper)."""
        with _COLLECTION_LOCK:
            client_ids = self._collection.get()["ids"]
            if client_ids:
                self._collection.delete(ids=client_ids)


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return a :class:`VectorStore` for the configured collection."""
    return VectorStore(settings=settings)