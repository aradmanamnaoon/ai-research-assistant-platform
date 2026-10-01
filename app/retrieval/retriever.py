"""LangChain retriever interface over the vector store.

Implements ``BaseRetriever`` so the retrieval agent can be used anywhere
LangChain expects a retriever (chains, tools, ``Runnable`` composition).
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.retrieval.vector_store import VectorStore


class DocumentRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by the Chroma collection.

    Attributes:
        vector_store: The underlying :class:`~app.retrieval.vector_store.VectorStore`.
        top_k: Number of documents fetched per query.
    """

    vector_store: VectorStore
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Fetch relevant documents (sync path used by LangChain)."""
        hits = self.vector_store.search(query, top_k=self.top_k)
        return [
            Document(page_content=hit["text"], metadata=hit["metadata"])
            for hit in hits
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Fetch relevant documents (async path)."""
        return self._get_relevant_documents(
            query, run_manager=run_manager, **kwargs
        )


def build_retriever(
    vector_store: VectorStore | None = None,
    top_k: int | None = None,
) -> DocumentRetriever:
    """Factory for :class:`DocumentRetriever`.

    Args:
        vector_store: Optional pre-built store.
        top_k: Optional override for results per query.
    """
    from app.config import get_settings

    settings = get_settings()
    store = vector_store or VectorStore(settings=settings)
    return DocumentRetriever(vector_store=store, top_k=top_k or settings.retrieval_top_k)