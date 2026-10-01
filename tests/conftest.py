"""Shared pytest fixtures: isolated settings, fake model, seeded stores."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.fake import SmartFakeChatModel
from app.memory.conversations import ConversationStore
from app.memory.sqlite_checkpointer import SqliteCheckpointSaver
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.vector_store import VectorStore


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing at a temp data dir with the fake provider."""
    s = Settings(
        llm_provider="fake",
        data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "checkpoints.sqlite",
        notes_path=tmp_path / "notes" / "notes.jsonl",
        langsmith_tracing=False,
        environment="development",
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def vector_store(settings) -> VectorStore:
    """An empty vector store backed by the temp directory."""
    return VectorStore(settings=settings)


@pytest.fixture
def seeded_store(vector_store, settings) -> VectorStore:
    """Vector store pre-loaded with two small knowledge documents."""
    from app.retrieval.ingest import ingest_text

    ingest_text(
        "Retrieval-Augmented Generation (RAG) grounds LLM answers in retrieved "
        "documents. It reduces hallucination and supports private knowledge bases.",
        source="rag.md",
        settings=settings,
        store=vector_store,
    )
    ingest_text(
        "LangGraph persists agent state with checkpointers. A custom checkpointer "
        "implements get_tuple, list, put and put_writes.",
        source="langgraph.md",
        settings=settings,
        store=vector_store,
    )
    return vector_store


@pytest.fixture
def retriever(seeded_store) -> DocumentRetriever:
    """Retriever over the seeded store."""
    return DocumentRetriever(vector_store=seeded_store, top_k=3)


@pytest.fixture
def checkpointer() -> SqliteCheckpointSaver:
    """In-memory SQLite checkpointer."""
    return SqliteCheckpointSaver(":memory:")


@pytest.fixture
def graph(settings, checkpointer, retriever):
    """Compiled research graph with fake model and isolated stores."""
    from app.graphs.research_graph import build_research_graph

    return build_research_graph(
        model=SmartFakeChatModel(),
        settings=settings,
        checkpointer=checkpointer,
        retriever=retriever,
    )


@pytest.fixture
def conversation_store(settings) -> ConversationStore:
    """Conversation store backed by the temp SQLite file."""
    return ConversationStore(settings=settings)


@pytest.fixture
def research_service(graph, conversation_store, settings):
    """ResearchService wired to the isolated graph and stores."""
    from app.services.research_service import ResearchService

    return ResearchService(
        graph=graph, conversations=conversation_store, settings=settings
    )


@pytest.fixture
def document_service(seeded_store, settings):
    """DocumentService wired to the seeded store."""
    from app.services.document_service import DocumentService

    return DocumentService(store=seeded_store, settings=settings)


@pytest.fixture
def client(research_service, document_service):
    """FastAPI TestClient with overridden service dependencies."""
    from fastapi.testclient import TestClient

    from app.api.deps import get_document_service, get_research_service
    from app.api.main import app

    app.dependency_overrides[get_research_service] = lambda: research_service
    app.dependency_overrides[get_document_service] = lambda: document_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()