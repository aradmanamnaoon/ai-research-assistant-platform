"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    question: str = Field(..., min_length=1, max_length=8000, examples=["What is retrieval-augmented generation?"])
    thread_id: str | None = Field(
        default=None,
        max_length=64,
        description="Conversation thread id; a new one is generated when omitted.",
    )


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    thread_id: str
    question: str
    answer: str
    citations: list[str] = []
    sources: list[str] = []
    plan: dict[str, Any] = {}
    steps_taken: dict[str, Any] = {}
    error: str | None = None
    completed_at: str | None = None


class StreamEvent(BaseModel):
    """One server-sent event from the streaming chat endpoint."""

    event: str = Field(..., examples=["node_update"])
    node: str | None = None
    data: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None


class IngestTextRequest(BaseModel):
    """Request body for ingesting raw text."""

    text: str = Field(..., min_length=1, max_length=2_000_000)
    source: str = Field(..., min_length=1, max_length=256, examples=["notes.md"])


class IngestResult(BaseModel):
    """Result of a successful ingestion."""

    source: str
    chunks: int
    total_chunks: int


class DocumentInfo(BaseModel):
    """Summary of one ingested document."""

    source: str
    chunks: int


class SearchPreviewRequest(BaseModel):
    """Request body for the retrieval preview endpoint."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class HistoryMessage(BaseModel):
    """One stored conversation turn."""

    id: int
    thread_id: str
    role: str
    content: str
    created_at: str


class ThreadSummary(BaseModel):
    """Summary of one conversation thread."""

    thread_id: str
    message_count: int
    last_activity: str | None = None


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    version: str
    llm_provider: str
    model_name: str
    documents_indexed: int
    langsmith_enabled: bool