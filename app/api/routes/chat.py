"""Chat routes: synchronous, streaming (SSE), and history."""

from __future__ import annotations

import json
import uuid
from typing import Any, Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_research_service
from app.api.schemas import ChatRequest, ChatResponse, HistoryMessage, ThreadSummary
from app.services.research_service import ResearchService
from app.logging_config import get_logger

logger = get_logger("api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: ResearchService = Depends(get_research_service),
) -> ChatResponse:
    """Ask a research question and get the fully synthesized answer.

    Runs the complete multi-agent pipeline: plan -> retrieve/research/
    tools (parallel when requested) -> synthesize -> quality gate.
    """
    thread_id = request.thread_id or uuid.uuid4().hex[:12]
    history_messages = service.conversation_messages(thread_id)
    try:
        payload = service.ask(question=request.question, thread_id=thread_id, history=history_messages)
    except Exception as exc:  # noqa: BLE001 - surface as HTTP error
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail=f"Research pipeline failed: {exc}") from exc
    return ChatResponse(**payload)


@router.post("/stream")
def chat_stream(
    request: ChatRequest,
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    """Ask a research question, streaming per-node progress via SSE.

    Each agent node emits one ``node_update`` event as it finishes; the
    final ``done`` event carries the complete response payload.
    """
    thread_id = request.thread_id or uuid.uuid4().hex[:12]
    history_messages = service.conversation_messages(thread_id)

    def event_source() -> Generator[str, None, None]:
        try:
            for event in service.stream(
                question=request.question, thread_id=thread_id, history=history_messages
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat failed")
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(exc)}})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{thread_id}/history", response_model=list[HistoryMessage])
def history(
    thread_id: str,
    limit: int = 50,
    offset: int = 0,
    service: ResearchService = Depends(get_research_service),
) -> list[HistoryMessage]:
    """Return the stored turns of one conversation thread."""
    return [
        HistoryMessage(**message)
        for message in service.history(thread_id, limit=limit, offset=offset)
    ]


@router.get("/threads", response_model=list[ThreadSummary])
def list_threads(
    service: ResearchService = Depends(get_research_service),
) -> list[ThreadSummary]:
    """List all conversation threads with activity summaries."""
    return [ThreadSummary(**summary) for summary in service.list_threads()]


@router.delete("/{thread_id}")
def delete_thread(
    thread_id: str,
    service: ResearchService = Depends(get_research_service),
) -> dict[str, Any]:
    """Delete a conversation thread's stored turns."""
    deleted = service.delete_thread(thread_id)
    return {"thread_id": thread_id, "deleted_messages": deleted}