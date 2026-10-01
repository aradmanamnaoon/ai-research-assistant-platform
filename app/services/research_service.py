"""Research service: the API-facing facade over the compiled graph.

Holds the compiled LangGraph runnable, the checkpointer and the
conversation store; provides synchronous and streaming execution plus
conversation-history helpers. All external calls are wrapped with
LangSmith ``@traceable`` for observability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator

from langchain_core.messages import BaseMessage
from langsmith import traceable

from app.config import Settings, get_settings
from app.graphs.research_graph import create_default_graph
from app.logging_config import get_logger
from app.memory.conversations import ConversationStore

logger = get_logger("services.research")


class ResearchService:
    """Facade for running the research graph.

    Attributes:
        graph: The compiled LangGraph runnable.
        conversations: Conversation history store.
    """

    def __init__(
        self,
        graph: Any | None = None,
        conversations: ConversationStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Assemble the service (graph is lazily built when omitted)."""
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.graph = graph or create_default_graph()
        self.conversations = conversations or ConversationStore(self.settings)

    @traceable(name="research_ask", run_type="chain")
    def ask(
        self,
        question: str,
        thread_id: str,
        history: list[BaseMessage] | None = None,
    ) -> dict[str, Any]:
        """Run the full research pipeline for one question.

        Args:
            question: The user's question.
            thread_id: Conversation thread id (checkpointer key).
            history: Optional prior messages for planner context.

        Returns:
            Response payload: answer, citations, sources, plan, steps,
            timing metadata.
        """
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        config = {"configurable": {"thread_id": thread_id}}
        input_state = {
            "question": question,
            "history": history or [],
            "iteration": 0,
            "missing_information": "",
            "max_tool_calls": self.settings.max_research_tool_calls,
        }
        final_state = self.graph.invoke(input_state, config=config)

        payload = self._payload_from_state(final_state, thread_id, started)
        self.conversations.add_message(thread_id, "user", question)
        self.conversations.add_message(
            thread_id, "assistant", payload["answer"]
        )
        return payload

    @traceable(name="research_stream", run_type="chain")
    def stream(
        self,
        question: str,
        thread_id: str,
        history: list[BaseMessage] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream graph progress as JSON-serializable event dicts.

        Yields one event per node update, then a final ``done`` event
        carrying the complete payload.
        """
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        config = {"configurable": {"thread_id": thread_id}}
        input_state = {
            "question": question,
            "history": history or [],
            "iteration": 0,
            "missing_information": "",
            "max_tool_calls": self.settings.max_research_tool_calls,
        }
        final_state: dict[str, Any] = {}
        for mode, chunk in self.graph.stream(
            input_state,
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node_name, update in (chunk or {}).items():
                    if node_name in ("__start__", "__end__"):
                        continue
                    yield {
                        "event": "node_update",
                        "node": node_name,
                        "data": _safe_update(update),
                    }
            else:
                final_state = chunk or {}

        payload = self._payload_from_state(final_state, thread_id, started)
        self.conversations.add_message(thread_id, "user", question)
        self.conversations.add_message(thread_id, "assistant", payload["answer"])
        yield {"event": "done", "payload": payload}

    # ------------------------------------------------------------------ #
    # History helpers
    # ------------------------------------------------------------------ #
    def history(
        self, thread_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Return conversation turns for a thread (oldest first)."""
        return self.conversations.get_history(thread_id, limit=limit, offset=offset)

    def conversation_messages(self, thread_id: str, limit: int = 10) -> list[BaseMessage]:
        """Return recent turns as LangChain messages for prompt context.

        Args:
            thread_id: Conversation identifier.
            limit: Maximum number of recent turns to include.

        Returns:
            Alternating ``HumanMessage`` / ``AIMessage`` list, oldest first.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        turns = self.conversations.get_history(thread_id, limit=limit)
        messages: list[BaseMessage] = []
        for turn in turns:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
        return messages[-limit:]

    def list_threads(self) -> list[dict[str, Any]]:
        """List all known conversation threads."""
        return self.conversations.list_threads()

    def delete_thread(self, thread_id: str) -> int:
        """Delete a thread's stored turns."""
        return self.conversations.delete_thread(thread_id)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _payload_from_state(
        self, state: dict[str, Any], thread_id: str, started: str
    ) -> dict[str, Any]:
        """Project final graph state into the API response payload."""
        plan = state.get("plan", {})
        return {
            "thread_id": thread_id,
            "question": state.get("question", ""),
            "answer": state.get("final_answer", ""),
            "citations": state.get("citations", []),
            "sources": state.get("sources", []),
            "plan": plan,
            "steps_taken": {
                "retrieved_docs": len(state.get("retrieved_docs", [])),
                "research_notes": len(state.get("research_notes", [])),
                "tool_results": len(state.get("tool_results", [])),
                "iterations": state.get("iteration", 0),
            },
            "error": state.get("error"),
            "completed_at": started,
        }


def _safe_update(update: dict[str, Any] | None) -> dict[str, Any]:
    """Make a node's state update JSON-serializable for SSE."""
    if not update:
        return {}
    safe: dict[str, Any] = {}
    for key, value in update.items():
        if key == "history":
            continue
        try:
            import json

            json.dumps(value)
            safe[key] = value
        except (TypeError, ValueError):
            safe[key] = str(value)[:500]
    return safe