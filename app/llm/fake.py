"""Deterministic fake chat model for tests and offline demos.

:class:`SmartFakeChatModel` inspects the conversation to produce a
plausible plan / tool call / research note / synthesis. It is used when
``LLM_PROVIDER=fake`` so the full platform can run with zero API keys.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class SmartFakeChatModel(BaseChatModel):
    """Context-aware fake that drives the whole research graph offline."""

    bound_tools: list[dict[str, Any]] = []
    _call_counter: int = 0

    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain for run metadata."""
        return "smart-fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        **kwargs: Any,
    ) -> Any:
        """Record bound tools and return a bound clone."""
        clone = self.model_copy()
        clone.bound_tools = [convert_to_openai_tool(t) for t in tools]
        return clone

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        self._call_counter += 1
        system = next((m.content for m in messages if m.type == "system"), "")
        last = messages[-1] if messages else None

        if str(system).strip().startswith("You are the research planner"):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=json.dumps(self._fake_plan())))]
            )
        if isinstance(last, ToolMessage):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=self._fake_research_note()))]
            )
        if self.bound_tools and self._call_counter % 2 == 1:
            names = [t["function"]["name"] for t in self.bound_tools]
            chosen = names[0]
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": chosen,
                                    "args": self._fake_tool_args(chosen),
                                    "id": f"fake_call_{self._call_counter}",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._fake_synthesis()))]
        )

    # ------------------------------------------------------------------ #
    # Canned outputs
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fake_plan() -> dict[str, Any]:
        return {
            "reasoning": "Deterministic offline plan",
            "needs_retrieval": True,
            "needs_web_research": True,
            "needs_tools": False,
            "tools_to_run": [],
            "steps": ["retrieve documents", "search wikipedia", "synthesize"],
        }

    @staticmethod
    def _fake_research_note() -> str:
        return "Fake research findings: the topic was located in a public source."

    @staticmethod
    def _fake_synthesis() -> str:
        return "This is a deterministic offline answer produced by SmartFakeChatModel."

    @staticmethod
    def _fake_tool_args(name: str) -> dict[str, Any]:
        if "wikipedia" in name or "search" in name:
            return {"query": "artificial intelligence"}
        if "arxiv" in name:
            return {"query": "language model agents"}
        return {"query": "research topic"}