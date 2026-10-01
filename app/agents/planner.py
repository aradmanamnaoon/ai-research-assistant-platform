"""Planner agent: classifies the question and produces a research plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from app.agents.llm_utils import extract_json, invoke_llm
from app.agents.prompts import PLANNER_PROMPT
from app.logging_config import get_logger
from app.tools.registry import EXECUTION_TOOLS

logger = get_logger("agents.planner")


@dataclass
class ResearchPlan:
    """Structured result of the planning step.

    Attributes:
        reasoning: Why this plan was chosen.
        needs_retrieval: Query the uploaded document library.
        needs_web_research: Run web/academic searches.
        needs_tools: Execute specific utility tools.
        tools_to_run: Tools requested by name with arguments.
        steps: Ordered human-readable steps.
        research_query: Optimized search query for web/academic tools.
    """

    reasoning: str = ""
    needs_retrieval: bool = False
    needs_web_research: bool = False
    needs_tools: bool = False
    tools_to_run: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    research_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (stored in graph state)."""
        return {
            "reasoning": self.reasoning,
            "needs_retrieval": self.needs_retrieval,
            "needs_web_research": self.needs_web_research,
            "needs_tools": self.needs_tools,
            "tools_to_run": self.tools_to_run,
            "steps": self.steps,
            "research_query": self.research_query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchPlan:
        """Rebuild a plan from its dict form."""
        return cls(
            reasoning=data.get("reasoning", ""),
            needs_retrieval=bool(data.get("needs_retrieval", False)),
            needs_web_research=bool(data.get("needs_web_research", False)),
            needs_tools=bool(data.get("needs_tools", False)),
            tools_to_run=list(data.get("tools_to_run", []) or []),
            steps=list(data.get("steps", []) or []),
            research_query=data.get("research_query", ""),
        )


def parse_plan(text: str, question: str = "") -> ResearchPlan:
    """Parse an LLM planning response into a :class:`ResearchPlan`.

    Falls back to a keyword heuristic when the model output is not valid
    JSON, so the graph never hard-fails on planning.
    """
    data = extract_json(text or "")
    if data:
        plan = ResearchPlan.from_dict(data)
        plan.tools_to_run = [
            tool_call
            for tool_call in plan.tools_to_run
            if isinstance(tool_call, dict)
            and tool_call.get("name") in EXECUTION_TOOLS
        ]
        if not plan.research_query:
            plan.research_query = question[:200]
        return plan
    logger.warning("Plan JSON parse failed; using heuristic fallback")
    return heuristic_plan(question)


def heuristic_plan(question: str) -> ResearchPlan:
    """Rule-based plan used when LLM planning is unavailable or invalid."""
    lowered = question.lower()
    words = lowered.split()
    is_chitchat = len(words) <= 4 and any(
        greeting in lowered for greeting in ("hi", "hello", "hey", "thanks")
    )
    is_math = any(op in lowered for op in ("calculate", "compute"))
    needs_retrieval = any(
        term in lowered for term in ("my documents", "uploaded", "knowledge base", "the document")
    )
    return ResearchPlan(
        reasoning="Heuristic fallback plan (LLM plan unavailable)",
        needs_retrieval=needs_retrieval,
        needs_web_research=not is_chitchat and not needs_retrieval and not is_math,
        needs_tools=is_math and ("calculate" in lowered or "compute" in lowered),
        tools_to_run=[],
        steps=["answer directly"],
        research_query=question[:200],
    )


def plan_research(state: dict[str, Any], config: RunnableConfig, model: BaseChatModel) -> dict[str, Any]:
    """Graph node: produce a :class:`ResearchPlan` for the question.

    Args:
        state: Current graph state.
        config: LangGraph runnable config.
        model: Bound planner chat model (injected at graph build time).

    Returns:
        Partial state update with the ``plan`` key.
    """
    question = state["question"]
    history: list[BaseMessage] = state.get("history", [])

    messages = PLANNER_PROMPT.invoke(
        {"question": question, "history": history}
    ).to_messages()

    try:
        response = invoke_llm(model, messages)
        plan = parse_plan(response.content, question)
    except Exception as exc:  # noqa: BLE001 - planning must never crash the graph
        logger.error("Planner failed (%s); using heuristic plan", exc)
        plan = heuristic_plan(question)

    logger.info(
        "Plan for %r: retrieval=%s web=%s tools=%s",
        question[:60],
        plan.needs_retrieval,
        plan.needs_web_research,
        plan.needs_tools,
    )
    return {"plan": plan.to_dict()}