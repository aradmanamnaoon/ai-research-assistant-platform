"""Research agent: autonomous tool-calling loop over web/academic sources.

Wraps ``langgraph.prebuilt.create_react_agent`` so the LLM decides when
and which tools to call (ReAct pattern), while the outer research graph
treats the whole loop as a single node.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from app.agents.prompts import SYSTEM_RESEARCHER
from app.logging_config import get_logger
from app.tools.registry import RESEARCH_TOOLS

logger = get_logger("agents.researcher")


def build_researcher(
    model: BaseChatModel,
    tools: list | None = None,
) -> Any:
    """Compile the researcher sub-agent.

    Args:
        model: Chat model used inside the tool loop.
        tools: Tools the agent may call (defaults to the registry set).

    Returns:
        A compiled LangGraph runnable accepting ``{"messages": [...]}``.
    """
    return create_react_agent(
        model,
        tools=tools or RESEARCH_TOOLS,
        prompt=SYSTEM_RESEARCHER,
    )


def research_node(state: dict[str, Any], config: RunnableConfig, researcher: Any) -> dict[str, Any]:
    """Graph node: run the researcher agent for the current question.

    Args:
        state: Current graph state.
        config: LangGraph runnable config.
        researcher: Compiled researcher agent (injected at graph build time).

    Returns:
        Partial state update appending to ``research_notes``.
    """
    question = state["question"]
    plan = state.get("plan", {})
    retrieved_docs = state.get("retrieved_docs", [])
    prior_notes = state.get("research_notes", [])
    missing = state.get("missing_information", "")

    step = plan.get("research_query") or question
    if plan.get("steps"):
        step = f"{step}\nResearch steps: {', '.join(plan['steps'])}"

    task = HumanMessage(
        content=(
            f"Research request: {step}\n\n"
            f"Original question: {question}\n"
            + (
                f"\nGaps found in the previous draft: {missing}\n"
                if missing
                else ""
            )
            + (
                f"\nDocument passages already retrieved:\n{_format_docs(retrieved_docs)}\n"
                if retrieved_docs
                else ""
            )
            + (
                f"\nFindings already gathered (do NOT repeat these searches):\n"
                f"{_format_notes(prior_notes)}\n"
                if prior_notes
                else ""
            )
        )
    )

    try:
        result = researcher.invoke(
            {"messages": [task]},
            config={"recursion_limit": 3 + 2 * int(state.get("max_tool_calls", 6))},
        )
        messages: list = result.get("messages", [])
        findings = next(
            (
                m.content
                for m in reversed(messages)
                if isinstance(m, AIMessage) and (m.content or "").strip()
            ),
            "",
        )
        tool_calls_made = [
            call["name"]
            for m in messages
            if isinstance(m, AIMessage)
            for call in (m.tool_calls or [])
        ]
    except Exception as exc:  # noqa: BLE001 - degrade to a note, never crash
        logger.error("Researcher failed: %s", exc)
        findings = (
            f"Research step failed: {exc}. Continue with the remaining context."
        )
        tool_calls_made = []

    note = {
        "query": step,
        "findings": findings,
        "tools_used": tool_calls_made,
    }
    logger.info("Researcher produced %d chars of findings", len(findings))
    return {"research_notes": [note]}


def _format_docs(docs: list[dict[str, Any]]) -> str:
    """Render retrieved document dicts for the researcher prompt."""
    return "\n".join(
        f"- [{d.get('source', 'doc')}] {d.get('text', '')[:400]}"
        for d in docs
    )


def _format_notes(notes: list[dict[str, Any]]) -> str:
    """Render prior research notes for the researcher prompt."""
    return "\n".join(
        f"- Query: {n.get('query', '')}: {n.get('findings', '')[:400]}"
        for n in notes
    )