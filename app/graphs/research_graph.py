"""Research graph: LangGraph orchestration of all agent roles.

Topology::

    START -> planner -> [retriever, tool_agent, researcher] (parallel,
    selected by plan) -> synthesizer -> reflector -> (researcher | END)

The planner's plan selects branches; compatible branches execute in
parallel (fan-out) and merge via ``operator.add`` reducers. The reflector
implements a bounded refinement loop: when the draft answer has obvious
gaps, the graph routes back to the researcher once per iteration, up to
``settings.max_graph_iterations``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.llm_utils import extract_json, invoke_llm
from app.agents.planner import plan_research
from app.agents.prompts import GAP_ANALYSIS_PROMPT
from app.agents.researcher import build_researcher, research_node
from app.agents.retrieval_agent import retrieval_node
from app.agents.synthesizer import synthesizer_node
from app.agents.tool_agent import tool_agent_node
from app.config import Settings, get_settings
from app.graphs.state import ResearchState
from app.logging_config import get_logger
from app.llm.factory import build_chat_model

logger = get_logger("graphs.research")


def reflect_node(
    state: dict[str, Any],
    config: RunnableConfig,
    model: BaseChatModel,
) -> dict[str, Any]:
    """Graph node: gap analysis on the draft answer.

    Args:
        state: Current graph state.
        config: LangGraph runnable config.
        model: Chat model used for the quality gate.

    Returns:
        Partial state update with ``missing_information`` and the
        incremented ``iteration`` counter.
    """
    draft = state.get("draft_answer", "")
    if not draft.strip():
        return {
            "missing_information": "No draft was produced; research is required.",
            "iteration": state.get("iteration", 0) + 1,
        }

    messages = GAP_ANALYSIS_PROMPT.invoke(
        {"question": state["question"], "draft": draft}
    ).to_messages()
    try:
        response = invoke_llm(model, messages)
        verdict = extract_json(response.content) or {}
        sufficient = bool(verdict.get("sufficient", True))
        missing = str(verdict.get("missing", ""))
    except Exception as exc:  # noqa: BLE001 - the gate must never crash the graph
        logger.error("Gap analysis failed (%s); accepting draft", exc)
        sufficient, missing = True, ""

    if sufficient:
        missing = ""
    logger.info("Gap analysis: sufficient=%s missing=%r", sufficient, missing[:80])
    return {
        "missing_information": missing,
        "iteration": state.get("iteration", 0) + 1,
    }


def route_from_plan(state: dict[str, Any]) -> list[str]:
    """Conditional edge: select parallel branches from the plan.

    Returns a *list* so LangGraph fans out to multiple nodes at once.
    """
    plan = state.get("plan", {})
    targets: list[str] = []
    if plan.get("needs_retrieval"):
        targets.append("retriever")
    if plan.get("needs_tools"):
        targets.append("tool_agent")
    if plan.get("needs_web_research") or not targets:
        targets.append("researcher")
    return list(dict.fromkeys(targets))


def route_after_reflection(
    state: dict[str, Any],
    settings: Settings | None = None,
) -> Literal["researcher", "__end__"]:
    """Conditional edge: refine once more or finish."""
    settings = settings or get_settings()
    missing = state.get("missing_information", "")
    iteration = state.get("iteration", 0)
    if settings.enable_loop_refinement and missing and iteration < settings.max_graph_iterations:
        return "researcher"
    return END


def build_research_graph(
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    retriever: Any = None,
) -> Any:
    """Assemble and compile the full research graph.

    Args:
        model: Chat model override (defaults to the settings factory).
        settings: Settings override.
        checkpointer: Optional checkpointer for persistent state.
        retriever: Optional retriever override (tests).

    Returns:
        The compiled LangGraph runnable.
    """
    settings = settings or get_settings()
    model = model or build_chat_model(settings)

    builder = StateGraph(ResearchState)

    builder.add_node("planner", partial(plan_research, model=model))
    builder.add_node(
        "retriever",
        partial(retrieval_node, model=model, retriever=retriever),
    )
    builder.add_node(
        "researcher",
        partial(research_node, researcher=build_researcher(model)),
    )
    builder.add_node("tool_agent", partial(tool_agent_node, model=model))
    builder.add_node("synthesizer", partial(synthesizer_node, model=model))
    builder.add_node("reflector", partial(reflect_node, model=model))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_from_plan,
        ["retriever", "tool_agent", "researcher", "synthesizer"],
    )
    builder.add_edge("retriever", "synthesizer")
    builder.add_edge("tool_agent", "synthesizer")
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "reflector")
    builder.add_conditional_edges("reflector", route_after_reflection, ["researcher", END])

    return builder.compile(checkpointer=checkpointer)


def create_default_graph() -> Any:
    """Build the graph with settings-driven defaults.

    Convenience factory used by the API layer and scripts.
    """
    settings = get_settings()
    settings.ensure_dirs()
    from app.memory.sqlite_checkpointer import SqliteCheckpointSaver

    checkpointer = SqliteCheckpointSaver(settings.sqlite_path)
    return build_research_graph(
        settings=settings,
        checkpointer=checkpointer,
    )