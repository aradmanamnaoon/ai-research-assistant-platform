"""Tool execution agent: runs planner-requested tools and interprets results."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.agents.llm_utils import invoke_llm
from app.agents.prompts import TOOL_AGENT_PROMPT
from app.logging_config import get_logger
from app.tools.registry import get_tool

logger = get_logger("agents.tool_agent")


def tool_agent_node(
    state: dict[str, Any],
    config: RunnableConfig,
    model: BaseChatModel,
) -> dict[str, Any]:
    """Graph node: execute the tools requested in the plan.

    Tool execution errors are captured per-call so one failing tool never
    aborts the pipeline.

    Args:
        state: Current graph state.
        config: LangGraph runnable config.
        model: Chat model used to interpret the results.

    Returns:
        Partial state update appending to ``tool_results``.
    """
    question = state["question"]
    plan = state.get("plan", {})
    requests = plan.get("tools_to_run", [])

    results: list[dict[str, Any]] = []
    for request in requests:
        name = request.get("name", "")
        args = request.get("args", {}) or {}
        tool = get_tool(name)
        if tool is None:
            results.append(
                {"tool": name, "status": "error", "result": f"Unknown tool: {name}"}
            )
            continue
        try:
            output = tool.invoke(args)
            results.append({"tool": name, "status": "ok", "result": str(output)[:2000]})
            logger.info("Tool %s executed for %r", name, question[:40])
        except Exception as exc:  # noqa: BLE001 - isolate tool failures
            logger.error("Tool %s failed: %s", name, exc)
            results.append(
                {"tool": name, "status": "error", "result": str(exc)[:500]}
            )

    if not results:
        return {"tool_results": []}

    formatted = _format_results(results)
    messages = TOOL_AGENT_PROMPT.invoke(
        {"question": question, "results": formatted}
    ).to_messages()
    try:
        response = invoke_llm(model, messages)
        interpretation = (response.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Tool-result interpretation failed: %s", exc)
        interpretation = "Tool results available but interpretation failed."

    results.append({"tool": "interpretation", "status": "ok", "result": interpretation})
    return {"tool_results": results}


def _format_results(results: list[dict[str, Any]]) -> str:
    """Render tool results for the interpretation prompt."""
    return "\n\n".join(
        f"Tool: {r['tool']} ({r['status']})\n{r['result']}" for r in results
    )