"""Synthesizer (final response agent): writes the cited final answer."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.agents.llm_utils import invoke_llm
from app.agents.prompts import SYNTHESIZER_PROMPT
from app.logging_config import get_logger

logger = get_logger("agents.synthesizer")

_CITATION_RE = re.compile(r"\[(\d+)\]")


def synthesizer_node(
    state: dict[str, Any],
    config: RunnableConfig,
    model: BaseChatModel,
) -> dict[str, Any]:
    """Graph node: synthesize the final cited answer.

    Assembles numbered sources from retrieved documents, research notes
    and tool results, prompts the model to cite them inline, and parses
    the citations back out of the produced answer.

    Args:
        state: Current graph state.
        config: LangGraph runnable config.
        model: Chat model used for synthesis.

    Returns:
        Partial state update with ``draft_answer``, ``final_answer``,
        ``citations`` and ``sources``.
    """
    question = state["question"]
    plan = state.get("plan", {})

    sources = _collect_sources(state)
    context = SYNTHESIZER_PROMPT.invoke(
        {
            "question": question,
            "intent": plan.get("reasoning", "not specified"),
            "doc_context": _render_docs(state),
            "research_findings": _render_notes(state),
            "tool_results": _render_tools(state),
        }
    ).to_messages()

    try:
        response = invoke_llm(model, context)
        draft = (response.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - synthesize from context on failure
        logger.error("Synthesis LLM call failed: %s", exc)
        draft = _fallback_answer(question, state)
        error = str(exc)
    else:
        error = None

    cited_numbers = sorted({int(m) for m in _CITATION_RE.findall(draft)})
    citations = [sources[n - 1] for n in cited_numbers if 0 < n <= len(sources)]
    sources_block = _sources_block(sources) if draft else ""

    logger.info("Synthesized answer (%d chars, %d citations)", len(draft), len(citations))
    return {
        "draft_answer": draft,
        "final_answer": f"{draft}\n\n{sources_block}".strip(),
        "citations": citations,
        "sources": sources,
        "error": error,
    }


def _collect_sources(state: dict[str, Any]) -> list[str]:
    """Build the ordered source list referenced by [n] citations."""
    sources: list[str] = []
    for doc in state.get("retrieved_docs", []):
        if doc.get("source") == "retrieval_summary":
            continue
        label = f"Uploaded document: {doc.get('source', 'unknown')}"
        if label not in sources:
            sources.append(label)
    for note in state.get("research_notes", []):
        for url in re.findall(r"https?://\S+", note.get("findings", "")):
            url = url.rstrip(").,;")
            if url not in sources:
                sources.append(url)
    for result in state.get("tool_results", []):
        if result.get("tool") in ("wikipedia_search", "arxiv_search", "tavily_search"):
            label = f"Tool result: {result['tool']}"
            if label not in sources:
                sources.append(label)
    return sources


def _render_docs(state: dict[str, Any]) -> str:
    """Render retrieved documents, labeled with their source index."""
    sources = _collect_sources(state)
    lines: list[str] = []
    for doc in state.get("retrieved_docs", []):
        if doc.get("source") == "retrieval_summary":
            lines.append(f"(Retriever summary) {doc.get('text', '')}")
            continue
        label = f"Uploaded document: {doc.get('source', 'unknown')}"
        index = sources.index(label) + 1 if label in sources else "?"
        lines.append(f"[{index}] ({doc.get('source', '?')}) {doc.get('text', '')[:900]}")
    return "\n\n".join(lines) or "(no documents retrieved)"


def _render_notes(state: dict[str, Any]) -> str:
    """Render research findings for the synthesis prompt."""
    return "\n\n".join(
        f"Query: {note.get('query', '')}\nFindings: {note.get('findings', '')}"
        for note in state.get("research_notes", [])
    ) or "(no research findings)"


def _render_tools(state: dict[str, Any]) -> str:
    """Render tool results for the synthesis prompt."""
    return "\n".join(
        f"- {result.get('tool')} ({result.get('status')}): {result.get('result', '')}"
        for result in state.get("tool_results", [])
    ) or "(no tool results)"


def _sources_block(sources: list[str]) -> str:
    """Format the trailing ``Sources:`` block."""
    if not sources:
        return ""
    lines = [f"[{i + 1}] {source}" for i, source in enumerate(sources)]
    return "Sources:\n" + "\n".join(lines)


def _fallback_answer(question: str, state: dict[str, Any]) -> str:
    """Context-only answer used when the synthesis LLM call fails."""
    parts = [f"I could not complete full synthesis for: {question}"]
    notes = state.get("research_notes", [])
    if notes:
        parts.append(f"Partial research findings:\n{notes[0].get('findings', '')}")
    tools = [t for t in state.get("tool_results", []) if t.get("status") == "ok"]
    if tools:
        parts.append(f"Tool results: {tools[0].get('result', '')}")
    return "\n\n".join(parts)