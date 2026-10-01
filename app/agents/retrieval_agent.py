"""Retrieval agent: queries the document library and distills relevance.

This node combines semantic search over the vector store with an LLM
relevance pass, so downstream synthesis only sees passages that actually
bear on the question.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.agents.prompts import RETRIEVAL_PROMPT
from app.agents.llm_utils import invoke_llm
from app.logging_config import get_logger
from app.retrieval.retriever import build_retriever

logger = get_logger("agents.retrieval")


def retrieval_node(
    state: dict[str, Any],
    config: RunnableConfig,
    model: BaseChatModel,
    retriever: Any = None,
) -> dict[str, Any]:
    """Graph node: retrieve and filter documents for the question.

    Args:
        state: Current graph state.
        config: LangGraph runnable config.
        model: Chat model used for the relevance pass.
        retriever: Optional retriever override (tests).

    Returns:
        Partial state update appending to ``retrieved_docs``.
    """
    question = state["question"]
    plan = state.get("plan", {})
    queries = {question}
    if plan.get("research_query"):
        queries.add(plan["research_query"])

    retriever = retriever or build_retriever()
    passages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for query in queries:
        try:
            hits = retriever.invoke(query)
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
            logger.error("Retrieval failed for %r: %s", query[:60], exc)
            continue
        for hit in hits:
            key = (hit.metadata.get("source"), hit.page_content[:120])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            passages.append(
                {
                    "source": hit.metadata.get("source", "unknown"),
                    "text": hit.page_content,
                    "score": float(hit.metadata.get("score", 0.0)),
                }
            )

    if not passages:
        logger.info("No documents retrieved for %r", question[:60])
        return {"retrieved_docs": []}

    formatted = _format_passages(passages)
    messages = RETRIEVAL_PROMPT.invoke(
        {"question": question, "passages": formatted}
    ).to_messages()
    try:
        response = invoke_llm(model, messages)
        summary = (response.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval relevance pass failed: %s", exc)
        summary = "Relevance filtering unavailable; passages provided unfiltered."

    distill = {
        "source": "retrieval_summary",
        "text": summary,
        "score": 1.0,
    }
    logger.info("Retrieved %d passages for %r", len(passages), question[:60])
    return {"retrieved_docs": passages[:10] + [distill]}


def _format_passages(passages: list[dict[str, Any]]) -> str:
    """Render passages with source labels for the relevance prompt."""
    return "\n\n".join(
        f"Passage {i + 1} [source: {p['source']}] (score {p['score']:.2f}):\n{p['text'][:800]}"
        for i, p in enumerate(passages)
    )