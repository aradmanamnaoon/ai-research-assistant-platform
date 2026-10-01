"""Graph state schema for the research pipeline.

State is the shared memory of the whole multi-agent workflow. List fields
use ``operator.add`` reducers so parallel branches (retriever running
concurrently with the researcher) merge their outputs instead of
overwriting each other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    """State flowing through the research graph.

    Attributes:
        question: The user's question for this turn.
        history: Prior conversation turns for context.
        plan: Planner output (see :class:`~app.agents.planner.ResearchPlan`).
        retrieved_docs: Document passages fetched by the retrieval agent.
        research_notes: Findings produced by the research agent.
        tool_results: Results from the tool execution agent.
        draft_answer: Pre-citation answer draft.
        final_answer: The answer returned to the user (with sources).
        citations: Source labels cited in the final answer.
        sources: Ordered source list; citation ``[n]`` points here.
        missing_information: Gap analysis output driving the refinement loop.
        iteration: Refinement-loop counter (0 = first pass).
        max_tool_calls: Cap forwarded to the researcher sub-agent.
        error: Last non-fatal error, if any.
    """

    question: str
    history: list[BaseMessage]
    plan: dict[str, Any]
    retrieved_docs: Annotated[list[dict[str, Any]], operator.add]
    research_notes: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    draft_answer: str
    final_answer: str
    citations: list[str]
    sources: list[str]
    missing_information: str
    iteration: int
    max_tool_calls: int
    error: str | None