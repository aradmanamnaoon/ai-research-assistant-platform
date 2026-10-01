"""Tool registry: single source of truth for agent tools.

The researcher agent binds the web-research tools; the tool-execution
agent resolves tools by name for planner-requested direct execution.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.web import tavily_search, wikipedia_search
from app.tools.arxiv import arxiv_search
from app.tools.local import calculator, list_notes, save_note

# Tools the researcher agent may call autonomously in its tool loop.
RESEARCH_TOOLS: list[BaseTool] = [wikipedia_search, arxiv_search, tavily_search]

# Tools executable directly when the planner requests them by name.
EXECUTION_TOOLS: dict[str, BaseTool] = {
    tool.name: tool
    for tool in [wikipedia_search, arxiv_search, tavily_search, calculator, save_note, list_notes]
}


def get_all_tools() -> list[BaseTool]:
    """Return all tools available to the agents."""
    return list(RESEARCH_TOOLS) + [t for t in EXECUTION_TOOLS.values() if t not in RESEARCH_TOOLS]


def get_tool(name: str) -> BaseTool | None:
    """Look up a tool by name.

    Args:
        name: Tool name, e.g. ``"wikipedia_search"``.

    Returns:
        The tool, or ``None`` when the name is unknown.
    """
    return EXECUTION_TOOLS.get(name)


def get_tool_catalog() -> list[dict[str, str]]:
    """Return a JSON-friendly catalog for API introspection."""
    return [
        {"name": tool.name, "description": tool.description.split("\n")[0]}
        for tool in get_all_tools()
    ]