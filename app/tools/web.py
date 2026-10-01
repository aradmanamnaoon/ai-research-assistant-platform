"""Web research tools.

``wikipedia_search`` uses the public MediaWiki Action API (no key).
``tavily_search`` calls the Tavily REST API when ``TAVILY_API_KEY`` is
configured, and fails with an actionable message otherwise.
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("tools.web")

_HTTP_TIMEOUT = 15.0
# Wikimedia's robot policy requires a descriptive UA with contact info.
_USER_AGENT = (
    "AIResearchAssistant/1.0 "
    "(https://github.com/ai-research-assistant; contact: research@example.com) "
    "httpx"
)

_WIKIPEDIA_HEADERS = {"User-Agent": _USER_AGENT}


@tool
def wikipedia_search(query: str, limit: int = 3) -> str:
    """Search Wikipedia and return short extracts for the top results.

    Use this for encyclopedic, factual, historical or general-knowledge
    questions. Returns up to ``limit`` article extracts.

    Args:
        query: Free-text search query.
        limit: Maximum number of results (1-5).

    Returns:
        Formatted search results with titles, URLs and extracts.
    """
    limit = max(1, min(limit, 5))
    logger.info("wikipedia_search: %r", query)

    with httpx.Client(timeout=_HTTP_TIMEOUT, headers=_WIKIPEDIA_HEADERS, follow_redirects=True) as client:
        search_resp = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
                "origin": "*",
            },
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])
        if not hits:
            return f"No Wikipedia results found for: {query}"

        titles = [hit["title"] for hit in hits]
        extracts_resp = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "titles": "|".join(titles),
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
                "origin": "*",
            },
        )
        extracts_resp.raise_for_status()
        extracts = extracts_resp.json().get("query", {}).get("pages", {})

    lines: list[str] = []
    for hit in hits:
        page = extracts.get(str(hit["pageid"]), {})
        title = hit.get("title", "")
        snippet = (page.get("extract") or hit.get("snippet", "")).strip()
        if snippet:
            snippet = " ".join(snippet.split())[:800]
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            lines.append(f"[Wikipedia] {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(lines) if lines else f"No extracts available for: {query}"


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the live web via the Tavily API for recent information.

    Use this for news, current events, prices, or anything that may be
    too recent for encyclopedic sources.

    Args:
        query: Web search query.
        max_results: Maximum number of results to return.

    Returns:
        Formatted web results (title, URL, snippet) or an error message
        explaining that TAVILY_API_KEY must be configured.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        return (
            "Tavily web search is unavailable: TAVILY_API_KEY is not set. "
            "Configure it in .env, or rely on wikipedia_search / arxiv_search."
        )

    logger.info("tavily_search: %r", query)
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        response = client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={"query": query, "max_results": max_results},
        )
        response.raise_for_status()
        payload = response.json()

    results = payload.get("results", [])
    if not results:
        return f"No web results found for: {query}"

    lines = []
    for item in results[:max_results]:
        lines.append(
            f"[Web] {item.get('title', 'Untitled')}\n"
            f"URL: {item.get('url', 'n/a')}\n"
            f"{item.get('content', '')[:500]}"
        )
    return "\n\n".join(lines)