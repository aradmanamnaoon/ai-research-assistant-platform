"""Academic paper search tool backed by the public arXiv Atom API (no key)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

from app.logging_config import get_logger

logger = get_logger("tools.arxiv")

_HTTP_TIMEOUT = 20.0
_ARXIV_NS = "{http://www.w3.org/2005/Atom}"
_USER_AGENT = "AI-Research-Assistant/1.0 (portfolio project; mailto:research@example.com)"

try:  # Prefer the hardened parser when available (see requirements.txt).
    import defusedxml.ElementTree as DET  # type: ignore[import-untyped]

    _parse_xml = DET.fromstring
except ImportError:  # pragma: no cover - exercised only without defusedxml
    logger.warning(
        "defusedxml is not installed; using stdlib ElementTree with a "
        "DOCTYPE guard. Install defusedxml for full XXE protection."
    )

    def _parse_xml(text: str) -> ET.Element:
        """Parse XML after rejecting entity declarations (XXE guard)."""
        head = text[:4096].upper()
        if "<!DOCTYPE" in head or "<!ENTITY" in head:
            raise ET.ParseError("DOCTYPE/entity declarations are not allowed")
        return ET.fromstring(text)


@tool
def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search arXiv for academic papers matching a research query.

    Use this for scientific literature, ML/AI papers, formal definitions,
    or when a question requires academic sources.

    Args:
        query: arXiv search query (supports field prefixes like ``ti:``
            and ``abs:``).
        max_results: Maximum number of papers to return (1-10).

    Returns:
        Formatted paper entries with title, authors, URL and abstract.
    """
    max_results = max(1, min(max_results, 10))
    logger.info("arxiv_search: %r", query)

    with httpx.Client(timeout=_HTTP_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as client:
        response = client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            },
        )
        response.raise_for_status()

    try:
        root = _parse_xml(response.text)
    except ET.ParseError:
        logger.warning("arxiv_search returned invalid XML for %r", query)
        return f"arXiv returned an invalid response for: {query}"

    entries: list[str] = []
    for entry in root.findall(f"{_ARXIV_NS}entry"):
        title = (entry.findtext(f"{_ARXIV_NS}title") or "").strip().replace("\n", " ")
        url = (entry.findtext(f"{_ARXIV_NS}id") or "").strip()
        authors = [
            (a.findtext(f"{_ARXIV_NS}name") or "").strip()
            for a in entry.findall(f"{_ARXIV_NS}author")
        ]
        summary = (entry.findtext(f"{_ARXIV_NS}summary") or "").strip()
        summary = " ".join(summary.split())[:700]
        published = (entry.findtext(f"{_ARXIV_NS}published") or "")[:10]
        author_str = ", ".join(a for a in authors if a)[:120] or "Unknown"
        entries.append(
            f"[arXiv] {title}\nPublished: {published}\n"
            f"Authors: {author_str}\nURL: {url}\n{summary}"
        )

    return "\n\n".join(entries) if entries else f"No arXiv papers found for: {query}"