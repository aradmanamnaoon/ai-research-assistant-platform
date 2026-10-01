"""Document ingestion pipeline: load -> split -> embed -> store.

Supports ``.md``, ``.txt``, ``.csv``, ``.json`` text files. PDF support is
available when ``pypdf`` is installed (declared in requirements.txt).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.retrieval.splitter import default_splitter
from app.retrieval.vector_store import VectorStore

logger = get_logger("retrieval.ingest")

_TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".rst", ".html"}
_SUPPORTED = _TEXT_SUFFIXES | {".pdf"}

_MAX_DOC_BYTES = 20 * 1024 * 1024


def load_file(path: Path) -> list[Document]:
    """Load a single file into one :class:`Document` (no splitting yet).

    Args:
        path: File path to load.

    Returns:
        A one-element list of documents, or an empty list if the file
        type is unsupported or empty.

    Raises:
        ValueError: On unsupported/oversized files.
    """
    if path.suffix.lower() not in _SUPPORTED:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: {sorted(_SUPPORTED)}"
        )
    if path.stat().st_size > _MAX_DOC_BYTES:
        raise ValueError(f"File too large ({path.stat().st_size} bytes)")

    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)

    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        raw = _flatten_json(raw)
    elif path.suffix.lower() == ".html":
        raw = _strip_html(raw)

    return [Document(page_content=raw, metadata={"source": path.name, "title": path.stem})]


def ingest_text(
    text: str,
    source: str,
    metadata: dict | None = None,
    settings: Settings | None = None,
    store: VectorStore | None = None,
) -> list[str]:
    """Split ``text``, embed and upsert it into the vector store.

    Args:
        text: Raw document text.
        source: Stable source identifier (usually the file name).
        metadata: Extra metadata merged into every chunk.
        settings: Optional settings override.
        store: Optional pre-built vector store.

    Returns:
        The list of stored chunk ids.
    """
    settings = settings or get_settings()
    store = store or VectorStore(settings=settings)
    splitter = default_splitter(settings.chunk_size, settings.chunk_overlap)

    base_metadata = {"source": source, **(metadata or {})}
    chunks = splitter.split_with_metadata(text, base_metadata)
    documents = [
        Document(
            page_content=chunk.text,
            metadata={**chunk.metadata, "chunk_index": chunk.index},
        )
        for chunk in chunks
    ]
    return store.add_documents(documents)


def ingest_directory(
    directory: Path,
    settings: Settings | None = None,
    store: VectorStore | None = None,
) -> dict[str, list[str]]:
    """Ingest every supported file in ``directory`` (non-recursive).

    Args:
        directory: Directory containing example documents.
        settings: Optional settings override.
        store: Optional pre-built vector store.

    Returns:
        Mapping of file name -> stored chunk ids.
    """
    settings = settings or get_settings()
    store = store or VectorStore(settings=settings)
    results: dict[str, list[str]] = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED:
            continue
        try:
            docs = load_file(path)
            ids = ingest_text(
                docs[0].page_content,
                source=path.name,
                metadata=docs[0].metadata,
                settings=settings,
                store=store,
            )
            results[path.name] = ids
            logger.info("Ingested %s (%d chunks)", path.name, len(ids))
        except (OSError, ValueError) as exc:
            logger.error("Failed to ingest %s: %s", path.name, exc)
    return results


def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF using ``pypdf`` when installed."""
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ValueError(
            "PDF ingestion requires pypdf (declared in requirements.txt)."
        ) from exc
    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return [Document(page_content=text, metadata={"source": path.name, "title": path.stem})]


def _flatten_json(raw: str, max_depth: int = 4) -> str:
    """Flatten a JSON document into readable key-value lines."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    lines: list[str] = []

    def walk(node: Any, prefix: str, depth: int) -> None:  # type: ignore[name-defined]
        if depth > max_depth:
            lines.append(str(node)[:2000])
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else key, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, prefix, depth + 1)
        else:
            text = str(node).replace("\n", " ")
            if text.strip():
                lines.append(f"{prefix}: {text}")

    walk(data, "", 0)
    return "\n".join(lines)


def _strip_html(raw: str) -> str:
    """Rudimentary HTML tag stripping."""
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()