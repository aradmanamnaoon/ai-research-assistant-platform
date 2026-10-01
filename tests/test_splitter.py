"""Tests for the recursive text splitter."""

from __future__ import annotations

import pytest

from app.retrieval.splitter import RecursiveCharacterSplitter


def test_short_text_single_chunk() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text("hello world")
    assert chunks == ["hello world"]


def test_long_text_chunk_sizes() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=10)
    text = "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(6))
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks), "chunks should roughly respect chunk_size"


def test_paragraph_boundaries_preferred() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=120, chunk_overlap=10)
    text = "\n\n".join(
        f"PARA{i} " + "content " * 14 for i in range(3)
    )
    chunks = splitter.split_text(text)
    assert len(chunks) == 3
    assert all(f"PARA{i}" in chunks[i] for i in range(3))


def test_overlap_carries_context() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=60, chunk_overlap=20)
    text = " ".join(f"sentence{i}" for i in range(30))
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    # Overlap: the tail of chunk 0 should appear in chunk 1.
    assert chunks[0][-15:] in chunks[1]


def test_code_block_preserved() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=5)
    code = "```python\ndef f():\n    return 1 + 2\n```"
    text = f"Intro text.\n\n{code}\n\nOutro."
    chunks = splitter.split_text(text)
    joined = "\n".join(chunks)
    assert "def f():" in joined
    assert "return 1 + 2" in joined


def test_empty_and_whitespace_only() -> None:
    splitter = RecursiveCharacterSplitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("   \n\n  ") == []


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 150)],
)
def test_invalid_parameters_raise(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_split_with_metadata_attaches_fields() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=5)
    chunks = splitter.split_with_metadata("x " * 100, metadata={"source": "a.md"})
    assert all(c.metadata["source"] == "a.md" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))