"""Character-aware recursive text splitting with markdown awareness.

``langchain-text-splitters`` is not installed in this environment, so this
module provides a small, dependency-free equivalent implementing the same
recursive separator hierarchy (paragraphs -> sentences -> words), plus a
markdown heuristic that avoids splitting inside fenced code blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk produced by the splitter.

    Attributes:
        text: Chunk body.
        index: Zero-based position within its source document.
        metadata: Extra fields merged into chunk metadata at ingest time.
    """

    text: str
    index: int
    metadata: dict = field(default_factory=dict)


class RecursiveCharacterSplitter:
    """Split text into overlapping chunks using a separator hierarchy.

    Mirrors the classic recursive-character strategy: try to break on
    paragraph boundaries first, then sentences, then words, and only then
    fall back to hard character cuts. Fenced code blocks (``` ... ```)
    are kept intact when they fit within a single chunk.
    """

    _SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into overlapping chunks."""
        chunks: list[str] = []
        for section in self._split_code_blocks(text):
            chunks.extend(self._recursive_split(section))
        return [c.strip() for c in chunks if c.strip()]

    def split_with_metadata(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split ``text`` and attach ``metadata`` plus chunk indices."""
        metadata = dict(metadata or {})
        return [
            Chunk(text=chunk, index=i, metadata=metadata)
            for i, chunk in enumerate(self.split_text(text))
        ]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _split_code_blocks(self, text: str) -> list[str]:
        """Isolate fenced code blocks so they are not cut mid-block."""
        if "```" not in text:
            return [text]
        sections: list[str] = []
        current: list[str] = []
        in_code = False
        for line in text.splitlines(keepends=True):
            if line.strip().startswith("```"):
                if in_code:
                    current.append(line)
                    sections.append("".join(current))
                    current = []
                    in_code = False
                else:
                    if current:
                        sections.append("".join(current))
                        current = []
                    current.append(line)
                    in_code = True
                continue
            current.append(line)
        if current:
            sections.append("".join(current))
        return sections

    def _recursive_split(self, text: str) -> list[str]:
        """Split ``text`` by the first separator that makes progress."""
        if len(text) <= self.chunk_size:
            return [text]

        for sep in self._SEPARATORS:
            if sep == " " or sep in text:
                pieces = text.split(sep)
                # Only useful when the split actually reduces chunk sizes.
                if len(pieces) > 1 and max(len(p) for p in pieces) <= self.chunk_size:
                    return self._merge(pieces, sep)

        # Hard cut fallback (no usable separator).
        step = self.chunk_size - self.chunk_overlap
        return [
            text[i : i + self.chunk_size] for i in range(0, len(text), step)
        ]

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        """Greedy-merge pieces into chunks of at most ``chunk_size`` with overlap."""
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else current + sep + piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
                # Carry the tail of the previous chunk as overlap.
                tail = current[-self.chunk_overlap :]
                current = tail + sep + piece
            else:
                current = piece  # Oversized piece; recurse on it below.
        if current:
            chunks.append(current)
        return chunks


def default_splitter(chunk_size: int = 1000, chunk_overlap: int = 150) -> RecursiveCharacterSplitter:
    """Splitter configured with the conventional defaults."""
    return RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)