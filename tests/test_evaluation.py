"""Tests for evaluation components (offline, deterministic)."""

from __future__ import annotations

from app.evaluation.datasets import load_examples
from app.evaluation.evaluators import (
    citation_validity,
    groundedness_heuristic,
    response_relevance,
)


def test_load_examples_shape() -> None:
    examples = load_examples()
    assert len(examples) >= 5
    assert all("question" in e and "reference" in e for e in examples)


def test_response_relevance_high_on_matching_answer() -> None:
    result = response_relevance(
        {"question": "What is retrieval augmented generation?"},
        {"output": "Retrieval augmented generation grounds answers in documents."},
    )
    assert result["score"] > 0.5


def test_response_relevance_low_on_irrelevant_answer() -> None:
    result = response_relevance(
        {"question": "Explain quantum entanglement and superposition"},
        {"output": "The weather is nice today."},
    )
    assert result["score"] < 0.5


def test_citation_validity_passes() -> None:
    result = citation_validity(
        {"question": "q"},
        {"output": "RAG helps [1].\n\nSources:\n[1] doc.md"},
    )
    assert result["score"] == 1.0


def test_citation_validity_dangling() -> None:
    result = citation_validity(
        {"question": "q"},
        {"output": "Claim [2].\n\nSources:\n[1] doc.md"},
    )
    assert result["score"] < 1.0


def test_citation_validity_no_citations() -> None:
    result = citation_validity({"question": "q"}, {"output": "No citations here."})
    assert result["score"] == 0.0


def test_groundedness_rewards_citations() -> None:
    cited = groundedness_heuristic({}, {"output": "Fact [1]."})
    uncited = groundedness_heuristic({}, {"output": "Fact without citation."})
    assert cited["score"] > uncited["score"]


def test_groundedness_penalizes_superlatives() -> None:
    result = groundedness_heuristic({}, {"output": "This is obviously always correct."})
    assert result["score"] < 1.0