"""Evaluators for the research assistant (LangSmith-compatible).

Evaluators follow the LangSmith evaluator contract: a callable receiving
``(inputs, outputs, reference_outputs)`` and returning a dict with
``key``, ``score`` (numeric or bool) and ``comment``. Two kinds are
provided:

* Deterministic code evaluators — citations, grounding, relevance.
* LLM-as-judge evaluators — helpfulness, correctness (requires a model;
  degrades gracefully when no API key is configured).
"""

from __future__ import annotations

import re
from typing import Any

from app.logging_config import get_logger

logger = get_logger("evaluation.evaluators")


# --------------------------------------------------------------------- #
# Deterministic evaluators
# --------------------------------------------------------------------- #
def response_relevance(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score whether the answer addresses the question's keywords.

    A cheap lexical proxy: fraction of significant question words that
    also appear in the answer.
    """
    question = str(inputs.get("question", "")).lower()
    answer = str(outputs.get("output", outputs.get("answer", ""))).lower()

    stopwords = {
        "what", "which", "who", "when", "where", "why", "how", "is", "are",
        "the", "a", "an", "of", "in", "to", "and", "for", "with", "does",
        "do", "can", "should", "tell", "me", "about", "explain", "give",
    }
    keywords = {
        word.strip(".,?!'\"")
        for word in question.split()
        if len(word) > 2 and word.strip(".,?!'\"") not in stopwords
    }
    if not keywords:
        return {"key": "response_relevance", "score": 1.0, "comment": "No significant keywords"}

    covered = sum(1 for word in keywords if word in answer)
    score = covered / len(keywords)
    return {
        "key": "response_relevance",
        "score": score,
        "comment": f"{covered}/{len(keywords)} question keywords appear in the answer",
    }


def citation_validity(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check that bracketed citations resolve to real listed sources.

    Extracts ``[n]`` references from the answer and verifies each number
    is backed by a matching ``[n] source`` line in the Sources section.
    """
    answer = str(outputs.get("output", outputs.get("answer", "")))
    used = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    if not used:
        return {
            "key": "citation_validity",
            "score": 0.0,
            "comment": "No inline citations found in the answer",
        }

    listed = {int(m) for m in re.findall(r"^\[(\d+)\]", answer, flags=re.MULTILINE)}
    if not listed:
        return {
            "key": "citation_validity",
            "score": 0.5,
            "comment": "Citations used but no Sources section found",
        }

    dangling = used - listed
    score = 1.0 if not dangling else max(0.0, 1.0 - len(dangling) / max(len(used), 1))
    return {
        "key": "citation_validity",
        "score": score,
        "comment": "All citations resolve" if not dangling else f"Dangling citations: {sorted(dangling)}",
    }


def groundedness_heuristic(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Heuristic grounding check: penalize answers that cite nothing and
    use strong unsupported superlatives."""
    answer = str(outputs.get("output", outputs.get("answer", "")))
    has_citations = bool(re.search(r"\[\d+\]", answer))
    unsupported_claims = len(
        re.findall(r"\b(always|never|everyone knows|obviously|guaranteed)\b", answer, flags=re.I)
    )
    score = 1.0
    if not has_citations:
        score -= 0.5
    score -= 0.1 * unsupported_claims
    score = max(0.0, score)
    return {
        "key": "groundedness_heuristic",
        "score": score,
        "comment": (
            f"citations={'yes' if has_citations else 'no'}, "
            f"unsupported_hedges={unsupported_claims}"
        ),
    }


# --------------------------------------------------------------------- #
# LLM-as-judge evaluators
# --------------------------------------------------------------------- #
def helpfulness_judge(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LLM-as-judge: rate helpfulness 0-1 against the question.

    Uses the platform's own chat model. When no LLM is available the
    evaluator reports a neutral score with an explanatory comment rather
    than raising, so evaluation runs never hard-fail.
    """
    question = inputs.get("question", "")
    answer = str(outputs.get("output", outputs.get("answer", "")))
    try:
        model = _judge_model()
        prompt = (
            "Rate the helpfulness of this answer from 0.0 to 1.0.\n\n"
            f"Question: {question}\n\nAnswer:\n{answer[:4000]}\n\n"
            'Respond with ONLY a JSON object: {"score": <0.0-1.0>, '
            '"comment": "<one sentence>"}'
        )
        response = model.invoke(prompt)
        data = _parse_judge_json(str(response.content))
        score = float(data.get("score", 0.0))
        return {
            "key": "helpfulness_judge",
            "score": max(0.0, min(1.0, score)),
            "comment": str(data.get("comment", "")),
        }
    except Exception as exc:  # noqa: BLE001 - judge is optional
        logger.warning("helpfulness_judge unavailable (%s)", exc)
        return {
            "key": "helpfulness_judge",
            "score": 0.5,
            "comment": f"Judge unavailable: {exc}",
        }


def correctness_judge(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LLM-as-judge: compare the answer against a reference answer."""
    question = inputs.get("question", "")
    answer = str(outputs.get("output", outputs.get("answer", "")))
    reference = str((reference_outputs or {}).get("output", ""))
    if not reference:
        return {
            "key": "correctness_judge",
            "score": 0.5,
            "comment": "No reference answer provided",
        }
    try:
        model = _judge_model()
        prompt = (
            "Compare this answer to the reference answer for the same "
            "question. Score factual agreement from 0.0 to 1.0.\n\n"
            f"Question: {question}\n\nAnswer:\n{answer[:4000]}\n\n"
            f"Reference:\n{reference[:2000]}\n\n"
            'Respond with ONLY: {"score": <0.0-1.0>, "comment": "<one sentence>"}'
        )
        response = model.invoke(prompt)
        data = _parse_judge_json(str(response.content))
        return {
            "key": "correctness_judge",
            "score": max(0.0, min(1.0, float(data.get("score", 0.0)))),
            "comment": str(data.get("comment", "")),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("correctness_judge unavailable (%s)", exc)
        return {
            "key": "correctness_judge",
            "score": 0.5,
            "comment": f"Judge unavailable: {exc}",
        }


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _judge_model() -> Any:
    """Build the judge model (respects LLM_PROVIDER, incl. fake)."""
    from app.llm.factory import build_chat_model

    return build_chat_model()


def _parse_judge_json(text: str) -> dict[str, Any]:
    """Parse the judge's JSON response tolerantly."""
    from app.agents.llm_utils import extract_json

    data = extract_json(text)
    if not data:
        raise ValueError(f"Judge returned non-JSON output: {text[:120]}")
    return data