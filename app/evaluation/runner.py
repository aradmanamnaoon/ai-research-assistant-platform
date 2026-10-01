"""Evaluation runner.

Two modes:

* **LangSmith mode** (``LANGSMITH_API_KEY`` set): runs the standard
  ``evaluate()`` flow against a synced LangSmith dataset and reports an
  experiment URL.
* **Local mode** (no key): runs the same target and evaluators against
  the local examples file and prints an aggregate report, so evaluation
  is demonstrable offline and in CI.
"""

from __future__ import annotations

import statistics
from typing import Any

from langsmith import traceable

from app.config import get_settings
from app.evaluation.datasets import EXAMPLES_PATH, load_examples
from app.evaluation.evaluators import (
    citation_validity,
    groundedness_heuristic,
    helpfulness_judge,
    response_relevance,
)
from app.logging_config import get_logger

logger = get_logger("evaluation.runner")

DEFAULT_EVALUATORS = [
    response_relevance,
    citation_validity,
    groundedness_heuristic,
    helpfulness_judge,
]


@traceable(name="research_target")
def research_target(inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluation target: run the full pipeline for one question."""
    from app.services.research_service import ResearchService

    service = ResearchService()
    thread_id = f"eval-{abs(hash(inputs['question'])) % 10_000_000}"
    payload = service.ask(inputs["question"], thread_id=thread_id)
    return {"output": payload["answer"], "citations": payload["citations"]}


def run_evaluation(
    dataset_name: str = "ai-research-assistant-eval",
    evaluators: list | None = None,
    limit: int | None = None,
    upload_to_langsmith: bool = True,
) -> dict[str, Any]:
    """Run the evaluation suite and return an aggregate report.

    Args:
        dataset_name: LangSmith dataset name (LangSmith mode only).
        evaluators: Evaluator list override.
        limit: Optional cap on the number of examples (local mode).
        upload_to_langsmith: Whether to try the LangSmith flow first.

    Returns:
        ``{"mode", "num_examples", "scores": {key: mean}, "url"?}``.
    """
    settings = get_settings()
    evaluators = evaluators or DEFAULT_EVALUATORS

    if upload_to_langsmith and settings.langsmith_api_key:
        return _run_langsmith(dataset_name, evaluators, settings.langsmith_project)
    return _run_local(evaluators, limit)


def _run_langsmith(dataset_name: str, evaluators: list, project: str) -> dict[str, Any]:
    """Standard LangSmith evaluation flow."""
    from langsmith import Client, evaluate

    from app.evaluation.datasets import create_langsmith_dataset

    client = Client()
    dataset = create_langsmith_dataset(dataset_name, client=client)

    logger.info("Running LangSmith evaluation on dataset %r", dataset_name)
    results = evaluate(
        research_target,
        data=dataset,
        evaluators=evaluators,
        experiment_prefix="research-assistant",
        metadata={"project": project},
        max_concurrency=2,
    )

    scores: dict[str, list[float]] = {}
    for row in results:
        for result in row["evaluation_results"]["results"]:
            scores.setdefault(result.key, []).append(float(result.score or 0.0))

    aggregate = {key: statistics.mean(values) for key, values in scores.items()}
    logger.info("Evaluation complete: %s", aggregate)
    return {
        "mode": "langsmith",
        "num_examples": len(list(results)),
        "scores": aggregate,
        "url": results.experiment_url if hasattr(results, "experiment_url") else None,
    }


def _run_local(evaluators: list, limit: int | None) -> dict[str, Any]:
    """Offline evaluation against the local examples file."""
    examples = load_examples(EXAMPLES_PATH)
    if limit:
        examples = examples[:limit]
    if not examples:
        return {"mode": "local", "num_examples": 0, "scores": {}}

    scores: dict[str, list[float]] = {}
    for example in examples:
        inputs = {"question": example["question"]}
        reference = {"output": example.get("reference", "")}
        try:
            outputs = research_target(inputs)
        except Exception as exc:  # noqa: BLE001
            logger.error("Target failed for %r: %s", example["question"][:60], exc)
            outputs = {"output": ""}
        for evaluator in evaluators:
            try:
                result = evaluator(inputs, outputs, reference)
                scores.setdefault(result["key"], []).append(float(result["score"]))
            except Exception as exc:  # noqa: BLE001
                logger.error("Evaluator %s failed: %s", evaluator.__name__, exc)

    aggregate = {key: round(statistics.mean(values), 4) for key, values in scores.items()}
    logger.info("Local evaluation: %s", aggregate)
    return {"mode": "local", "num_examples": len(examples), "scores": aggregate}