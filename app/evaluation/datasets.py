"""Dataset management for evaluation.

Loads example records from ``evaluation_examples/examples.json``, creates
or reuses a LangSmith dataset, and uploads examples with
question/reference pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings
from app.logging_config import get_logger

logger = get_logger("evaluation.datasets")

EXAMPLES_PATH = PROJECT_ROOT / "evaluation_examples" / "examples.json"


def load_examples(path: Path | None = None) -> list[dict[str, Any]]:
    """Load evaluation examples from disk.

    Each example: ``{"question", "reference", "topic", "expected_tools"}``.
    """
    path = path or EXAMPLES_PATH
    if not path.exists():
        logger.error("Examples file not found: %s", path)
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    valid = [r for r in records if r.get("question")]
    logger.info("Loaded %d evaluation examples", len(valid))
    return valid


def create_langsmith_dataset(
    dataset_name: str,
    examples: list[dict[str, Any]] | None = None,
    client: Any | None = None,
) -> Any:
    """Create (or reuse) a LangSmith dataset and seed it with examples.

    Args:
        dataset_name: Dataset name in LangSmith.
        examples: Optional examples override.
        client: Optional LangSmith client (defaults to a new one).

    Returns:
        The LangSmith dataset object.

    Raises:
        RuntimeError: When LangSmith is not configured.
    """
    from langsmith import Client

    settings = get_settings()
    if not settings.langsmith_api_key:
        raise RuntimeError(
            "LangSmith is not configured (LANGSMITH_API_KEY missing). "
            "Set it in .env to upload datasets."
        )
    client = client or Client()
    examples = examples if examples is not None else load_examples()

    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        dataset = datasets[0]
        logger.info("Reusing LangSmith dataset: %s", dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="AI Research Assistant evaluation suite",
        )
        logger.info("Created LangSmith dataset: %s", dataset_name)

    existing_inputs = {e.inputs.get("question") for e in client.list_examples(dataset_id=dataset.id)}
    added = 0
    for example in examples:
        question = example["question"]
        if question in existing_inputs:
            continue
        client.create_example(
            dataset_id=dataset.id,
            inputs={"question": question},
            outputs={"output": example.get("reference", "")},
            metadata={
                "topic": example.get("topic", ""),
                "expected_tools": example.get("expected_tools", []),
            },
        )
        added += 1
    logger.info("Uploaded %d new examples to %s", added, dataset_name)
    return dataset


def ensure_langsmith_dataset(dataset_name: str) -> Any:
    """Convenience wrapper: load examples from the default file and sync."""
    return create_langsmith_dataset(dataset_name, load_examples())