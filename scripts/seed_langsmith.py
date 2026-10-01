"""Create/sync the LangSmith evaluation dataset.

Requires LANGSMITH_API_KEY (and LANGSMITH_TRACING=true for traces).

Usage::

    python scripts/seed_langsmith.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.evaluation.datasets import ensure_langsmith_dataset  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402


def main() -> None:
    """Upload the local examples to LangSmith."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if not settings.langsmith_api_key:
        print("LANGSMITH_API_KEY is not configured. Set it in .env first.")
        sys.exit(1)

    dataset = ensure_langsmith_dataset(settings.langsmith_project + "-eval")
    print(f"Dataset ready: {dataset.name} (id={dataset.id})")


if __name__ == "__main__":
    main()