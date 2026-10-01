"""Run the evaluation suite.

LangSmith mode when LANGSMITH_API_KEY is set (full experiments with
traces and persistent results); local offline mode otherwise.

Usage::

    python scripts/run_evaluation.py            # full suite
    python scripts/run_evaluation.py --limit 3  # quick subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.evaluation.runner import run_evaluation  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402


def main() -> None:
    """Run evaluation and print the aggregate report."""
    parser = argparse.ArgumentParser(description="Run the research assistant evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples (local mode)")
    parser.add_argument(
        "--no-upload", action="store_true", help="Force local mode even when LangSmith is configured"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    report = run_evaluation(limit=args.limit, upload_to_langsmith=not args.no_upload)
    print("\nEvaluation report")
    print("-" * 50)
    print(f"Mode: {report['mode']}")
    print(f"Examples: {report['num_examples']}")
    for key, score in report["scores"].items():
        print(f"  {key:28s} {score:.3f}")
    if report.get("url"):
        print(f"Experiment: {report['url']}")
    print("-" * 50)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()