"""Ingest example documents into the vector store.

Usage::

    python scripts/ingest.py                      # ingest data/documents
    python scripts/ingest.py path/to/folder       # ingest a custom folder
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repository root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.retrieval.ingest import ingest_directory  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402


def main() -> None:
    """Ingest a directory of documents and print a summary."""
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "documents"
    if not directory.is_dir():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    store = VectorStore(settings=settings)
    results = ingest_directory(directory, settings=settings, store=store)

    print("\nIngestion complete")
    print("-" * 50)
    for name, ids in results.items():
        print(f"  {name}: {len(ids)} chunks")
    print("-" * 50)
    print(f"Total chunks in collection: {store.count()}")


if __name__ == "__main__":
    main()