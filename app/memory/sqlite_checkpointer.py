"""Persistent SQLite checkpointer for LangGraph.

Implements ``BaseCheckpointSaver`` on top of stdlib ``sqlite3`` so agent
state survives process restarts without requiring the optional
``langgraph-checkpoint-sqlite`` package. Serialization uses LangGraph's
own ``JsonPlusSerializer``, which round-trips arbitrary channel values
(including LangChain message objects) safely.

The schema follows the official checkpoint contract:

* ``checkpoints`` — one row per super-step, keyed by
  (thread_id, checkpoint_ns, checkpoint_id).
* ``writes`` — pending writes per task within a checkpoint.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

from app.logging_config import get_logger

logger = get_logger("memory.checkpointer")


class SqliteCheckpointSaver(BaseCheckpointSaver):
    """SQLite-backed checkpoint saver (stdlib sqlite3, thread-safe).

    Usage::

        saver = SqliteCheckpointSaver("data/checkpoints.sqlite")
        graph = builder.compile(checkpointer=saver)
        graph.invoke(
            {"question": "..."},
            {"configurable": {"thread_id": "user-1"}},
        )
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open (and create if needed) the checkpoint database.

        Args:
            db_path: Path to the SQLite file, or ``":memory:"`` for tests.
        """
        super().__init__()
        self.db_path = str(db_path)
        self.is_memory = self.db_path == ":memory:"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; transactions via `with conn`
        )
        self._setup()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        """Create tables and indices if they do not exist."""
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint BLOB NOT NULL,
                    metadata BLOB NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                CREATE INDEX IF NOT EXISTS checkpoints_by_thread
                    ON checkpoints (thread_id, checkpoint_ns);
                CREATE INDEX IF NOT EXISTS writes_by_checkpoint
                    ON writes (thread_id, checkpoint_ns, checkpoint_id);
                """
            )

    # ------------------------------------------------------------------ #
    # BaseCheckpointSaver contract
    # ------------------------------------------------------------------ #
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Load the latest checkpoint for a thread, with pending writes."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self._lock:
            if checkpoint_id:
                row = self._conn.execute(
                    "SELECT checkpoint, metadata, parent_checkpoint_id "
                    "FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT checkpoint, metadata, parent_checkpoint_id "
                    "FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_ns = ? "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                ).fetchone()

        if row is None:
            return None

        checkpoint = self._load_blob(row[0])
        metadata = self._load_blob(row[1])
        parent_checkpoint_id = row[2]

        checkpoint_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        parent_config: RunnableConfig | None = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint["id"]
        )
        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(  # noqa: A003 - matches the base-class name
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 - base-class name
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Iterate checkpoints for a thread, newest first.

        Args:
            config: Config scoped to a thread (or ``None`` for all threads).
            filter: Optional metadata filter (subset match).
            before: Optional config to page before a given checkpoint.
            limit: Optional maximum number of tuples to yield.
        """
        if config is None:
            thread_id, checkpoint_ns = None, None
        else:
            thread_id = config["configurable"].get("thread_id")
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata, "
            "parent_checkpoint_id FROM checkpoints WHERE 1=1"
        )
        params: list[Any] = []
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        if checkpoint_ns:
            query += " AND checkpoint_ns = ?"
            params.append(checkpoint_ns)
        if before and get_checkpoint_id(before):
            query += " AND checkpoint_id < ?"
            params.append(get_checkpoint_id(before))
        query += " ORDER BY checkpoint_id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        for (
            row_thread_id,
            row_ns,
            row_checkpoint_id,
            checkpoint_blob,
            metadata_blob,
            parent_id,
        ) in rows:
            metadata = self._load_blob(metadata_blob)
            if filter and not self._metadata_matches(metadata, filter):
                continue
            checkpoint = self._load_blob(checkpoint_blob)
            checkpoint_config: RunnableConfig = {
                "configurable": {
                    "thread_id": row_thread_id,
                    "checkpoint_ns": row_ns,
                    "checkpoint_id": row_checkpoint_id,
                }
            }
            parent_config: RunnableConfig | None = None
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": row_thread_id,
                        "checkpoint_ns": row_ns,
                        "checkpoint_id": parent_id,
                    }
                }
            yield CheckpointTuple(
                config=checkpoint_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=self._load_pending_writes(
                    row_thread_id, row_ns, row_checkpoint_id
                ),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        """Persist one checkpoint (called by the graph after each step)."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        checkpoint_blob = self._dump_blob(checkpoint)
        metadata_blob = self._dump_blob(metadata)

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                 checkpoint, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_id,
                    checkpoint_blob,
                    metadata_blob,
                ),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Persist intermediate writes for a checkpoint (e.g. interrupts)."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config) or ""

        with self._lock, self._conn:
            for idx, (channel, value) in enumerate(writes):
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                     channel, value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        self._dump_blob(value),
                    ),
                )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread."""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
            )
            self._conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def _dump_blob(self, value: Any) -> bytes:
        """Serialize an arbitrary value to a storable blob."""
        type_name, blob = self.serde.dumps_typed(value)
        # Prefix the type tag so loads_typed can be reconstructed exactly.
        return type_name.encode("utf-8") + b":" + blob

    def _load_blob(self, blob: bytes | None) -> Any:
        """Deserialize a blob produced by :meth:`_dump_blob`."""
        if blob is None:
            return None
        type_name, _, body = blob.partition(b":")
        return self.serde.loads_typed((type_name.decode("utf-8"), body))

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes for one checkpoint, LangGraph-format."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_id, channel, value FROM writes "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
                "ORDER BY idx",
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()
        return [
            (task_id, channel, self._load_blob(value))
            for task_id, channel, value in rows
        ]

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        """Subset match used by :meth:`list` filtering."""
        return all(
            key in metadata and metadata[key] == value
            for key, value in filter.items()
        )

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()