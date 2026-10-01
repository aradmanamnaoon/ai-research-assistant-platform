"""Conversation history store (SQLite via SQLAlchemy).

The LangGraph checkpointer persists graph state; this store persists a
denormalized turn log for fast history rendering and analytics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger("memory.conversations")


class Base(DeclarativeBase):
    """Declarative base for conversation models."""


class ConversationMessage(Base):
    """One turn of a conversation (question or answer)."""

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ConversationStore:
    """CRUD wrapper for conversation turns.

    Attributes:
        engine: SQLAlchemy engine.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Create the engine and ensure the schema exists.

        Args:
            settings: Optional settings override.
        """
        settings = settings or get_settings()
        settings.ensure_dirs()
        self.engine = create_engine(
            f"sqlite:///{settings.sqlite_path}", future=True
        )
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine, future=True)

    def add_message(
        self, thread_id: str, role: str, content: str
    ) -> ConversationMessage:
        """Persist one conversation turn.

        Args:
            thread_id: Conversation identifier.
            role: ``"user"`` or ``"assistant"``.
            content: Message body.

        Returns:
            The persisted row.
        """
        with Session(self.engine) as session:
            message = ConversationMessage(
                thread_id=thread_id, role=role, content=content
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            return message

    def get_history(
        self,
        thread_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return a thread's turns, oldest first.

        Args:
            thread_id: Conversation identifier.
            limit: Maximum number of turns.
            offset: Pagination offset.

        Returns:
            List of message dicts.
        """
        with Session(self.engine) as session:
            rows = session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.thread_id == thread_id)
                .order_by(ConversationMessage.id.asc())
                .offset(offset)
                .limit(limit)
            ).all()
            return [
                {
                    "id": row.id,
                    "thread_id": row.thread_id,
                    "role": row.role,
                    "content": row.content,
                    "created_at": row.created_at.isoformat(timespec="seconds"),
                }
                for row in rows
            ]

    def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return all threads with their message counts and last activity."""
        with Session(self.engine) as session:
            rows = session.execute(
                select(
                    ConversationMessage.thread_id,
                    func.count(ConversationMessage.id),
                    func.max(ConversationMessage.created_at),
                )
                .group_by(ConversationMessage.thread_id)
                .order_by(func.max(ConversationMessage.created_at).desc())
                .limit(limit)
            ).all()
            return [
                {
                    "thread_id": thread_id,
                    "message_count": count,
                    "last_activity": last.isoformat(timespec="seconds") if last else None,
                }
                for thread_id, count, last in rows
            ]

    def delete_thread(self, thread_id: str) -> int:
        """Delete a thread's messages. Returns the number deleted."""
        with Session(self.engine) as session:
            result = session.query(ConversationMessage).filter_by(
                thread_id=thread_id
            )
            count = result.delete()
            session.commit()
            return count