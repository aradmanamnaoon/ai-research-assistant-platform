"""LLM integration layer (custom chat model + factory)."""

from app.llm.base import ChatOpenAICompatible
from app.llm.factory import build_chat_model

__all__ = ["ChatOpenAICompatible", "build_chat_model"]