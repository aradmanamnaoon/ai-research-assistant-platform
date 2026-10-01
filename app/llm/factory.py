"""Chat-model factory.

Chooses the concrete model implementation from settings, and validates
that required credentials are present when a real provider is selected.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings
from app.llm.base import ChatOpenAICompatible
from app.llm.fake import SmartFakeChatModel
from app.logging_config import get_logger

logger = get_logger("llm.factory")


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model configured by ``settings``.

    Args:
        settings: Optional settings override (defaults to the singleton).

    Returns:
        A ready-to-use LangChain chat model.

    Raises:
        ValueError: If the OpenAI provider is selected without an API key.
    """
    settings = settings or get_settings()

    if settings.llm_provider == "fake":
        logger.info("Using deterministic fake chat model (LLM_PROVIDER=fake)")
        return SmartFakeChatModel()

    if not settings.openai_api_key:
        raise ValueError(
            "LLM_PROVIDER=openai requires OPENAI_API_KEY. "
            "Set it in .env or export it, or switch LLM_PROVIDER=fake."
        )

    logger.info("Using OpenAI-compatible chat model: %s", settings.model_name)
    return ChatOpenAICompatible(
        model_name=settings.model_name,
        temperature=settings.model_temperature,
        max_tokens=settings.model_max_tokens,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        request_timeout=settings.request_timeout_seconds,
    )