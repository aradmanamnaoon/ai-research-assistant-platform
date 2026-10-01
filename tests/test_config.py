"""Tests for configuration and the model factory."""

from __future__ import annotations

import pytest

from app.config import Settings


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings(llm_provider="fake")
        assert settings.model_name == "gpt-4o-mini"
        assert settings.retrieval_top_k == 5
        assert settings.api_prefix == "/api/v1"

    def test_ensure_dirs_creates_paths(self, tmp_path) -> None:
        settings = Settings(
            llm_provider="fake",
            data_dir=tmp_path / "d",
            chroma_dir=tmp_path / "d" / "chroma",
            sqlite_path=tmp_path / "d" / "cp.sqlite",
            notes_path=tmp_path / "d" / "notes" / "notes.jsonl",
        )
        settings.ensure_dirs()
        assert (tmp_path / "d" / "chroma").is_dir()
        assert (tmp_path / "d" / "notes").is_dir()

    def test_langsmith_enabled_requires_key_and_flag(self, tmp_path) -> None:
        assert not Settings(llm_provider="fake", langsmith_tracing=True).langsmith_enabled
        assert not Settings(llm_provider="fake", langsmith_api_key="k").langsmith_enabled
        assert Settings(
            llm_provider="fake", langsmith_tracing=True, langsmith_api_key="k"
        ).langsmith_enabled


class TestModelFactory:
    def test_fake_provider(self, settings) -> None:
        from app.llm.factory import build_chat_model

        model = build_chat_model(settings)
        assert type(model).__name__ == "SmartFakeChatModel"

    def test_openai_without_key_raises(self, settings) -> None:
        from app.llm.factory import build_chat_model

        settings = settings.model_copy(update={"llm_provider": "openai", "openai_api_key": None})
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_chat_model(settings)

    def test_openai_with_key_builds_compatible_model(self, settings) -> None:
        from app.llm.base import ChatOpenAICompatible
        from app.llm.factory import build_chat_model

        settings = settings.model_copy(
            update={"llm_provider": "openai", "openai_api_key": "test", "model_name": "test-model"}
        )
        model = build_chat_model(settings)
        assert isinstance(model, ChatOpenAICompatible)
        assert model.model_name == "test-model"