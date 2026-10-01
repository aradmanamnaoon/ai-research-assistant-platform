"""Application configuration.

All settings are environment driven (12-factor style). Secrets are never
hardcoded; ``.env`` files are supported via ``pydantic-settings``.

Example:
    from app.config import get_settings

    settings = get_settings()
    print(settings.model_name)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central application settings loaded from environment variables.

    Every attribute maps to an environment variable (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_name: str = "AI Research Assistant Platform"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    # "openai" uses the OpenAI-compatible Chat Completions API (works with
    # OpenAI, Azure OpenAI gateways, OpenRouter, vLLM, Ollama, etc.).
    # "fake" uses a deterministic scripted model (tests / offline demos).
    llm_provider: Literal["openai", "fake"] = "openai"
    model_name: str = "gpt-4o-mini"
    model_temperature: float = 0.2
    model_max_tokens: int = 1024
    request_timeout_seconds: float = 90.0

    # ------------------------------------------------------------------ #
    # Data / storage
    # ------------------------------------------------------------------ #
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    sqlite_path: Path = PROJECT_ROOT / "data" / "checkpoints.sqlite"
    notes_path: Path = PROJECT_ROOT / "data" / "notes" / "notes.jsonl"

    # ------------------------------------------------------------------ #
    # RAG
    # ------------------------------------------------------------------ #
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.0
    collection_name: str = "research_documents"

    # ------------------------------------------------------------------ #
    # Agent behaviour
    # ------------------------------------------------------------------ #
    max_graph_iterations: int = 3
    max_research_tool_calls: int = 6
    enable_web_research: bool = True
    enable_loop_refinement: bool = True

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    # ------------------------------------------------------------------ #
    # External services
    # ------------------------------------------------------------------ #
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    tavily_api_key: str | None = None

    # ------------------------------------------------------------------ #
    # LangSmith observability
    # ------------------------------------------------------------------ #
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "ai-research-assistant"

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    def ensure_dirs(self) -> None:
        """Create runtime data directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def langsmith_enabled(self) -> bool:
        """True when LangSmith tracing can actually be activated."""
        return bool(self.langsmith_tracing and self.langsmith_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton :class:`Settings` instance."""
    return Settings()