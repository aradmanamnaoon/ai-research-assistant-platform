"""Custom LangChain chat model backed by any OpenAI-compatible endpoint.

Rather than depending on ``langchain-openai`` (not installed in this
environment), this module implements ``BaseChatModel`` directly on top of
the ``openai`` SDK. It demonstrates the full LangChain model contract:

* message translation (LangChain <-> OpenAI Chat Completions)
* tool binding via ``bind_tools``
* tool-call parsing back into ``AIMessage.tool_calls``
* streaming (single-chunk implementation on top of ``_generate``)
* transient-error retries with exponential backoff
* LangSmith tracing for free (inherited from the LangChain callback system)

Works with OpenAI, OpenRouter, Azure gateways, vLLM, Ollama, or any
service exposing the ``/chat/completions`` interface.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.logging_config import get_logger

logger = get_logger("llm")

# Errors considered transient and worth retrying.
_RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ChatOpenAICompatible(BaseChatModel):
    """LangChain chat model adapter for OpenAI-compatible Chat APIs.

    Attributes:
        model_name: Model identifier understood by the endpoint.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens the model may generate.
        base_url: Optional endpoint override (OpenRouter, vLLM, Ollama...).
        api_key: Optional API key; ``None`` uses the ``OPENAI_API_KEY``
            environment variable.
        request_timeout: Per-request timeout in seconds.
        max_retries: Number of attempts for transient failures.
    """

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024
    base_url: str | None = None
    api_key: str | None = None
    request_timeout: float = 90.0
    max_retries: int = 3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The OpenAI client is constructed lazily and kept off the pydantic
        # field set so it never leaks into serialization / checkpointing.
        object.__setattr__(
            self,
            "_client",
            _build_client(self.base_url, self.api_key, self.request_timeout),
        )

    # ------------------------------------------------------------------ #
    # LangChain model identity
    # ------------------------------------------------------------------ #
    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain for run metadata."""
        return "openai-compatible"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Parameters surfaced in LangSmith traces."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    # ------------------------------------------------------------------ #
    # Tool binding
    # ------------------------------------------------------------------ #
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        **kwargs: Any,
    ) -> Any:
        """Bind LangChain tools as OpenAI function-calling tools.

        Args:
            tools: LangChain tools, Pydantic classes or JSON schemas.
            **kwargs: Extra kwargs forwarded to every completion request.

        Returns:
            A ``Runnable`` that merges the tool schemas into each call.
        """
        oai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=oai_tools, **kwargs)

    # ------------------------------------------------------------------ #
    # Core generation
    # ------------------------------------------------------------------ #
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Translate messages, call the endpoint, and parse the reply."""
        payload_messages = [_to_openai_message(m) for m in messages]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        data = self._request_with_retry(payload)
        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = _parse_tool_calls(message)
        generation = ChatGeneration(
            message=AIMessage(
                content=message.get("content") or "",
                additional_kwargs={
                    "refusal": message.get("refusal"),
                    "finish_reason": choice.get("finish_reason"),
                },
                tool_calls=tool_calls,
                response_metadata={
                    "model": data.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                },
                usage_metadata=_usage_metadata(data.get("usage")),
            ),
        )
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Minimal streaming: yield the full response as one chunk.

        The graph-level SSE endpoint streams *events* (``stream_mode=
        "updates"``), so single-chunk streaming is sufficient here while
        keeping the model compatible with streaming callers.
        """
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        generation = result.generations[0]
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=generation.message.content,
                additional_kwargs=generation.message.additional_kwargs,
                tool_calls=generation.message.tool_calls,
            )
        )

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #
    def _request_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to the chat endpoint, retrying transient failures.

        Args:
            payload: OpenAI Chat Completions request body.

        Returns:
            The raw completion payload.

        Raises:
            RuntimeError: After exhausting retries on a retryable error, or
                immediately on a non-retryable error.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**payload)
                return response.model_dump()
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                status = getattr(exc, "status_code", None)
                retryable = status in _RETRYABLE_HTTP_STATUS or not hasattr(
                    exc, "status_code"
                )
                if not retryable or attempt == self.max_retries:
                    logger.error("LLM request failed: %s", exc)
                    raise RuntimeError(f"LLM request failed: {exc}") from exc
                delay = min(2.0 ** (attempt - 1), 8.0)
                logger.warning(
                    "LLM transient error (%s), retrying in %.1fs "
                    "(attempt %d/%d)",
                    exc,
                    delay,
                    attempt,
                    self.max_retries,
                )
                last_error = exc
                time.sleep(delay)
        raise RuntimeError(f"LLM request failed after retries: {last_error}")


def _build_client(
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> Any:
    """Construct an OpenAI SDK client (lazy import keeps tests cheap)."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _to_openai_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message into an OpenAI Chat Completions dict."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolMessage):
        content = message.content
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": content,
        }
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["args"]),
                    },
                }
                for call in message.tool_calls
            ]
            # Some strict providers reject non-empty content on tool-call
            # assistant turns.
            if not payload["content"]:
                payload["content"] = ""
        return payload
    return {"role": "user", "content": str(message.content)}


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract OpenAI tool calls into LangChain ``AIMessage.tool_calls``."""
    parsed: list[dict[str, Any]] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
        except json.JSONDecodeError:
            logger.warning("Malformed tool arguments: %r", raw_arguments)
            args = {"raw": raw_arguments}
        parsed.append(
            {
                "name": function.get("name", "unknown"),
                "args": args,
                "id": call.get("id") or f"call_{index}",
                "type": "tool_call",
            }
        )
    return parsed


def _usage_metadata(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Normalize OpenAI usage counters into LangChain usage metadata."""
    if not usage:
        return None
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }