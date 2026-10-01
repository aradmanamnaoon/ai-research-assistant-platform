"""Tests for the custom chat model, message conversion and JSON extraction."""

from __future__ import annotations

import json

from app.agents.llm_utils import extract_json
from app.llm.base import ChatOpenAICompatible, _parse_tool_calls, _to_openai_message
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class TestMessageConversion:
    def test_system_message(self) -> None:
        assert _to_openai_message(SystemMessage(content="sys")) == {"role": "system", "content": "sys"}

    def test_human_message(self) -> None:
        assert _to_openai_message(HumanMessage(content="hi")) == {"role": "user", "content": "hi"}

    def test_tool_message_stringifies_structured_content(self) -> None:
        message = ToolMessage.model_construct(content={"a": 1}, tool_call_id="t1")
        converted = _to_openai_message(message)
        assert converted["role"] == "tool"
        assert converted["tool_call_id"] == "t1"
        assert json.loads(converted["content"]) == {"a": 1}

    def test_ai_message_with_tool_calls(self) -> None:
        converted = _to_openai_message(
            AIMessage(content="", tool_calls=[{"name": "f", "args": {"x": 1}, "id": "c1", "type": "tool_call"}])
        )
        assert converted["role"] == "assistant"
        assert converted["tool_calls"][0]["function"]["name"] == "f"
        assert json.loads(converted["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


class TestToolCallParsing:
    def test_parse_tool_calls(self) -> None:
        parsed = _parse_tool_calls(
            {"tool_calls": [{"id": "x", "function": {"name": "calc", "arguments": '{"expr": "1+1"}'}}]}
        )
        assert parsed[0]["name"] == "calc"
        assert parsed[0]["args"] == {"expr": "1+1"}
        assert parsed[0]["type"] == "tool_call"

    def test_malformed_arguments_become_raw(self) -> None:
        parsed = _parse_tool_calls(
            {"tool_calls": [{"id": "x", "function": {"name": "calc", "arguments": "{bad json"}}]}
        )
        assert parsed[0]["args"] == {"raw": "{bad json"}


class TestExtractJson:
    def test_plain_json(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        text = 'Sure!\n```json\n{"needs_retrieval": true}\n```'
        assert extract_json(text) == {"needs_retrieval": True}

    def test_json_with_surrounding_prose(self) -> None:
        text = 'Here is the plan: {"a": 1, "b": {"c": 2}} hope that helps'
        assert extract_json(text) == {"a": 1, "b": {"c": 2}}

    def test_braces_inside_strings(self) -> None:
        text = '{"template": "hello {name}", "n": 3}'
        assert extract_json(text) == {"template": "hello {name}", "n": 3}

    def test_no_json(self) -> None:
        assert extract_json("no object here") is None
        assert extract_json("") is None


class TestChatOpenAICompatible:
    def _model_with_stub_client(self, completion_payload: dict) -> ChatOpenAICompatible:
        """Build a model whose HTTP client is replaced by an in-memory stub."""
        model = ChatOpenAICompatible(
            model_name="stub", api_key="test-key", temperature=0
        )

        class StubResponse:
            def __init__(self, data: dict) -> None:
                self._data = data

            def model_dump(self) -> dict:
                return self._data

        class StubCompletions:
            def create(self, **kwargs):
                object.__setattr__(model, "last_payload", kwargs)
                return StubResponse(completion_payload)

        class StubChat:
            completions = StubCompletions()

        object.__setattr__(model, "_client", type("StubClient", (), {"chat": StubChat()})())
        return model

    def test_generate_returns_aimessage(self) -> None:
        payload = {
            "choices": [
                {"message": {"content": "hello", "refusal": None}, "finish_reason": "stop"}
            ],
            "model": "stub",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        model = self._model_with_stub_client(payload)
        result = model.invoke([HumanMessage(content="hi")])
        assert result.content == "hello"
        assert result.usage_metadata["total_tokens"] == 7
        assert model.last_payload["model"] == "stub"  # type: ignore[attr-defined]

    def test_generate_parses_tool_calls(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "1", "function": {"name": "f", "arguments": '{"x": 2}'}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        model = self._model_with_stub_client(payload)
        result = model.invoke([HumanMessage(content="call it")])
        assert result.tool_calls[0]["name"] == "f"
        assert result.tool_calls[0]["args"] == {"x": 2}

    def test_bind_tools_sends_openai_schema(self) -> None:
        payload = {"choices": [{"message": {"content": "ok", "refusal": None}, "finish_reason": "stop"}]}
        model = self._model_with_stub_client(payload)

        from langchain_core.tools import tool

        @tool
        def my_tool(query: str) -> str:
            """Search something."""

        model.bind_tools([my_tool]).invoke([HumanMessage(content="go")])
        sent = model.last_payload["tools"][0]  # type: ignore[attr-defined]
        assert sent["type"] == "function"
        assert sent["function"]["name"] == "my_tool"