"""Tests for local tools (calculator, notes) and the tool registry."""

from __future__ import annotations

import pytest

from app.tools.local import calculator, list_notes, save_note
from app.tools.registry import EXECUTION_TOOLS, RESEARCH_TOOLS, get_all_tools, get_tool


class TestCalculator:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2 + 2", 4),
            ("(2.5 * 3) / 0.5", 15.0),
            ("2 ** 10", 1024),
            ("-5 + 3", -2),
            ("10 % 3", 1),
            ("7 // 2", 3),
        ],
    )
    def test_valid_expressions(self, expression: str, expected: float) -> None:
        result = calculator.invoke({"expression": expression})
        assert result.endswith(str(expected))

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os')",
            "1/0",
            "os.getcwd",
            "'a' + 'b'",
            "2 +@ 3",
        ],
    )
    def test_invalid_expressions(self, expression: str) -> None:
        result = calculator.invoke({"expression": expression})
        assert result.startswith("Calculator error")

    def test_no_code_execution(self) -> None:
        """Attribute access / names must never evaluate."""
        result = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
        assert "error" in result.lower()


class TestNotes:
    def test_save_and_list_roundtrip(self, settings) -> None:
        import app.tools.local as local

        local.get_settings = lambda: settings
        save_note.invoke({"content": "RAG reduces hallucination", "topic": "rag"})
        save_note.invoke({"content": "Checkpointer needs sqlite", "topic": "langgraph"})
        listed = list_notes.invoke({"topic": "rag"})
        assert "RAG reduces hallucination" in listed
        assert "sqlite" not in listed

    def test_list_empty(self, settings) -> None:
        import app.tools.local as local

        local.get_settings = lambda: settings
        assert list_notes.invoke({"topic": ""}) == "No notes found."


class TestRegistry:
    def test_all_tools_unique_names(self) -> None:
        names = [t.name for t in get_all_tools()]
        assert len(names) == len(set(names))

    def test_research_tools_are_tools(self) -> None:
        assert any(t.name == "wikipedia_search" for t in RESEARCH_TOOLS)
        assert any(t.name == "arxiv_search" for t in RESEARCH_TOOLS)

    def test_get_tool_lookup(self) -> None:
        assert get_tool("calculator") is not None
        assert get_tool("does_not_exist") is None

    def test_execution_tools_cover_planner_surface(self) -> None:
        for expected in ("calculator", "save_note", "list_notes", "tavily_search"):
            assert expected in EXECUTION_TOOLS

    def test_tavily_without_key_is_graceful(self, monkeypatch) -> None:
        from app.tools import web

        monkeypatch.setattr(web, "get_settings", lambda: type("S", (), {"tavily_api_key": None})())
        result = web.tavily_search.invoke({"query": "anything"})
        assert "TAVILY_API_KEY" in result