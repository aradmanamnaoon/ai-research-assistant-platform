"""Tests for the research graph orchestration (fake model, no network)."""

from __future__ import annotations

from app.agents.planner import parse_plan


class TestPlannerParsing:
    def test_valid_json_plan(self) -> None:
        plan = parse_plan(
            '{"reasoning": "r", "needs_retrieval": true, "needs_web_research": false,'
            ' "needs_tools": true, "tools_to_run": [{"name": "calculator", "args": {"expression": "1+1"}}],'
            ' "steps": ["s1"], "research_query": "q"}',
            question="q",
        )
        assert plan.needs_retrieval is True
        assert plan.needs_tools is True
        assert plan.tools_to_run[0]["name"] == "calculator"

    def test_unknown_tools_filtered(self) -> None:
        plan = parse_plan(
            '{"needs_tools": true, "tools_to_run": [{"name": "rm_rf"}]}',
            question="q",
        )
        assert plan.tools_to_run == []

    def test_garbage_falls_back_to_heuristic(self) -> None:
        plan = parse_plan("not json at all", question="What is quantum computing?")
        assert plan.needs_web_research is True
        assert "fallback" in plan.reasoning.lower()

    def test_heuristic_detects_retrieval_intent(self) -> None:
        plan = parse_plan("", question="What does my uploaded document say about RAG?")
        assert plan.needs_retrieval is True

    def test_heuristic_detects_greeting(self) -> None:
        plan = parse_plan("", question="hello")
        assert plan.needs_web_research is False


class TestResearchGraph:
    def _input(self, question: str) -> dict:
        return {
            "question": question,
            "history": [],
            "iteration": 0,
            "missing_information": "",
            "max_tool_calls": 6,
        }

    def test_full_pipeline_produces_answer(self, graph) -> None:
        result = graph.invoke(self._input("What are the latest agent advances?"),
                              config={"configurable": {"thread_id": "g1"}})
        assert result["final_answer"]
        assert result["plan"]
        assert result["iteration"] >= 1

    def test_retrieval_branch_runs(self, graph, seeded_store) -> None:
        result = graph.invoke(
            self._input("What does my uploaded document say about RAG?"),
            config={"configurable": {"thread_id": "g2"}},
        )
        # Fake plan sets needs_retrieval=True -> retriever branch runs.
        assert result["retrieved_docs"]

    def test_researcher_branch_runs_and_uses_tools(self, graph) -> None:
        result = graph.invoke(
            self._input("Explain ReAct"),
            config={"configurable": {"thread_id": "g3"}},
        )
        notes = result["research_notes"]
        assert notes
        assert any("findings" in n for n in notes)

    def test_sources_and_citations_shape(self, graph) -> None:
        result = graph.invoke(
            self._input("How does LangGraph persist state?"),
            config={"configurable": {"thread_id": "g4"}},
        )
        assert isinstance(result["citations"], list)
        assert isinstance(result["sources"], list)

    def test_error_resilience(self, graph, monkeypatch) -> None:
        """A failing planner must not crash the pipeline (heuristic path)."""
        from app.agents import planner as planner_module

        def boom(*args, **kwargs):
            raise RuntimeError("LLM down")

        monkeypatch.setattr(planner_module, "invoke_llm", boom)
        result = graph.invoke(
            self._input("Tell me about vector databases"),
            config={"configurable": {"thread_id": "g5"}},
        )
        assert result["final_answer"]

    def test_streaming_yields_node_updates(self, graph) -> None:
        events = list(
            graph.stream(
                self._input("What is RAG?"),
                config={"configurable": {"thread_id": "g6"}},
                stream_mode="updates",
            )
        )
        node_names = set()
        for update in events:
            node_names.update(update.keys())
        assert "planner" in node_names
        assert "synthesizer" in node_names