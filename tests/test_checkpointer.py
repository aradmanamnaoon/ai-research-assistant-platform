"""Tests for the custom SQLite checkpointer."""

from __future__ import annotations

import operator
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict


class TinyState(TypedDict):
    messages: Annotated[list, operator.add]
    counter: int


def _build_tiny_graph(saver: Any) -> Any:
    builder = StateGraph(TinyState)
    builder.add_node(
        "add", lambda s: {"messages": [f"step-{s['counter']}"], "counter": s["counter"] + 1}
    )
    builder.add_edge(START, "add")
    builder.add_edge("add", END)
    return builder.compile(checkpointer=saver)


class TestSqliteCheckpointSaver:
    def test_put_and_get_tuple(self, checkpointer) -> None:
        graph = _build_tiny_graph(checkpointer)
        graph.invoke({"messages": [], "counter": 0}, config={"configurable": {"thread_id": "t1"}})

        config = {"configurable": {"thread_id": "t1"}}
        current = graph.get_state(config)
        assert current.values["messages"] == ["step-0"]
        assert current.values["counter"] == 1

    def test_state_persists_across_invocations(self, checkpointer) -> None:
        graph = _build_tiny_graph(checkpointer)
        config = {"configurable": {"thread_id": "t2"}}
        graph.invoke({"messages": [], "counter": 0}, config=config)
        graph.invoke({"messages": [], "counter": 1}, config=config)

        current = graph.get_state(config)
        assert current.values["messages"] == ["step-0", "step-1"]
        assert current.values["counter"] == 2

    def test_threads_are_isolated(self, checkpointer) -> None:
        graph = _build_tiny_graph(checkpointer)
        graph.invoke({"messages": [], "counter": 0}, config={"configurable": {"thread_id": "a"}})
        graph.invoke({"messages": [], "counter": 5}, config={"configurable": {"thread_id": "b"}})

        a = graph.get_state({"configurable": {"thread_id": "a"}})
        b = graph.get_state({"configurable": {"thread_id": "b"}})
        assert a.values["counter"] == 1
        assert b.values["counter"] == 6

    def test_list_and_filter(self, checkpointer) -> None:
        graph = _build_tiny_graph(checkpointer)
        graph.invoke({"messages": [], "counter": 0}, config={"configurable": {"thread_id": "t3"}})
        graph.invoke({"messages": [], "counter": 1}, config={"configurable": {"thread_id": "t3"}})

        tuples = list(checkpointer.list({"configurable": {"thread_id": "t3"}}))
        assert len(tuples) >= 2
        # Newest first.
        assert tuples[0].checkpoint["id"] != tuples[-1].checkpoint["id"]

    def test_parent_chain(self, checkpointer) -> None:
        graph = _build_tiny_graph(checkpointer)
        config = {"configurable": {"thread_id": "t4"}}
        graph.invoke({"messages": [], "counter": 0}, config=config)
        graph.invoke({"messages": [], "counter": 1}, config=config)

        newest = list(checkpointer.list(config))[0]
        assert newest.parent_config is not None
        assert newest.parent_config["configurable"]["checkpoint_id"]

    def test_delete_thread(self, checkpointer) -> None:
        graph = _build_tiny_graph(checkpointer)
        config = {"configurable": {"thread_id": "gone"}}
        graph.invoke({"messages": [], "counter": 0}, config=config)
        checkpointer.delete_thread("gone")

        assert checkpointer.get_tuple(config) is None
        assert list(checkpointer.list(config)) == []

    def test_research_graph_checkpointer_roundtrip(self, graph) -> None:
        """Integration: the real research graph persists turns per thread."""
        config = {"configurable": {"thread_id": "conv-1"}}
        first = graph.invoke(
            {"question": "Explain RAG briefly", "history": [], "iteration": 0,
             "missing_information": "", "max_tool_calls": 6},
            config=config,
        )
        assert first["final_answer"]

        second = graph.invoke(
            {"question": "and now about agents", "history": [], "iteration": 0,
             "missing_information": "", "max_tool_calls": 6},
            config=config,
        )
        assert second["final_answer"]
        # Checkpointer kept both super-steps on the same thread.
        tuples = list(graph.checkpointer.list(config))
        assert len(tuples) >= 2