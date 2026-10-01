"""Tests for the vector store and retrieval layer."""

from __future__ import annotations

from langchain_core.documents import Document

from app.retrieval.ingest import ingest_text


class TestVectorStore:
    def test_ingest_and_count(self, vector_store) -> None:
        ingest_text("alpha beta gamma delta", source="a.md", store=vector_store)
        ingest_text("one two three four", source="b.md", store=vector_store)
        assert vector_store.count() >= 2

    def test_reingest_is_idempotent(self, vector_store) -> None:
        ingest_text("same text", source="same.md", store=vector_store)
        first_count = vector_store.count()
        ingest_text("same text", source="same.md", store=vector_store)
        assert vector_store.count() == first_count

    def test_search_returns_relevant_hit_first(self, seeded_store) -> None:
        results = seeded_store.search("What reduces hallucination in LLM answers?", top_k=3)
        assert results
        assert results[0]["metadata"]["source"] == "rag.md"
        assert results[0]["score"] > 0

    def test_search_result_shape(self, seeded_store) -> None:
        (hit,) = seeded_store.search("checkpoint put_writes", top_k=1)
        assert set(hit) == {"text", "metadata", "distance", "score"}
        assert isinstance(hit["score"], float)

    def test_delete_by_source(self, seeded_store) -> None:
        before = seeded_store.count()
        deleted = seeded_store.delete_by_source("rag.md")
        assert deleted > 0
        assert seeded_store.count() == before - deleted

    def test_list_sources(self, seeded_store) -> None:
        sources = seeded_store.list_sources()
        names = {s["source"] for s in sources}
        assert {"rag.md", "langgraph.md"} <= names
        assert all(s["chunks"] >= 1 for s in sources)

    def test_empty_documents_skipped(self, vector_store) -> None:
        ids = vector_store.add_documents([Document(page_content="")])
        assert ids == []


class TestRetriever:
    def test_retriever_implements_langchain_contract(self, retriever) -> None:
        from langchain_core.documents import Document

        docs = retriever.invoke("hallucination")
        assert docs
        assert all(isinstance(d, Document) for d in docs)

    def test_retriever_metadata_carries_source(self, retriever) -> None:
        docs = retriever.invoke("RAG grounds answers")
        assert docs
        assert docs[0].metadata.get("source")


class TestIngest:
    def test_ingest_text_returns_ids(self, vector_store) -> None:
        ids = ingest_text("some longer text about agents " * 10, source="x.md", store=vector_store)
        assert ids
        assert all("x.md::chunk-" in i for i in ids)

    def test_load_file_unsupported_type(self, tmp_path) -> None:
        import pytest

        from app.retrieval.ingest import load_file

        bad = tmp_path / "file.exe"
        bad.write_bytes(b"MZ")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(bad)