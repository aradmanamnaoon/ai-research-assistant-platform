"""API integration tests (isolated services, fake model)."""

from __future__ import annotations

import uuid


class TestSystem:
    def test_health(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["documents_indexed"] >= 2
        assert body["llm_provider"] in ("fake", "openai")

    def test_tool_catalog(self, client) -> None:
        response = client.get("/api/v1/tools")
        assert response.status_code == 200
        names = {t["name"] for t in response.json()["tools"]}
        assert "wikipedia_search" in names
        assert "calculator" in names


class TestChat:
    def test_chat_roundtrip(self, client) -> None:
        response = client.post(
            "/api/v1/chat",
            json={"question": "What is RAG?", "thread_id": "api-test-1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["thread_id"] == "api-test-1"
        assert "plan" in body
        assert body["steps_taken"]["iterations"] >= 1

    def test_chat_generates_thread_id(self, client) -> None:
        response = client.post("/api/v1/chat", json={"question": "hello there agents"})
        assert response.status_code == 200
        assert response.json()["thread_id"]

    def test_chat_history_recorded(self, client) -> None:
        response = client.post(
            "/api/v1/chat", json={"question": "q1", "thread_id": "hist-1"}
        )
        assert response.status_code == 200
        history = client.get("/api/v1/chat/hist-1/history").json()
        roles = [m["role"] for m in history]
        assert roles == ["user", "assistant"]

    def test_chat_validation_error(self, client) -> None:
        response = client.post("/api/v1/chat", json={"question": ""})
        assert response.status_code == 422

    def test_threads_listing(self, client) -> None:
        client.post("/api/v1/chat", json={"question": "q", "thread_id": "list-1"})
        threads = client.get("/api/v1/chat/threads").json()
        assert any(t["thread_id"] == "list-1" for t in threads)

    def test_thread_delete(self, client) -> None:
        client.post("/api/v1/chat", json={"question": "q", "thread_id": "del-1"})
        response = client.delete("/api/v1/chat/del-1")
        assert response.status_code == 200
        assert client.get("/api/v1/chat/del-1/history").json() == []

    def test_stream_endpoint_sse(self, client) -> None:
        response = client.post(
            "/api/v1/chat/stream",
            json={"question": "What is RAG?", "thread_id": "sse-1"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body = response.text
        assert "data: {" in body
        assert '"event": "done"' in body or '"event":"done"' in body


class TestDocuments:
    def test_ingest_text_and_list(self, client) -> None:
        response = client.post(
            "/api/v1/documents/text",
            json={"text": "Vector databases store embeddings for similarity search.",
                  "source": "vectors.md"},
        )
        assert response.status_code == 200
        assert response.json()["source"] == "vectors.md"

        docs = client.get("/api/v1/documents").json()
        assert any(d["source"] == "vectors.md" for d in docs)

    def test_search_preview(self, client) -> None:
        client.post(
            "/api/v1/documents/text",
            json={"text": "Fine-tuning adapts model weights to a task.",
                  "source": "finetune.md"},
        )
        response = client.post(
            "/api/v1/documents/search",
            json={"query": "adapting model weights", "top_k": 3},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert results
        assert results[0]["metadata"]["source"] == "finetune.md"

    def test_delete_document(self, client) -> None:
        client.post(
            "/api/v1/documents/text",
            json={"text": "temporary doc", "source": "temp.md"},
        )
        assert client.delete("/api/v1/documents/temp.md").status_code == 200
        assert client.delete("/api/v1/documents/temp.md").status_code == 404

    def test_upload_unsupported_type(self, client) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("malware.exe", b"binary", "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_upload_markdown(self, client, seeded_store) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("upload.md", b"# Title\n\nSome content about embeddings.", "text/markdown")},
        )
        assert response.status_code == 200
        assert response.json()["chunks"] >= 1

    def test_openapi_schema_available(self, client) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/api/v1/chat" in paths
        assert "/api/v1/documents/upload" in paths


class TestIsolation:
    def test_threads_are_isolated(self, client) -> None:
        client.post("/api/v1/chat", json={"question": "alpha", "thread_id": uuid.uuid4().hex})
        history_a = client.get("/api/v1/chat/iso-a/history").json()
        assert history_a == []