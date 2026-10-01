# API Reference

Base URL: `http://localhost:8000` — interactive docs at `/docs` (Swagger) and `/redoc`.

## Chat

### `POST /api/v1/chat`

Run the full research pipeline for one question.

**Request**

```json
{
  "question": "What is retrieval-augmented generation?",
  "thread_id": "optional-conversation-id"
}
```

A new `thread_id` (12-char hex) is generated when omitted. Pass the same
id to continue a conversation.

**Response `200`**

```json
{
  "thread_id": "a1b2c3d4e5f6",
  "question": "What is retrieval-augmented generation?",
  "answer": "Retrieval-augmented generation (RAG) grounds LLM answers ... [1]\n\nSources:\n[1] Uploaded document: retrieval_augmented_generation.md",
  "citations": ["Uploaded document: retrieval_augmented_generation.md"],
  "sources": ["Uploaded document: retrieval_augmented_generation.md"],
  "plan": {
    "reasoning": "Question asks about a concept covered in the document library",
    "needs_retrieval": true,
    "needs_web_research": false,
    "needs_tools": false,
    "tools_to_run": [],
    "steps": ["retrieve documents", "synthesize"],
    "research_query": "What is retrieval-augmented generation?"
  },
  "steps_taken": {"retrieved_docs": 4, "research_notes": 0, "tool_results": 0, "iterations": 1},
  "error": null,
  "completed_at": "2026-10-01T12:00:00+00:00"
}
```

**Errors**: `422` validation, `502` pipeline failure.

### `POST /api/v1/chat/stream`

Same pipeline, streamed as **Server-Sent Events**. One `node_update`
event per completed agent node, then a final `done` event with the full
payload.

```
data: {"event": "node_update", "node": "planner", "data": {"plan": {...}}}
data: {"event": "node_update", "node": "retriever", "data": {"retrieved_docs": [...]}}
data: {"event": "node_update", "node": "synthesizer", "data": {...}}
data: {"event": "done", "payload": {...full ChatResponse...}}
```

### `GET /api/v1/chat/threads`

List all threads: `[{"thread_id", "message_count", "last_activity"}]`.

### `GET /api/v1/chat/{thread_id}/history?limit=50&offset=0`

Stored turns of one thread (oldest first).

### `DELETE /api/v1/chat/{thread_id}`

Delete a thread's stored turns.

## Documents

### `POST /api/v1/documents/upload`

Multipart upload; ingest a file into the RAG index.
Allowed: `.md .txt .csv .json .rst .html .pdf`. Max 10 MB (configurable).
Re-uploading the same file name replaces its chunks (idempotent).

```bash
curl -F "file=@report.md" http://localhost:8000/api/v1/documents/upload
```

**Response `200`**: `{"source": "report.md", "chunks": 12, "total_chunks": 87}`
**Errors**: `413` too large, `415` unsupported type, `400` ingest error.

### `POST /api/v1/documents/text`

Ingest raw text: `{"text": "...", "source": "notes.md"}`.

### `GET /api/v1/documents`

List ingested sources: `[{"source", "chunks"}]`.

### `DELETE /api/v1/documents/{source}`

Delete all chunks of a document. `404` when unknown.

### `POST /api/v1/documents/search`

Raw semantic search (debug): `{"query": "...", "top_k": 5}` →

```json
{"query": "...", "results": [{"text": "...", "metadata": {...}, "distance": 0.31, "score": 0.69}]}
```

## System

### `GET /api/v1/health`

```json
{"status": "ok", "version": "1.0.0", "llm_provider": "openai", "model_name": "gpt-4o-mini", "documents_indexed": 87, "langsmith_enabled": true}
```

### `GET /api/v1/tools`

The agent tool catalog.

## Examples

Ask and continue a conversation:

```bash
# Turn 1
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'

# Turn 2 (same thread for context)
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does it reduce hallucination?", "thread_id": "a1b2c3d4e5f6"}'
```

Python client:

```python
import httpx

with httpx.Client(base_url="http://localhost:8000") as client:
    response = client.post("/api/v1/chat", json={"question": "Explain ReAct agents"})
    print(response.json()["answer"])
```