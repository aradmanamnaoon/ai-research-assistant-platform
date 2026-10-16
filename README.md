# AI Research Assistant Platform

An **experimental multi-agent research assistant reference implementation** built with **LangGraph**, **LangChain**, **ChromaDB RAG** and optional **LangSmith** observability, served by **FastAPI**.

> **Evaluation status:** This is not an independently validated research-answering system. Citations are formatted and reference-number checked, but the checks do **not** establish that a cited document supports a claim. The documented fake-model baseline reported `citation_validity = 0.00`; that is a reason to inspect results, not to infer real-model reliability. No fresh end-to-end test run is claimed here. See [evaluation limitations](docs/evaluation.md).

Give it a question and it decides what to do: retrieve from your uploaded documents, search Wikipedia/arXiv/the web, run utility tools — then synthesizes a cited, grounded answer, with a quality-gate loop that refines weak drafts.

```text
                         ┌──────────────────────────┐
                         │        FastAPI API       │
                         │ /chat /chat/stream /docs │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────▼────────────────────────┐
              │           LangGraph Research Graph             │
              │                                                │
              │  START ──► planner ──┬──► retriever ──────┐    │
              │                      ├──► tool_agent ─────┼──► synthesizer ──► reflector ──┐
              │                      └──► researcher ─────┘        ▲              │        │
              │                                    ▲               │              └─ loop ─┘──► END
              │                                    └───────────────┴───────────────────────┘
              └────────────────────────────────────────────────┘
                 parallel branches merge via state reducers
```

Compatible branches run **in parallel** (fan-out) and merge through state reducers. Everything is checkpointed to SQLite, so conversations survive restarts and can be inspected via time-travel.

**Tried an authorized synthetic example?** [Report first-run or citation feedback](https://github.com/aradmanamnaoon/ai-research-assistant-platform/issues/new?template=first-run-feedback.yml), including reproducible failures. Do not upload private documents, prompts, or secrets. Reports are voluntary; they are not evidence of independently verified answer quality.

## Features

- **Planner agent** — classifies each question and emits a structured JSON research plan (retrieval? web? tools? steps?), with a heuristic fallback so planning failures never crash the graph
- **Retrieval agent** — semantic search over ChromaDB (multi-query + dedup + LLM relevance pass) with **local ONNX embeddings** — zero embedding API needed
- **Research agent** — ReAct tool loop (`create_react_agent`) over Wikipedia, arXiv and optional Tavily; never repeats prior searches; bounded by a recursion limit
- **Tool execution agent** — planner-requested tools with per-call error isolation (safe calculator, persistent research notebook)
- **Final response agent** — citation-formatted synthesis with parsed `[n]` references and a Sources section; reference-number validity is not factual attribution
- **Reflector** — LLM gap analysis with a bounded refinement loop back to the researcher
- **Parallel fan-out** — planner-selected branches execute concurrently
- **Persistent state** — custom `BaseCheckpointSaver` (SQLite) + SQLAlchemy conversation store
- **Custom LangChain chat model** — `BaseChatModel` on the OpenAI SDK: works with OpenAI, OpenRouter, vLLM, Ollama or any OpenAI-compatible endpoint
- **SSE streaming** — per-node progress events for UIs
- **LangSmith** — full tracing plus an evaluation suite (deterministic + LLM-as-judge evaluators, dataset seeding, `evaluate()` experiments)
- **Offline mode** — `LLM_PROVIDER=fake` runs the entire platform deterministically with zero API keys (used by the 100+ test suite)

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 1.x (`StateGraph`, conditional edges, fan-out, checkpointing) |
| Agent framework | LangChain core (`BaseChatModel`, `@tool`, `BaseRetriever`, prompts) |
| LLM | OpenAI-compatible Chat Completions via a custom adapter |
| RAG | ChromaDB + local ONNX embeddings (`all-MiniLM-L6-v2`) |
| Observability | LangSmith (tracing, datasets, experiments) |
| API | FastAPI + Uvicorn (REST + SSE), OpenAPI docs |
| Persistence | Custom SQLite checkpointer, SQLAlchemy, JSONL notebook |
| Testing | pytest — 103 tests, fully offline/hermetic |

## Project Structure

```
app/
├── agents/          # planner, researcher, retrieval, tool, synthesizer nodes
├── api/             # FastAPI app, routes, schemas, dependencies
├── evaluation/      # evaluators, dataset management, runner
├── graphs/          # ResearchState + graph assembly, reflector, routing
├── llm/             # custom OpenAI-compatible chat model + factory + fakes
├── memory/          # SQLite checkpointer (BaseCheckpointSaver) + conversation store
├── retrieval/       # splitter, ChromaDB vector store, retriever, ingestion
├── services/        # ResearchService, DocumentService facades
├── tools/           # wikipedia, arxiv, tavily, calculator, notes, registry
└── config.py        # pydantic-settings, single source of configuration
data/documents/      # example knowledge documents (seeded markdown)
evaluation_examples/ # evaluation dataset
scripts/             # ingest, seed_langsmith, run_evaluation
tests/               # unit + integration + API tests (offline)
docs/                # architecture.md, api.md, evaluation.md
```

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/aradmanamnaoon/ai-research-assistant-platform
cd ai-research-assistant-platform
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Setup

Edit `.env` — the only required value for the full experience:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...            # any OpenAI-compatible endpoint
# OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional
```

Everything else is optional:

- `TAVILY_API_KEY` — live web search (Wikipedia/arXiv need no keys)
- `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` — tracing & evaluation experiments
- `LLM_PROVIDER=fake` — fully offline deterministic mode (no keys at all)

## Usage

### 1. Ingest the example knowledge base

```bash
python scripts/ingest.py
```

### 2. Start the API

```bash
uvicorn app.api.main:app --reload
# or: python -m app.api.main
```

Interactive docs: http://localhost:8000/docs

### 3. Ask questions

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation and why does it matter?"}'
```

Continue the conversation by passing the returned `thread_id`:

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does it reduce hallucination?", "thread_id": "a1b2c3d4e5f6"}'
```

Upload your own documents and ask about them:

```bash
curl -F "file=@my-paper.pdf" http://localhost:8000/api/v1/documents/upload
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does my uploaded document say about ...?"}'
```

### 4. Python client

```python
import httpx

with httpx.Client(base_url="http://localhost:8000") as client:
    result = client.post("/api/v1/chat", json={"question": "Explain the ReAct pattern"}).json()
    print(result["answer"])
    print("Plan:", result["plan"]["steps"])
    print("Citations:", result["citations"])
```

### 5. Run offline (zero keys)

```bash
LLM_PROVIDER=fake uvicorn app.api.main:app --reload
```

## Evaluation

The suite runs the **complete pipeline** per example and scores it with deterministic metrics (relevance proxy, citation validity, groundedness heuristic) plus LLM-as-judge evaluators (helpfulness, correctness).

```bash
# Offline mode (no LLM / LangSmith required)
python scripts/run_evaluation.py --no-upload

# Full LangSmith experiments with per-step traces
python scripts/seed_langsmith.py && python scripts/run_evaluation.py
```

Previously documented example scores for the scripted offline mode (NOT rerun or independently verified in this documentation update; these demonstrate the test harness, not answer quality):

| Evaluator | Score (offline baseline) |
|---|---|
| response_relevance | 0.02 |
| citation_validity | 0.00 |
| groundedness_heuristic | 0.50 |
| helpfulness_judge | 0.50 |

With a real LLM configured (`LLM_PROVIDER=openai`), the same commands produce LangSmith experiments with drill-down traces for every run. See [docs/evaluation.md](docs/evaluation.md).

## Testing

```bash
pytest tests/          # 103 tests, fully offline
```

Covers: text splitting, vector store idempotency & relevance, safe calculator, notes, planner parsing (JSON + heuristic fallbacks), graph orchestration (fan-out, loop, error resilience, streaming), the custom checkpointer (persistence, thread isolation, parent chain, deletion), the custom chat model (message conversion, tool-call parsing, bound tool schemas, stubbed HTTP), all evaluators, and the full API surface.

## Deployment

### Docker

```bash
docker compose up --build        # API on :8000, data persisted in a named volume
```

### Any cloud (single container)

```bash
docker build -t ai-research-assistant .
docker run -p 8000:8000 --env-file .env -v research-data:/app/data ai-research-assistant
```

The image ships the example documents; run `python scripts/ingest.py` inside the container (or `docker compose exec api python scripts/ingest.py`) to index them.

## Architecture Documentation

- [docs/architecture.md](docs/architecture.md) — graph topology, agent roles, persistence design, error-handling matrix, design decisions
- [docs/api.md](docs/api.md) — full REST/SSE reference with examples
- [docs/evaluation.md](docs/evaluation.md) — evaluator suite and regression workflow

## Future Improvements

- Streaming tokens end-to-end (`stream_mode="messages"`) once the model adapter supports chunked streaming
- Hybrid search (BM25 + vectors) and cross-encoder re-ranking
- Human-in-the-loop approval for tool calls via LangGraph `interrupt()`
- Postgres checkpointer for multi-instance deployments
- Agent trajectory evaluation (tool-choice and loop-efficiency scoring)
- Minimal React UI over the SSE stream

## License

[MIT](LICENSE)