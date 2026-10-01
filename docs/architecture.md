# Architecture

## System Overview

```
                                  ┌───────────────────────┐
                                  │      FastAPI API      │
                                  │  /chat  /chat/stream  │
                                  │  /documents  /threads │
                                  └──────────┬────────────┘
                                             │
                                  ┌──────────▼────────────┐
                                  │   ResearchService     │
                                  │ (facade + history)    │
                                  └──────────┬────────────┘
                                             │
        ┌────────────────────────────────────▼────────────────────────────────────┐
        │                     LangGraph  Research Graph                          │
        │                                                                        │
        │   START ──► planner ──┬──► retriever ─────┐                            │
        │                       ├──► tool_agent ────┼──► synthesizer ──► reflector
        │                       └──► researcher ────┘         ▲            │      │
        │                                    ▲                │            │      │
        │                                    └────────────────┴────────────┘      │
        │                                        (bounded refinement loop)   END  │
        └────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Planner agent (`app/agents/planner.py`)

Classifies the question and emits a structured `ResearchPlan` (JSON):
whether to retrieve from the document library, run web research, execute
specific tools, and in what order. Parsed with a tolerant JSON extractor;
falls back to a keyword heuristic so planning failures never crash the
graph. The planner sees recent conversation history for context.

### 2. Retrieval agent (`app/agents/retrieval_agent.py`)

Runs semantic search over the ChromaDB collection (multi-query: the raw
question plus the planner's optimized query), dedupes passages, then uses
an LLM relevance pass to distill what is actually useful.

### 3. Research agent (`app/agents/researcher.py`)

A compiled `create_react_agent` (ReAct tool loop) bound to
`wikipedia_search`, `arxiv_search` and `tavily_search`. The LLM decides
when to call tools and when to stop. It receives already-gathered
findings so it never repeats searches, and a recursion limit bounds the
loop. Tool failures are returned to the model as observations.

### 4. Tool execution agent (`app/agents/tool_agent.py`)

Executes the planner's explicitly requested tools (`calculator`,
`save_note`, ...) with per-call error isolation, then interprets the
results with an LLM.

### 5. Final response agent (`app/agents/synthesizer.py`)

Assembles numbered sources from documents/research/tool results, prompts
for a directly-answered, citation-grounded synthesis, and parses the
`[n]` citations back into a structured `citations` list. Falls back to a
context-only answer if the LLM call fails.

### 6. Reflector (`app/graphs/research_graph.py`)

Gap analysis on the draft. When the answer has obvious gaps and the
iteration budget allows, the graph routes back to the researcher — a
bounded refinement loop (`max_graph_iterations`).

## Parallel Fan-out

The planner's conditional edge returns a **list** of branch targets, so
LangGraph executes compatible branches (e.g. retrieval *and* web research)
concurrently. Their outputs merge through `operator.add` reducers defined
in `ResearchState`.

## Persistence

| Store | Technology | Purpose |
|---|---|---|
| Graph state | `SqliteCheckpointSaver` (custom `BaseCheckpointSaver`) | Full state per super-step keyed by `thread_id`; survives restarts |
| Conversation turns | SQLAlchemy + SQLite | Fast history rendering, thread listing |
| Documents | ChromaDB (persistent, local ONNX embeddings) | RAG knowledge base |
| Research notes | JSONL | Persistent notebook exposed via `save_note` / `list_notes` tools |

The custom checkpointer (`app/memory/sqlite_checkpointer.py`) implements
`get_tuple`, `list`, `put`, `put_writes` and `delete_thread`, serializing
with LangGraph's `JsonPlusSerializer` so arbitrary channel values
(including LangChain messages) round-trip safely.

## LLM Layer

`app/llm/base.py` implements a custom `BaseChatModel` on the OpenAI SDK:
message translation, `bind_tools` (OpenAI function-calling schemas), tool
call parsing, single-chunk streaming, and exponential-backoff retries on
transient errors. Any OpenAI-compatible endpoint works (OpenRouter, vLLM,
Ollama, Azure gateways). `LLM_PROVIDER=fake` swaps in a deterministic
model for tests and offline demos.

## Observability

LangSmith traces every LLM call, tool call, and graph step automatically
(LangChain callback integration), plus `@traceable` spans on the service
and evaluation layers. Configure with `LANGSMITH_API_KEY`,
`LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT`.

## Error-handling Strategy

| Error type | Handling |
|---|---|
| Transient LLM/HTTP errors | Retries with backoff (`ChatOpenAICompatible`, `RetryPolicy`-capable nodes) |
| Tool failures | Isolated per call; errors become observations for the LLM |
| Planning/synthesis failure | Heuristic plan / context-only fallback |
| Gap analysis failure | Draft accepted (gate never blocks) |
| Unexpected | Bubbles up to FastAPI → HTTP 502 |

## Design Decisions

1. **Custom chat model instead of `langchain-openai`** — removes a dependency, demonstrates the full LangChain model contract, and keeps the stack exactly matched to what's installed.
2. **Custom SQLite checkpointer** — first-party persistence without the optional `langgraph-checkpoint-sqlite` package; also a worked example of the `BaseCheckpointSaver` contract.
3. **Chroma's local ONNX embeddings** — RAG works offline with zero embedding API keys; the store is pluggable for hosted embeddings.
4. **Public-source tools first** (Wikipedia/arXiv, no keys) with Tavily as an optional upgrade — the demo works from a fresh clone with a single API key (or none).
5. **Deterministic fake model** — the entire platform (graph, API, evaluation) is testable and demoable offline, which keeps the test suite fast, hermetic and CI-friendly.