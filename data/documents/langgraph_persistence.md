# LangGraph: Graph-Based Agent Orchestration

LangGraph is a low-level orchestration framework for building stateful, multi-step LLM applications. Applications are modeled as directed graphs: **nodes** perform work (LLM calls, tool execution, data transforms) and **edges** define execution order, including conditional branches.

## State

Every graph has a shared state schema, typically a TypedDict. Fields accumulate through **reducers**: `operator.add` appends to lists, custom reducers implement merge logic. Fields without a reducer are overwritten by the last write. Nodes return *partial state updates* rather than mutating state.

## Persistence and Checkpointing

LangGraph persists graph state after every super-step through a **checkpointer**. Built-in options range from in-memory (`MemorySaver`) to SQLite and Postgres backends. A custom checkpointer only needs to implement the `BaseCheckpointSaver` interface: `get_tuple`, `list`, `put`, and `put_writes`.

Checkpoints enable:

- **Thread persistence**: conversations keyed by `thread_id` survive process restarts.
- **Time travel**: replay or fork execution from any prior checkpoint.
- **Human-in-the-loop**: the graph can pause (`interrupt`) and resume later.

## Control Flow

- **Conditional edges** route dynamically based on state (a router function returns the next node name or `END`).
- **Fan-out**: a conditional-edge router may return a *list* of node names; compatible branches run in parallel and their outputs merge through reducers.
- **`Command`**: a node can update state and choose the next node in one return value.
- **Subgraphs**: a compiled graph can be embedded as a node in a larger graph, or exposed as a tool.

## Streaming

LangGraph supports multiple stream modes simultaneously: `values` (full state after each step), `updates` (state deltas), `messages` (LLM tokens), and `custom` (progress events emitted from inside nodes with the stream writer). Server-sent events (SSE) endpoints commonly forward `updates` mode chunks to a UI so users see each agent step complete in real time.

## Production Concerns

- Set `recursion_limit` to bound loops.
- Attach `RetryPolicy` to nodes that call flaky external services.
- Use durable checkpointers (SQLite/Postgres) for anything that must survive restarts.
- Emit structured events so API clients can render progress.