# LLM Agent Architectures

An LLM agent is a system in which a language model decides how to make progress on a task by choosing actions (tool calls), observing their results, and iterating until the task is complete. This document summarizes the main architectural patterns.

## The ReAct Pattern

ReAct (Reason + Act) interleaves chain-of-thought reasoning with tool use. The model receives a task, thinks about what to do next, emits a tool call, observes the result, and repeats. The loop terminates when the model produces a final answer instead of a tool call. Key properties:

- **Grounded reasoning**: intermediate thoughts cite observed evidence rather than invented facts.
- **Self-correction**: a failed or empty tool result lets the model try a different tool or query.
- **Budget control**: production implementations bound the loop with a recursion limit or maximum tool-call count to prevent runaway costs.

## Planner / Executor Decomposition

For complex tasks, a **planner** agent decomposes the user request into discrete steps, and an **executor** carries them out. Benefits:

- The plan is inspectable and auditable before any tool runs.
- Independent steps can execute in parallel (fan-out), reducing latency.
- Failure handling can be per-step rather than per-run.

The plan should be re-planned when execution reveals the original assumption was wrong, but unbounded re-planning loops are a known failure mode; cap replanning iterations.

## Multi-Agent Orchestration

Multi-agent systems split responsibilities across specialized agents — for example a researcher (web search), an analyst (math/data tools), and a writer (synthesis). A supervisor or orchestrator graph routes work between them. Orchestration frameworks such as LangGraph model each agent as a node in a directed graph with explicit state, conditional edges, and persistence.

## Memory

Agent memory has two tiers:

- **Short-term memory**: the conversation and working state of the current run, typically persisted with a checkpointer so runs can resume after failure.
- **Long-term memory**: facts and preferences that persist across sessions, stored in a database or vector store and injected into prompts when relevant.

## Failure Handling

Robust agents assume every external interaction can fail. Standard mitigations:

1. Retry transient errors (rate limits, network blips) with exponential backoff.
2. Return tool errors to the model as observations so it can adapt instead of crashing.
3. Isolate tool execution: one failing tool must not abort the whole pipeline.
4. Bound all loops: recursion limits, token budgets, and wall-clock timeouts.