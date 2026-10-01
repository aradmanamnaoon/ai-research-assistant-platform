# Evaluating LLM Applications

Evaluating LLM systems requires different methods than traditional software testing, because outputs are probabilistic and natural language quality is subjective. This document covers the standard toolbox.

## Evaluation Levels

1. **Unit-level**: test individual components — prompt templates render correctly, parsers extract valid JSON, tools return expected shapes with mocked dependencies.
2. **Trajectory-level**: for agents, evaluate *how* the answer was produced — which tools were called, in what order, whether the loop terminated within budget.
3. **End-to-end**: run the full pipeline on a benchmark dataset and score final answers.

## Datasets

An evaluation dataset contains input records and usually reference outputs. Good practice:

- Cover the main task types the system must handle (retrieval-heavy, tool-heavy, conversational).
- Include adversarial and out-of-scope questions to measure robustness.
- Keep 20–100 examples for regression tracking; more for research claims.

## Metrics

**Deterministic metrics** are cheap, reproducible, and should run on every commit:

- Relevance proxies (keyword overlap between question and answer).
- Citation validity (do bracketed citations resolve to listed sources?).
- Groundedness heuristics (penalize unsupported superlatives, reward citations).
- Latency and token cost percentiles.

**LLM-as-judge metrics** use a strong model to grade subjective qualities:

- Faithfulness: is every claim supported by the provided context?
- Helpfulness: does the answer address the user's actual question?
- Correctness: does it agree with a reference answer?

Judges should return structured scores with a comment, and their own reliability can be calibrated against human-labeled samples.

## Platforms

LangSmith provides tracing (every LLM/tool call recorded), datasets, experiments, and both code and LLM evaluators. Tracing is enabled with environment variables (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`), and every run can be inspected as a nested trace of steps.

## Continuous Evaluation

Wire evaluation into CI: run the deterministic suite on pull requests, run judge-based suites nightly or on demand, and track aggregate scores over time so regressions surface early. Store experiment results with metadata (model version, prompt version) so score changes can be attributed to changes.