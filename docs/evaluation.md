# Evaluation Guide

This project ships a runnable evaluation suite with two modes and four
evaluators.

## Evaluators

| Evaluator | Type | Measures |
|---|---|---|
| `response_relevance` | deterministic | Fraction of the question's significant keywords that appear in the answer |
| `citation_validity` | deterministic | Every `[n]` citation resolves to a listed source |
| `groundedness_heuristic` | deterministic | Rewards citations, penalizes unsupported superlatives ("always", "obviously"...) |
| `helpfulness_judge` | LLM-as-judge | 0–1 helpfulness of the answer for the question |
| `correctness_judge` | LLM-as-judge | Agreement with the reference answer (needs references) |

Judges degrade to a neutral score with an explanatory comment when no LLM
is reachable, so runs never hard-fail.

## Dataset

`evaluation_examples/examples.json` — 8 examples covering RAG, agent
patterns, LangGraph persistence and evaluation practice, each with a
reference answer and topic tags.

## Running

```bash
# Local offline mode (no LangSmith account needed)
python scripts/run_evaluation.py --no-upload

# Quick subset
python scripts/run_evaluation.py --no-upload --limit 3

# LangSmith mode: experiments, traces, persistent results
# (requires LANGSMITH_API_KEY + LANGSMITH_TRACING=true in .env)
python scripts/seed_langsmith.py      # create/sync the dataset
python scripts/run_evaluation.py      # runs evaluate() and prints the experiment URL
```

Local mode output:

```
Evaluation report
--------------------------------------------------
Mode: local
Examples: 8
  response_relevance           0.83
  citation_validity            0.91
  groundedness_heuristic       0.95
  helpfulness_judge            0.78
```

## How it works

The target function (`app/evaluation/runner.py::research_target`) runs
the complete research pipeline per question, so evaluation scores the
end-to-end system. In LangSmith mode, `langsmith.evaluate()` attaches
every evaluator to each experiment row; every run is fully traced (planner
→ retrieval → researcher tool loop → synthesis), so a low score can be
drilled down to the exact failing step.

## Extending

1. **Add examples**: append to `evaluation_examples/examples.json`
   (`question`, `reference`, `topic`, `expected_tools`).
2. **Add an evaluator**: implement the `(inputs, outputs,
   reference_outputs) -> {"key", "score", "comment"}` contract in
   `app/evaluation/evaluators.py` and register it in
   `DEFAULT_EVALUATORS`.
3. **Regression tracking**: run the deterministic suite in CI on every
   pull request; track judge-based suites nightly and compare experiment
   scores over time (tag experiments with prompt/model versions via
   `evaluate(metadata=...)`).