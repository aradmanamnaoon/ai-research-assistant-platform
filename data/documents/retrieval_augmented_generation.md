# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an architecture that improves large language model (LLM) responses by grounding them in external knowledge sources at inference time. Instead of relying solely on parametric knowledge learned during training, a RAG system retrieves relevant documents or passages and injects them into the model's context window before generation.

## Core Pipeline

A standard RAG pipeline has four stages:

1. **Ingestion** — source documents are loaded, cleaned, and split into chunks (typically 200–1000 tokens with 10–20% overlap).
2. **Indexing** — each chunk is converted into an embedding vector and stored in a vector database such as Chroma, FAISS, Pinecone, or Weaviate, usually with metadata (source, title, section).
3. **Retrieval** — the user's query is embedded and the k nearest chunks are retrieved using cosine similarity or inner product.
4. **Generation** — the retrieved passages are formatted into the prompt, and the model generates an answer conditioned on them, ideally with inline citations.

## Why RAG Matters

RAG addresses three chronic LLM problems:

- **Hallucination**: grounded answers cite real passages, so unsupported claims are easier to detect.
- **Knowledge cutoff**: the index can be refreshed continuously without retraining the model.
- **Private knowledge**: proprietary documents never need to be baked into model weights, keeping data governance simple.

## Retrieval Quality Techniques

Naive top-k cosine retrieval is only the baseline. Production systems add:

- **Hybrid search**: combining dense vector similarity with sparse BM25 keyword matching, then merging results with reciprocal rank fusion.
- **Re-ranking**: a cross-encoder model re-scores the top 20–50 candidates and keeps only the best few, substantially improving precision.
- **Query rewriting**: an LLM rewrites or expands the user query (multi-query retrieval, HyDE) to bridge the vocabulary gap between questions and documents.
- **Chunk strategy**: smaller chunks improve retrieval precision; larger chunks preserve context. Parent-document retrieval fetches the full parent section after matching a small child chunk.

## Evaluation

RAG quality is measured along two axes: retrieval quality (recall@k, precision@k, MRR) and generation quality (faithfulness/groundedness, answer relevance, citation validity). Faithfulness asks whether every claim in the answer is supported by the retrieved passages; relevance asks whether the answer actually addresses the question. LLM-as-judge evaluators are commonly used for both.