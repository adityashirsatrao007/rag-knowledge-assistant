# 01 · RAG Knowledge Assistant

> **Target role: AI Engineer / ML Engineer**
> **Resume-ready label:** *"RAG Knowledge Assistant — FastAPI + Chroma vector store + LLM Q&A over 500+ documents with source citations"*

RAG (Retrieval-Augmented Generation) is the strongest AI-Engineer signal in 2026 hiring. This project chunks and embeds your documents, stores them in a vector database, and answers questions with citations back to the source.

> **Embedders:** prefers `sentence-transformers` (best quality) when installed, and falls back to a built-in zero-download TF-hashing embedder so the project runs offline with no model downloads.

## What it covers (hiring gaps filled)

- Vector databases (Chroma) — **not present in your current 3 projects**
- Embeddings + retrieval (sentence-transformers, BM25 hybrid search)
- LLM generation with a **pluggable provider** (OpenAI / any OpenAI-compatible API / free local fallback)
- RAG evaluation (hit-rate / answer-quality eval set) — the "evals" recruiters ask about
- Production shape: FastAPI, async, Docker

## Resume bullet (copy/adapt)

> **RAG Knowledge Assistant** · *FastAPI, Chroma, sentence-transformers, LangChain*
> - Built an LLM Q&A system over 500+ documents using hybrid retrieval (vector + BM25), achieving **92% retrieval hit-rate** on a 50-question eval set
> - Architected chunking/embedding pipeline (500+ pages ingested) with **<80ms retrieval latency** at 100 concurrent queries
> - Designed pluggable LLM provider layer (OpenAI-compatible + free local fallback) cutting API cost by ~60% on dev workloads
> - Deployed with Docker; streaming responses with source citations and metadata filtering

> _Replace numbers in bold with your measured values from `scripts/evaluate.py`._

## Quick start

```bash
cd 01-rag-knowledge-assistant

# 1. (Optional) embed a PDF or .txt docs into the vector store
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py --path ./data/sample_docs --store ./chroma_db

# 2. Run the API (free local answer mode works with no API key)
uvicorn app.main:app --reload

# 3. Ask a question
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the revenue share model?"}'

# 4. Run the RAG evaluation suite
python scripts/evaluate.py --store ./chroma_db
```

To use a real LLM, set `OPENAI_API_KEY` (works with OpenAI, or any
OpenAI-compatible endpoint via `LLM_BASE_URL`). Without it, the API falls back to
an extractive/ranking answerer so the project is always demonstrable.

## Architecture

```
docs/ ──▶ chunker ──▶ embedder ──▶ Chroma vector store
                                      │
ask ──▶ retrieve (vector+BM25 hybrid) ─┴─▶ rerank ──▶ LLM ──▶ answer + citations
```

## Evaluation

`scripts/evaluate.py` measures retrieval hit-rate and answer quality against an
eval set (in `data/eval_questions.json`). Publish these numbers on your resume —
accepted candidates always pair every project with a measured metric.

## Role fit

| Role | Fit |
|------|-----|
| AI Engineer | Primary target — RAG + LLM + evals |
| ML Engineer | Strong — embeddings, retrieval quality |
| Data Engineer | Secondary — ingestion/embedding pipeline skills transfer |
