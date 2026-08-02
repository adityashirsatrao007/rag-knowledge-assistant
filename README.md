<div align="center">

# RAG Knowledge Assistant

**Retrieval-Augmented Generation over your documents, with grounded citations.**

FastAPI · Chroma · sentence-transformers · BM25 hybrid retrieval · Docker

[![CI](https://github.com/adityashirsatrao007/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/adityashirsatrao007/rag-knowledge-assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

A production-shaped RAG service that chunks and embeds documents into a **Chroma** vector store, retrieves with **hybrid search (dense + BM25)** to answer questions, and returns every answer with **citations back to the source document**.

Works offline out of the box: it prefers `sentence-transformers` embeddings when installed, and falls back to a zero-download hashing embedder so the full pipeline runs with no model downloads or API keys. Plug in any OpenAI-compatible endpoint (`OPENAI_API_KEY` / `LLM_BASE_URL`) for LLM-backed answers.

## Features

- **Hybrid retrieval** — vector similarity + BM25 lexical scoring, reranked before generation
- **Grounded answers** — every response carries source citations and metadata filters
- **Pluggable LLM layer** — OpenAI, any OpenAI-compatible API, or a free local extractive answerer
- **RAG evaluation harness** — `scripts/evaluate.py` measures retrieval hit-rate and answer quality against a 50-question eval set
- **Production shape** — FastAPI, async, streaming responses, Docker + docker-compose

## Architecture

```
docs/ ──▶ chunker ──▶ embedder ──▶ Chroma vector store
                                       │
ask ──▶ retrieve (vector + BM25) ──▶ rerank ──▶ LLM ──▶ answer + citations
```

## Quick start

```bash
cd rag-knowledge-assistant

# 1. (Optional) embed your own PDFs/txt into the vector store
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py --path ./data/sample_docs --store ./chroma_db

# 2. Run the API (free local answer mode, no API key needed)
uvicorn app.main:app --reload

# 3. Ask a question
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the revenue share model?"}'

# 4. Run the evaluation suite
python scripts/evaluate.py --store ./chroma_db
```

Or run everything with Docker:

```bash
docker compose up --build
```

Set `OPENAI_API_KEY` (or `LLM_BASE_URL` for any OpenAI-compatible endpoint) to use a real LLM. Without it the API falls back to an extractive answerer so the project is always demonstrable.

## Evaluation

`scripts/evaluate.py` reports retrieval hit-rate and answer quality against `data/eval_questions.json`. Measured results are published in the repo so retrieval quality is verifiable, not just claimed.

## Project layout

```
app/                 FastAPI application (ingest, ask, RAG orchestration)
scripts/             CLI entry points (ingest, evaluate)
data/                sample documents + evaluation question set
tests/               RAG smoke tests (run in CI)
```

## License

[MIT](LICENSE)
