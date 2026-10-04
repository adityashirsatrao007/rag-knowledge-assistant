<div align="center">

# RAG Knowledge Assistant

**Retrieval-Augmented Generation over your documents, with grounded citations.**

> **⚡ Impact:** runs offline with no model downloads or API keys · hybrid BM25 + dense (Chroma) · every answer grounded with source citations
>
> 🖥️ **Live demo:** <https://ssl-rob-elected-you.trycloudflare.com> — FastAPI app, interactive `/docs`

FastAPI · Chroma · sentence-transformers · BM25 hybrid retrieval · Docker

[![CI](https://github.com/adityashirsatrao007/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/adityashirsatrao007/rag-knowledge-assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

A production-shaped RAG service that chunks and embeds documents into a **Chroma** vector store, retrieves with **hybrid search (dense + BM25)** to answer questions, and returns every answer with **citations back to the source document**.

Works offline out of the box: it prefers `sentence-transformers` embeddings when installed, and falls back to a zero-download hashing embedder so the full pipeline runs with no model downloads or API keys. Plug in any OpenAI-compatible endpoint (`OPENAI_API_KEY` / `LLM_BASE_URL`) for LLM-backed answers.

![Curved library walls of books around a staircase, standing in for retrieval across a document corpus](figures/retrieval-corpus.jpg)

*Image: StockSnap.io, released under CC0.*

## Features

- **Hybrid retrieval** — vector similarity + BM25 lexical scoring, reranked before generation
- **Grounded answers** — every response carries source citations and metadata filters
- **Pluggable LLM layer** — OpenAI, any OpenAI-compatible API, or a free local extractive answerer
- **RAG evaluation harness** — `scripts/evaluate.py` scores retrieval hit-rate against `data/eval_questions.json`, the committed starter set
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

`scripts/evaluate.py` reports retrieval hit-rate against `data/eval_questions.json`. The committed starter set is 3 questions over the 2 documents in `data/sample_docs/`; add to both files to measure a larger corpus.

## Project layout

```
app/                 FastAPI application (ingest, ask, RAG orchestration)
scripts/             CLI entry points (ingest, evaluate)
data/                sample documents + evaluation question set
tests/               RAG smoke tests (run in CI)
```

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome! For significant changes, please open an issue first to discuss the proposal.

1. Fork the repository and create a feature branch.
2. Make your changes (add tests where applicable).
3. Ensure the CI workflow passes.
4. Open a pull request with a clear description.

This project is released under the MIT License — see [`LICENSE`](LICENSE).
