"""RAG Knowledge Assistant — FastAPI application.

Hybrid retrieval (vector + BM25) over a Chroma vector store, with a pluggable
LLM provider. Falls back to extractive answering so the demo works without
an API key.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ingest import Ingester
from app.rag import RAGEngine


class AskRequest(BaseModel):
    query: str
    k: int = 5
    collection: str = "documents"


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_time_ms: float
    provider: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store_path = os.getenv("STORE_PATH", "./chroma_db")
    app.state.ingester = Ingester(app.state.store_path)
    app.state.rag = RAGEngine(
        store_path=app.state.store_path,
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model=os.getenv("LLM_MODEL"),
    )
    yield
    app.state.rag.close()


app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "ok", "service": "rag-knowledge-assistant"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-knowledge-assistant"}


@app.get("/stats")
def stats():
    return app.state.ingester.stats()


@app.post("/ingest")
def ingest(path: str):
    count = app.state.ingester.ingest_path(path)
    return {"chunks_ingested": count, "total_chunks": app.state.ingester.count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = app.state.rag.answer(
            req.query, k=req.k, collection=req.collection
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return AskResponse(**result)
