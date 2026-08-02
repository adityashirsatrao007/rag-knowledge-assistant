"""Hybrid retrieval + generation engine.

Vector search (Chroma) fused with BM25 keyword search, then an LLM provider.
Falls back to an extractive answerer when no API key is configured so the
project is always demonstrable.
"""

import os
import time
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from app.ingest import get_embedding_function


class RAGEngine:
    def __init__(self, store_path="./chroma_db", llm_api_key=None,
                 llm_base_url=None, llm_model=None):
        self.client = chromadb.PersistentClient(path=store_path)
        self.embed_fn = get_embedding_function()
        self.collection = self.client.get_collection(
            "documents", embedding_function=self.embed_fn
        )
        self.api_key = llm_api_key
        self.base_url = llm_base_url
        self.model = llm_model or "gpt-4o-mini"

    def close(self):
        try:
            self.client._system.stop()
        except Exception:  # noqa: BLE001
            pass

    def _bm25_index(self):
        data = self.collection.get(include=["documents"])
        tokenized = [d.lower().split() for d in data["documents"]]
        return BM25Okapi(tokenized), data

    def _retrieve(self, query: str, k: int = 5):
        start = time.perf_counter()
        vector_hits = self.collection.query(
            query_texts=[query], n_results=max(k * 2, 10)
        )
        bm25, data = self._bm25_index()
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [data["documents"][i] for i in ranked if scores[i] > 0]

        fused: dict[str, tuple[str, str]] = {}
        for i, doc in enumerate(vector_hits["documents"][0]):
            fused[doc] = (doc, vector_hits["metadatas"][0][i].get("source", "unknown"))
        for doc in bm25_docs:
            fused.setdefault(doc, (doc, "keyword"))
        top = [{"text": t, "source": s} for t, s in list(fused.values())[:k]]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return top, elapsed_ms

    def _ask_llm(self, query: str, context: str) -> str:
        if not self.api_key:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only using the provided context. If the "
                            "context lacks the answer, say 'Not covered in the "
                            "documents'. Be concise."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
            )
            return resp.choices[0].message.content
        except Exception:  # noqa: BLE001
            return None

    def _extractive_answer(self, query: str, docs: list[dict]) -> str:
        words = [w for w in query.lower().split() if len(w) > 3]
        best, best_score = None, 0
        for d in docs:
            score = sum(d["text"].lower().count(w) for w in words)
            if score > best_score:
                best, best_score = d["text"], score
        if not best:
            return "Not covered in the documents."
        return " ".join(best.split()[:80]) + " (… retrieved from source)"

    def answer(self, query: str, k: int = 5, collection: str = "documents"):
        docs, elapsed_ms = self._retrieve(query, k)
        context = "\n\n".join(d["text"][:1500] for d in docs)
        answer = self._ask_llm(query, context)
        provider = self.model if answer else "extractive-fallback"
        if answer is None:
            answer = self._extractive_answer(query, docs)
        return {
            "answer": answer,
            "sources": [
                {"text": d["text"][:300], "source": d["source"]} for d in docs
            ],
            "retrieval_time_ms": round(elapsed_ms, 2),
            "provider": provider,
        }
