"""Document chunking, embedding, and vector-store ingestion."""

import os
import time

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


class TFEmbeddingFunction:
    """Zero-download fallback embedder (TF hashing over character n-grams).

    No model download required, so the project always runs offline. Lower
    quality than sentence-transformers, but fully functional and dependency-free.
    """

    def __init__(self, dim: int = 768, ngram: int = 3):
        import numpy as np

        self._np = np
        self.dim = dim
        self.ngram = ngram

    def name(self) -> str:
        return "tf-hashing-ngram"

    @property
    def is_legacy(self) -> bool:
        return False

    def _tokens(self, text: str) -> list[str]:
        words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
        toks = []
        for w in words:
            toks.append(w)
            if len(w) >= self.ngram:
                toks.extend(w[i : i + self.ngram] for i in range(len(w) - self.ngram + 1))
        return toks

    def _embed(self, text: str):
        vec = self._np.zeros(self.dim, dtype="float32")
        for t in self._tokens(text):
            vec[hash(t) % self.dim] += 1.0
        norm = self._np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

    def __call__(self, input):
        if isinstance(input, str):
            return self._embed(input).tolist()
        return [self._embed(t).tolist() for t in input]

    def embed_documents(self, input):
        if isinstance(input, str):
            return [self._embed(input).tolist()]
        return [self._embed(d).tolist() for d in input]

    def embed_query(self, input):
        if isinstance(input, str):
            return self._embed(input).tolist()
        return [self._embed(q).tolist() for q in input]


def get_embedding_function():
    """Prefer sentence-transformers; fall back to a zero-download embedder so
    the project runs offline without the heavy torch dependency."""
    try:
        import sentence_transformers  # noqa: F401

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except ImportError:
        return TFEmbeddingFunction()


class Ingester:
    def __init__(self, store_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=store_path)
        self.embed_fn = get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            "documents", embedding_function=self.embed_fn
        )

    def _read_file(self, path: str) -> str:
        if path.endswith(".pdf"):
            reader = PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def _chunk(self, text: str, size: int = 700, overlap: int = 120) -> list[str]:
        chunks, start = [], 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            window = text[start:end]
            if end < n:
                cut = max(window.rfind("."), window.rfind("\n"))
                if cut > size // 2:
                    end = start + cut + 1
            chunks.append(text[start:end].strip())
            if end >= n:
                break
            start = end - overlap
        return [c for c in chunks if len(c) > 40]

    def ingest_file(self, path: str, source_id: str) -> int:
        text = self._read_file(path)
        chunks = self._chunk(text)
        ids = [f"{source_id}-{i}" for i in range(len(chunks))]
        if not ids:
            return 0
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=[{"source": source_id, "chunk": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def ingest_path(self, path: str) -> int:
        total = 0
        if os.path.isfile(path):
            return self.ingest_file(path, os.path.basename(path))
        for root, _, files in os.walk(path):
            for name in sorted(files):
                if name.endswith((".txt", ".md", ".pdf")):
                    full = os.path.join(root, name)
                    total += self.ingest_file(full, os.path.relpath(full, path))
        return total

    def count(self) -> int:
        return self.collection.count()

    def stats(self) -> dict:
        return {"total_chunks": self.collection.count()}
