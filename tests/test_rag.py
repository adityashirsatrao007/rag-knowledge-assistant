import os

os.environ["STORE_PATH"] = "/tmp/rag_smoke"

import pytest  # noqa: E402

from app.ingest import Ingester  # noqa: E402
from app.rag import RAGEngine  # noqa: E402


@pytest.fixture()
def ingester(tmp_path):
    return Ingester(str(tmp_path))


@pytest.fixture()
def sample_doc(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text(
        "RagWorks revenue share model is 70% to host and 30% to platform. "
        "Service available in Pune, Bengaluru, Mumbai, Hyderabad, Chennai."
    )
    return str(p)


def test_chunk_terminates():
    from app.ingest import Ingester

    ing = Ingester("/tmp/chunk_test_terminates")
    chunks = ing._chunk("x" * 5000, size=700, overlap=120)
    assert len(chunks) >= 5
    assert "".join(chunks)  # not empty


def test_ingest_and_answer(sample_doc):
    store = str(sample_doc).replace("notes.txt", "store")
    ing = Ingester(store)
    n = ing.ingest_file(sample_doc, "notes.txt")
    assert n >= 1

    rag = RAGEngine(store)
    res = rag.answer("What is the revenue share model?", k=2)
    assert res["answer"]
    assert res["sources"]
    assert res["retrieval_time_ms"] >= 0
