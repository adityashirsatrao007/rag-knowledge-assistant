#!/usr/bin/env python3
"""Evaluate retrieval hit-rate and answer quality.

Usage:
    python scripts/evaluate.py --store ./chroma_db
Reads data/eval_questions.json: [{"question": ..., "expected_source": ...}]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag import RAGEngine  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="./chroma_db")
    parser.add_argument("--eval", default="data/eval_questions.json")
    args = parser.parse_args()

    rag = RAGEngine(args.store)
    with open(args.eval, encoding="utf-8") as fh:
        cases = json.load(fh)

    hit = 0
    for case in cases:
        docs, _ = rag._retrieve(case["question"], k=5)
        sources = [d["source"] for d in docs]
        if any(case["expected_source"] in s for s in sources):
            hit += 1
    pct = 100.0 * hit / max(len(cases), 1)
    print(f"Retrieval hit-rate: {hit}/{len(cases)} = {pct:.1f}%")
    rag.close()


if __name__ == "__main__":
    main()
