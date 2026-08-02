#!/usr/bin/env python3
"""Ingest documents into the vector store.

Usage:
    python scripts/ingest.py --path ./data/sample_docs [--store ./chroma_db]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingest import Ingester  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="File or folder to ingest")
    parser.add_argument("--store", default="./chroma_db")
    args = parser.parse_args()

    ingester = Ingester(args.store)
    n = ingester.ingest_path(args.path)
    print(f"Ingested {n} chunks. Total in store: {ingester.count()}")


if __name__ == "__main__":
    main()
