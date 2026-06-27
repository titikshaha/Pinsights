"""
ingest_corpus.py — Embed and upsert the fashion history and aesthetic execution
corpus chunks into Qdrant.

Run once to initialise (or re-run to refresh after editing corpus JSON files):
    python -m backend.rag.ingest_corpus

Or directly:
    python backend/rag/ingest_corpus.py
"""

import json
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml.embedder import embed_texts, load_model
from backend.rag.qdrant_client import (
    ensure_collections,
    upsert_chunks,
    collection_count,
    clear_collection,
    COLLECTION_FASHION_HISTORY,
    COLLECTION_AESTHETIC_EXECUTION,
)

CORPUS_DIR = Path(__file__).parent / "corpus"


def load_corpus(filename: str):
    path = CORPUS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def chunk_to_embed_text(chunk: dict) -> str:
    """
    Create the text string that gets embedded for each chunk.

    We embed a rich description so that image-to-text cross-modal
    retrieval works well: the query is an image centroid vector and
    the stored text should be richly descriptive.
    """
    parts = [
        chunk.get("aesthetic", ""),
        chunk.get("era", ""),
        " ".join(chunk.get("tags", [])),
        chunk.get("text", ""),
        chunk.get("what_it_requires", ""),
        chunk.get("common_miss", ""),
        " ".join(chunk.get("your_tell", [])) if "your_tell" in chunk else "",
    ]
    return " ".join(p for p in parts if p).strip()


def ingest_collection(corpus_file: str, collection_name: str, force: bool = False) -> int:
    count = collection_count(collection_name)
    chunks = load_corpus(corpus_file)

    if count > 0 and not force:
        print(f"[Ingest] '{collection_name}' already has {count} points. Skipping. Use --force to re-ingest.")
        return count

    if force and count > 0:
        print(f"[Ingest] Force mode: clearing '{collection_name}' ({count} existing points)...")
        clear_collection(collection_name)

    print(f"[Ingest] Embedding {len(chunks)} chunks for '{collection_name}'...")
    texts = [chunk_to_embed_text(c) for c in chunks]

    import numpy as np
    embeddings = embed_texts(texts)

    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {embeddings.shape[0]} embeddings for {len(chunks)} chunks"
        )

    upsert_chunks(collection_name, chunks, embeddings)
    return len(chunks)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pinsights RAG corpus ingestion")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if collection is populated")
    args = parser.parse_args()

    print("=== Pinsights RAG Corpus Ingestion ===\n")

    from dotenv import load_dotenv
    load_dotenv()

    # Load CLIP model
    from backend.ml.embedder import _ensure_loaded
    _ensure_loaded()

    # Ensure collections exist
    ensure_collections()

    # Ingest fashion history
    n_history = ingest_collection(
        "fashion_history.json",
        COLLECTION_FASHION_HISTORY,
        force=args.force,
    )
    print(f"DONE fashion_history: {n_history} chunks\n")

    # Ingest aesthetic execution
    n_execution = ingest_collection(
        "aesthetic_execution.json",
        COLLECTION_AESTHETIC_EXECUTION,
        force=args.force,
    )
    print(f"DONE aesthetic_execution: {n_execution} chunks\n")

    print("=== Ingestion complete ===")
    print(f"Total chunks: {n_history + n_execution}")


if __name__ == "__main__":
    main()
