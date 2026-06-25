"""
qdrant_client.py — Qdrant vector store connection and collection helpers.

Two collections:
  - fashion_history:    cultural origin, era, visual codes for ~15 aesthetics
  - aesthetic_execution: what each aesthetic actually requires to land

Both embed using the CLIP text encoder so we can query them with
image centroid embeddings (cross-modal retrieval).
"""

from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)

COLLECTION_FASHION_HISTORY = "fashion_history"
COLLECTION_AESTHETIC_EXECUTION = "aesthetic_execution"
EMBED_DIM = 512  # CLIP ViT-B/32

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Return singleton Qdrant client."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _client = QdrantClient(url=url, api_key=api_key)
        print(f"[Qdrant] Connected to {url}")
    return _client


def ensure_collections() -> None:
    """Create collections if they don't exist."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    for name in [COLLECTION_FASHION_HISTORY, COLLECTION_AESTHETIC_EXECUTION]:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            print(f"[Qdrant] Created collection: {name}")
        else:
            print(f"[Qdrant] Collection exists: {name}")


def upsert_chunks(
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: "np.ndarray",  # shape (N, 512)
) -> None:
    """Upsert embedded chunks into a collection."""
    import numpy as np

    client = get_client()
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=vec.tolist(),
                payload=chunk,
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"[Qdrant] Upserted {len(points)} points into '{collection_name}'")


def search(
    collection_name: str,
    query_vector: "np.ndarray",
    top_k: int = 5,
    aesthetic_filter: Optional[str] = None,
) -> List[ScoredPoint]:
    """
    Search a collection by vector similarity.

    Args:
        query_vector: 1D numpy array of shape (512,)
        top_k: number of results to return
        aesthetic_filter: if provided, filter results to this specific aesthetic

    Returns:
        List of ScoredPoint objects with .payload and .score
    """
    client = get_client()

    query_filter = None
    if aesthetic_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="aesthetic",
                    match=MatchValue(value=aesthetic_filter),
                )
            ]
        )

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return results


def collection_count(collection_name: str) -> int:
    """Return number of points in a collection."""
    client = get_client()
    info = client.get_collection(collection_name)
    return info.points_count or 0


def clear_collection(collection_name: str) -> None:
    """Delete all points from a collection (for re-ingestion)."""
    client = get_client()
    client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"[Qdrant] Cleared and recreated collection: {collection_name}")
