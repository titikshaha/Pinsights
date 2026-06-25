"""
intake_agent.py — Intake Agent: embed images, cluster, extract palettes.

Input:  list of image file paths
Output: IntakeResult with cluster objects, representative images, palettes,
        and centroid embeddings ready for RAG querying.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

from backend.ml.embedder import embed_images
from backend.ml.clusterer import auto_cluster, ClusterResult
from backend.ml.palette import aggregate_cluster_palette, palette_to_tags


@dataclass
class ClusterObject:
    cluster_id: int
    size: int
    representative_paths: List[str]          # absolute paths to top-5 representative images
    dominant_palette: List[str]              # hex colors, most-dominant first
    palette_tags: List[str]                  # descriptive color tags for RAG
    centroid: np.ndarray                     # shape (512,), for Qdrant query
    image_paths: List[str]                   # all image paths in this cluster


@dataclass
class IntakeResult:
    clusters: List[ClusterObject]
    noise_paths: List[str]                   # images HDBSCAN flagged as noise
    total_images: int
    method: str                              # "hdbscan" or "kmeans"
    silhouette: Optional[float]
    all_embeddings: np.ndarray               # (N, 512) full embedding matrix
    all_paths: List[str]                     # original image path list (aligned with embeddings)


async def run_intake(image_paths: List[str], progress_callback=None) -> IntakeResult:
    """
    Run the full intake pipeline:
      1. Embed all images with CLIP
      2. Cluster with HDBSCAN (or KMeans fallback)
      3. Extract palette per cluster
      4. Return structured IntakeResult

    Args:
        image_paths: list of absolute paths to image files
        progress_callback: optional async callable(stage: str, pct: float)
    """
    if not image_paths:
        raise ValueError("No image paths provided")

    # Filter existing files
    valid_paths = [p for p in image_paths if Path(p).exists()]
    if not valid_paths:
        raise ValueError("None of the provided image paths exist on disk")

    if progress_callback:
        await progress_callback("embedding", 0.0)

    # 1. Embed
    embeddings = embed_images(valid_paths)
    if embeddings.shape[0] == 0:
        raise RuntimeError("Embedding failed — no valid images produced embeddings")

    if progress_callback:
        await progress_callback("clustering", 0.4)

    # 2. Cluster
    cluster_result: ClusterResult = auto_cluster(embeddings)

    if progress_callback:
        await progress_callback("palette", 0.65)

    # 3. Build cluster objects
    cluster_objects: List[ClusterObject] = []

    for cid in sorted(cluster_result.centroids.keys()):
        # Get all paths in this cluster
        member_indices = [i for i, l in enumerate(cluster_result.labels) if l == cid]
        member_paths = [valid_paths[i] for i in member_indices]

        # Representative paths (closest to centroid)
        rep_indices = cluster_result.representative_indices.get(cid, member_indices[:5])
        rep_paths = [valid_paths[i] for i in rep_indices]

        # Palette extraction on representative images (faster, representative of cluster)
        palette_paths = rep_paths if len(rep_paths) >= 3 else member_paths[:10]
        dominant_palette = aggregate_cluster_palette(palette_paths, k_per_image=5, k_final=6)
        palette_tags = palette_to_tags(dominant_palette)

        cluster_objects.append(ClusterObject(
            cluster_id=cid,
            size=len(member_paths),
            representative_paths=rep_paths,
            dominant_palette=dominant_palette,
            palette_tags=palette_tags,
            centroid=cluster_result.centroids[cid],
            image_paths=member_paths,
        ))

    noise_paths = [valid_paths[i] for i in cluster_result.noise_indices]

    if progress_callback:
        await progress_callback("intake_complete", 0.80)

    return IntakeResult(
        clusters=cluster_objects,
        noise_paths=noise_paths,
        total_images=len(valid_paths),
        method=cluster_result.method,
        silhouette=cluster_result.silhouette,
        all_embeddings=embeddings,
        all_paths=valid_paths,
    )
