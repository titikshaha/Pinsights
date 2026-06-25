"""
clusterer.py — HDBSCAN clustering with KMeans fallback.

HDBSCAN is preferred: it handles noise naturally (label -1),
finds natural cluster shapes, and doesn't require specifying k.
KMeans fallback is used when the image set is very small (<= 15 images).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


@dataclass
class ClusterResult:
    """Output from clustering a set of image embeddings."""
    labels: List[int]               # per-image cluster label (-1 = noise)
    n_clusters: int                 # number of valid clusters (excluding noise)
    centroids: Dict[int, np.ndarray]  # cluster_id → centroid vector
    representative_indices: Dict[int, List[int]]  # cluster_id → [top-5 image indices closest to centroid]
    noise_indices: List[int]        # indices labelled as noise by HDBSCAN
    method: str                     # "hdbscan" or "kmeans"
    silhouette: Optional[float] = None


def _reduce(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA reduction before clustering (improves HDBSCAN on high-dim vectors)."""
    if embeddings.shape[0] <= n_components:
        return embeddings
    actual = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=actual, random_state=42)
    return pca.fit_transform(embeddings)


def _get_representatives(
    embeddings: np.ndarray,
    labels: List[int],
    centroid: np.ndarray,
    cluster_id: int,
    top_k: int = 5,
) -> List[int]:
    """Return indices of top_k images closest to the cluster centroid."""
    indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
    if not indices:
        return []
    cluster_embs = embeddings[indices]
    dists = np.linalg.norm(cluster_embs - centroid, axis=1)
    order = np.argsort(dists)
    return [indices[i] for i in order[:top_k]]


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> ClusterResult:
    """Primary clustering method."""
    try:
        import hdbscan as hdb
    except ImportError:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")

    reduced = _reduce(embeddings)
    normalized = normalize(reduced)

    clusterer = hdb.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(normalized).tolist()

    unique_clusters = sorted(set(l for l in labels if l >= 0))
    n_clusters = len(unique_clusters)

    centroids: Dict[int, np.ndarray] = {}
    rep_indices: Dict[int, List[int]] = {}
    for cid in unique_clusters:
        mask = [i for i, l in enumerate(labels) if l == cid]
        centroid = embeddings[mask].mean(axis=0)
        centroids[cid] = centroid
        rep_indices[cid] = _get_representatives(embeddings, labels, centroid, cid)

    noise_indices = [i for i, l in enumerate(labels) if l == -1]

    # Silhouette only when ≥2 clusters and ≥2 non-noise points
    sil = None
    non_noise = [(i, l) for i, l in enumerate(labels) if l >= 0]
    if n_clusters >= 2 and len(non_noise) >= 4:
        try:
            from sklearn.metrics import silhouette_score
            X_nn = embeddings[[i for i, _ in non_noise]]
            y_nn = [l for _, l in non_noise]
            sil = float(silhouette_score(X_nn, y_nn))
        except Exception:
            pass

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        centroids=centroids,
        representative_indices=rep_indices,
        noise_indices=noise_indices,
        method="hdbscan",
        silhouette=sil,
    )


def cluster_kmeans(
    embeddings: np.ndarray,
    k: int = 3,
) -> ClusterResult:
    """Fallback clustering for small image sets."""
    reduced = _reduce(embeddings, n_components=min(20, embeddings.shape[0] - 1))
    k = min(k, embeddings.shape[0])

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(reduced).tolist()

    centroids: Dict[int, np.ndarray] = {}
    rep_indices: Dict[int, List[int]] = {}
    for cid in range(k):
        centroid = embeddings[[i for i, l in enumerate(labels) if l == cid]].mean(axis=0)
        centroids[cid] = centroid
        rep_indices[cid] = _get_representatives(embeddings, labels, centroid, cid)

    sil = None
    if k >= 2:
        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(reduced, labels))
        except Exception:
            pass

    return ClusterResult(
        labels=labels,
        n_clusters=k,
        centroids=centroids,
        representative_indices=rep_indices,
        noise_indices=[],
        method="kmeans",
        silhouette=sil,
    )


def auto_cluster(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
) -> ClusterResult:
    """
    Auto-select clustering strategy:
    - < 15 images → KMeans(k=2)
    - >= 15 images → HDBSCAN; fallback to KMeans if result has < 2 valid clusters
    """
    n = embeddings.shape[0]
    if n < 15:
        k = max(2, min(3, n // 3))
        return cluster_kmeans(embeddings, k=k)

    result = cluster_hdbscan(embeddings, min_cluster_size=min_cluster_size)

    if result.n_clusters < 2:
        # HDBSCAN found noise everywhere — fall back
        k = min(4, max(2, n // 10))
        return cluster_kmeans(embeddings, k=k)

    return result
