"""
palette.py — Dominant color extraction from images using KMeans.

Extracts per-image dominant palette and aggregates cluster-level palettes.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image


def _hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def extract_palette(image_path: str, k: int = 5, sample_size: int = 3000) -> List[str]:
    """
    Extract k dominant colors from an image as hex strings.

    Uses KMeans on downsampled pixel array for speed.
    Returns colors ordered by frequency (most dominant first).
    """
    try:
        from sklearn.cluster import KMeans as _KMeans

        img = Image.open(image_path).convert("RGB")
        # Resize for speed
        img.thumbnail((150, 150), Image.LANCZOS)
        pixels = np.array(img).reshape(-1, 3).astype(np.float32)

        if len(pixels) < k:
            return [_hex(tuple(pixels[0].astype(int)))]

        # Subsample
        if len(pixels) > sample_size:
            idx = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = pixels[idx]

        km = _KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=100)
        km.fit(pixels)

        # Order by cluster size (frequency)
        labels = km.labels_
        centers = km.cluster_centers_
        counts = np.bincount(labels)
        order = np.argsort(-counts)

        return [_hex(tuple(centers[i].astype(int))) for i in order]

    except Exception as e:
        print(f"[Palette] Warning: could not extract palette from {image_path}: {e}")
        return []


def aggregate_cluster_palette(
    image_paths: List[str],
    k_per_image: int = 5,
    k_final: int = 6,
) -> List[str]:
    """
    Compute a representative palette for a cluster by aggregating per-image colors.

    Strategy:
      1. Extract k_per_image dominant colors from each image
      2. Pool all colors together
      3. Re-cluster into k_final representative colors
    """
    from sklearn.cluster import KMeans as _KMeans

    all_colors: List[List[float]] = []
    for path in image_paths:
        palette = extract_palette(path, k=k_per_image)
        for hex_color in palette:
            hex_color = hex_color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            all_colors.append([float(r), float(g), float(b)])

    if not all_colors:
        return []

    colors_array = np.array(all_colors, dtype=np.float32)
    k = min(k_final, len(colors_array))

    if k == 1:
        return [_hex(tuple(colors_array[0].astype(int)))]

    km = _KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=200)
    km.fit(colors_array)
    counts = np.bincount(km.labels_)
    order = np.argsort(-counts)

    return [_hex(tuple(km.cluster_centers_[i].astype(int))) for i in order]


def palette_to_tags(hex_colors: List[str]) -> List[str]:
    """
    Convert a hex palette to descriptive color tags for RAG query augmentation.
    Maps RGB to approximate color family labels.
    """
    tags = []
    for hex_color in hex_colors:
        hex_color = hex_color.lstrip("#")
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        except Exception:
            continue

        # Lightness approximation
        l = 0.299 * r + 0.587 * g + 0.114 * b

        if l < 30:
            tags.append("black")
        elif l > 220:
            tags.append("white")
        elif r > 180 and g < 100 and b < 100:
            tags.append("red")
        elif r > 180 and g > 140 and b < 100:
            tags.append("camel/tan")
        elif r < 100 and g < 100 and b > 150:
            tags.append("navy/blue")
        elif r < 120 and g > 140 and b < 120:
            tags.append("green")
        elif r > 180 and g > 180 and b < 100:
            tags.append("yellow")
        elif r > 150 and g < 100 and b > 150:
            tags.append("purple")
        elif abs(r - g) < 20 and abs(g - b) < 20:
            if l < 100:
                tags.append("dark grey")
            elif l < 180:
                tags.append("grey")
            else:
                tags.append("light grey/cream")
        elif r > 160 and g > 100 and b < 80:
            tags.append("brown/rust")
        else:
            tags.append("muted/neutral")

    # Deduplicate, preserve order
    seen = set()
    result = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result
