"""
embedder.py — open-clip-torch ViT-B/32 image and text embedding wrapper.

Runs locally, no external API needed. Singleton pattern so the model
is loaded once at startup and reused across all requests.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
import open_clip

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> None:
    """Load CLIP model into memory (call once at app startup)."""
    global _model, _preprocess, _tokenizer, _device
    if _model is not None:
        return  # already loaded

    _device = _get_device()
    print(f"[Embedder] Loading {model_name} ({pretrained}) on {_device} ...")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    _tokenizer = open_clip.get_tokenizer(model_name)
    _model = _model.to(_device)
    _model.eval()
    print("[Embedder] Model ready.")


def _ensure_loaded() -> None:
    if _model is None:
        model_name = os.getenv("CLIP_MODEL", "ViT-B-32")
        pretrained = os.getenv("CLIP_PRETRAINED", "openai")
        load_model(model_name, pretrained)


def embed_images(image_paths: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of image file paths using CLIP vision encoder.

    Returns:
        np.ndarray of shape (N, 768), L2-normalised.
    """
    _ensure_loaded()

    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_indices = []

        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(_preprocess(img))
                valid_indices.append(j)
            except Exception as e:
                print(f"[Embedder] Warning: could not open {path}: {e}")
                # Insert zero vector placeholder
                all_embeddings.append(np.zeros(768, dtype=np.float32))

        if not images:
            continue

        tensor = torch.stack(images).to(_device)
        with torch.no_grad():
            feats = _model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        batch_embeddings = feats.cpu().numpy().astype(np.float32)

        # Re-insert placeholders for failed images in original order
        result_batch = [None] * len(batch_paths)
        for k, vi in enumerate(valid_indices):
            result_batch[vi] = batch_embeddings[k]

        for item in result_batch:
            if item is not None:
                all_embeddings.append(item)

    if not all_embeddings:
        return np.zeros((0, 768), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of text strings using CLIP text encoder.

    Returns:
        np.ndarray of shape (N, 768), L2-normalised.
    """
    _ensure_loaded()

    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = _tokenizer(batch).to(_device)
        with torch.no_grad():
            feats = _model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy().astype(np.float32))

    if not all_embeddings:
        return np.zeros((0, 768), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)


def embed_single_text(text: str) -> np.ndarray:
    """Embed a single text string. Returns 1D array of shape (768,)."""
    return embed_texts([text])[0]
