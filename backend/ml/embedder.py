import os
from typing import List
import numpy as np
import torch
from PIL import Image

_device = "cpu"
if torch.cuda.is_available():
    _device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _device = "mps"

_model = None
_processor = None
_open_clip_model = None
_open_clip_preprocess = None
_tokenizer = None
_use_huggingface = False

def load_model(model_name: str, pretrained: str) -> None:
    global _model, _processor, _open_clip_model, _open_clip_preprocess, _tokenizer, _use_huggingface
    
    _use_huggingface = (pretrained.lower() == "huggingface")
    
    print(f"[Embedder] Loading {model_name} ({pretrained}) on {_device} ...")
    if _use_huggingface:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(model_name).to(_device)
        _processor = CLIPProcessor.from_pretrained(model_name)
    else:
        import open_clip
        _open_clip_model, _, _open_clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _open_clip_model = _open_clip_model.to(_device)
        _open_clip_model.eval()
        _tokenizer = open_clip.get_tokenizer(model_name)
        
    print("[Embedder] Model ready.")

def _ensure_loaded() -> None:
    if _model is None and _open_clip_model is None:
        model_name = os.getenv("CLIP_MODEL", "patrickjohncyh/fashion-clip")
        pretrained = os.getenv("CLIP_PRETRAINED", "huggingface")
        load_model(model_name, pretrained)

def embed_images(image_paths: List[str], batch_size: int = 32) -> np.ndarray:
    _ensure_loaded()
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_indices = []

        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                if not _use_huggingface:
                    img = _open_clip_preprocess(img)
                images.append(img)
                valid_indices.append(j)
            except Exception as e:
                print(f"[Embedder] Warning: could not open {path}: {e}")
                all_embeddings.append(np.zeros(512, dtype=np.float32))

        if not images:
            continue

        with torch.no_grad():
            if _use_huggingface:
                inputs = _processor(images=images, return_tensors="pt", padding=True).to(_device)
                feats = _model.get_image_features(**inputs)
                if hasattr(feats, 'pooler_output'):
                    feats = feats.pooler_output
                elif hasattr(feats, 'image_embeds'):
                    feats = feats.image_embeds
                elif not isinstance(feats, torch.Tensor) and hasattr(feats, '__getitem__'):
                    feats = feats[0]
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            else:
                tensor = torch.stack(images).to(_device)
                feats = _open_clip_model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)

        batch_embeddings = feats.cpu().numpy().astype(np.float32)

        result_batch = [None] * len(batch_paths)
        for k, vi in enumerate(valid_indices):
            result_batch[vi] = batch_embeddings[k]

        for item in result_batch:
            if item is not None:
                all_embeddings.append(item)

    if not all_embeddings:
        return np.zeros((0, 512), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    _ensure_loaded()
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        with torch.no_grad():
            if _use_huggingface:
                inputs = _processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(_device)
                feats = _model.get_text_features(**inputs)
                if hasattr(feats, 'pooler_output'):
                    feats = feats.pooler_output
                elif hasattr(feats, 'text_embeds'):
                    feats = feats.text_embeds
                elif not isinstance(feats, torch.Tensor) and hasattr(feats, '__getitem__'):
                    feats = feats[0]
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            else:
                text_tokens = _tokenizer(batch_texts).to(_device)
                feats = _open_clip_model.encode_text(text_tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embeddings.append(feats.cpu().numpy().astype(np.float32))

    if not all_embeddings:
        return np.zeros((0, 512), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)

def embed_single_image(image_path: str) -> np.ndarray:
    return embed_images([image_path])[0]

def embed_single_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
