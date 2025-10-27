"""
Purpose:
 - Read data/metadata/pins.csv (expects column: pin_id,image_path,caption,...)
 - Create image embeddings (CLIP) for each pin_id
 - Optionally create caption embeddings and combine them
 - Save per-pin embeddings to data/embeddings/{pin_id}.npy
 - Save combined embeddings to data/embeddings/embeddings.npy
 - Save manifest mapping to data/embeddings/manifest.csv
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# CONFIG
CSV_PATH = Path("data/metadata/pins.csv")
EMBED_DIR = Path("data/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
PER_PIN_DIR = EMBED_DIR / "per_pin"
PER_PIN_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = EMBED_DIR / "manifest.csv"
COMBINED_PATH = EMBED_DIR / "embeddings.npy"

# How to combine text + image vectors: "concat", "avg", or None
COMBINE_STRATEGY = "concat"   # options: "concat", "avg", None

# Batch sizes
IMAGE_BATCH = 64  # decrease if OOM on GPU/CPU
TEXT_BATCH = 64

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model + processor
model_name = "openai/clip-vit-base-patch32"  # common stable option
print("Loading CLIP model...")
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

# utilities
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Can't open image {path}: {e}")
        return None

# Load CSV
if not CSV_PATH.exists():
    raise SystemExit(f"CSV not found at {CSV_PATH}. Please create data/metadata/pins.csv")

df = pd.read_csv(CSV_PATH)
if "pin_id" not in df.columns or "image_path" not in df.columns:
    raise SystemExit("CSV must contain columns: pin_id,image_path")

# Normalize paths if needed (optional)
df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/")

# Build list of items to embed (filter out missing files)
records = []
for _, row in df.iterrows():
    img_path = Path(row["image_path"])
    if img_path.exists():
        records.append({
            "pin_id": str(row["pin_id"]),
            "image_path": str(img_path),
            "caption": "" if pd.isna(row.get("caption", "")) else str(row.get("caption", ""))
        })
    else:
        print(f"[WARN] Missing file for pin_id {row['pin_id']}: {img_path}")

print(f"Images to embed: {len(records)}")
if len(records) == 0:
    raise SystemExit("No images found to embed. Check CSV image_path values.")

# Prepare storage
pin_ids = []
embeddings_list = []  # will store combined embeddings
manifest_rows = []

# Process in image batches
for i in range(0, len(records), IMAGE_BATCH):
    batch = records[i:i+IMAGE_BATCH]
    # load PIL images
    images = []
    idx_map = []
    for idx, rec in enumerate(batch):
        img = load_image(rec["image_path"])
        if img is None:
            continue
        images.append(img)
        idx_map.append(rec)

    if len(images) == 0:
        continue

    # preprocess and forward
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)  # shape (B, D_img)
    image_features = image_features.cpu().numpy()

    # optional: normalize image embeddings to unit length
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)

    # If captions present and COMBINE_STRATEGY != None, compute text features for this batch (matching order)
    captions = [rec["caption"] for rec in idx_map]
    text_features = None
    if COMBINE_STRATEGY is not None and any([c.strip() for c in captions]):
        # compute text features in a separate loop to avoid long padding across many captions
        inputs_text = processor(text=captions, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            tfeat = model.get_text_features(**inputs_text)
        text_features = tfeat.cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

    # Combine and save per-pin
    for j, rec in enumerate(idx_map):
        pid = rec["pin_id"]
        img_emb = image_features[j]
        final_emb = img_emb
        if text_features is not None:
            txt_emb = text_features[j]
            if COMBINE_STRATEGY == "concat":
                # Check if caption exists; if not, pad with zeros
                if txt_emb is None or np.allclose(txt_emb, 0) or not rec["caption"].strip():
                    txt_emb = np.zeros_like(img_emb)
                # Ensure both are 1D arrays
                if img_emb.ndim > 1:
                    img_emb = img_emb.flatten()
                if txt_emb.ndim > 1:
                    txt_emb = txt_emb.flatten()
                final_emb = np.concatenate([img_emb, txt_emb])

            elif COMBINE_STRATEGY == "avg":
                # if dims mismatch (they won't with CLIP), project or pad — here dims match so average
                final_emb = (img_emb + txt_emb) / 2.0
            else:
                final_emb = img_emb

                # Ensure both embeddings are 1D and of same shape
        if img_emb.ndim > 1:
            img_emb = img_emb.flatten()
        if txt_emb is None or txt_emb.size == 0:
            txt_emb = np.zeros_like(img_emb)
        if txt_emb.ndim > 1:
            txt_emb = txt_emb.flatten()

        # Ensure same dimension before concat
        if img_emb.shape[0] != txt_emb.shape[0]:
            txt_emb = np.zeros_like(img_emb)

        final_emb = np.concatenate([img_emb, txt_emb])


        # save per-pin
        per_pin_path = PER_PIN_DIR / f"{pid}.npy"
        np.save(per_pin_path, final_emb.astype(np.float32))

        # record for combined array
        pin_ids.append(pid)
        embeddings_list.append(final_emb.astype(np.float32))
        manifest_rows.append({"pin_id": pid, "image_path": rec["image_path"], "embed_path": str(per_pin_path).replace("\\", "/")})

    # free memory
    del inputs, image_features
    if text_features is not None:
        del inputs_text, tfeat, text_features

# Create combined matrix
if len(embeddings_list) == 0:
    raise SystemExit("No embeddings generated.")

# Ensure consistent shape
emb_matrix = np.vstack(embeddings_list)
np.save(COMBINED_PATH, emb_matrix)
print(f"Saved combined embeddings: {COMBINED_PATH} (shape: {emb_matrix.shape})")

# Save manifest
manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
print(f"Saved manifest: {MANIFEST_PATH}")

print("Done. Per-pin embeddings in:", PER_PIN_DIR)
print("Combined embeddings path:", COMBINED_PATH)
print("Total embeddings:", len(embeddings_list))
