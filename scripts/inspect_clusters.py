# scripts/inspect_clusters.py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

# CONFIG
EMBED_PATH = Path("data/embeddings/embeddings.npy")      # combined embeddings array
PINS_CSV = Path("data/metadata/pins.csv")               # must contain pin_id,image_path
CLUSTERS_CSV = Path("data/clusters/20251108_2128_clusters.csv")  # your cluster assignments
OUT_DIR = Path("data/inspections")
NUM_PER_CLUSTER = 9   # how many representative images to display per cluster
IMAGE_SIZE = (224, 224)  # resize for grid

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
if not EMBED_PATH.exists():
    raise SystemExit(f"Embeddings not found at {EMBED_PATH}")
if not PINS_CSV.exists():
    raise SystemExit(f"Pins CSV not found at {PINS_CSV}")
if not CLUSTERS_CSV.exists():
    raise SystemExit(f"Clusters CSV not found at {CLUSTERS_CSV}")

emb = np.load(EMBED_PATH)          # shape (N, D)
df_pins = pd.read_csv(PINS_CSV, dtype=str)
df_clusters = pd.read_csv(CLUSTERS_CSV, dtype=str)  # expects pin_id and cluster columns

# Normalize types
if "pin_id" not in df_pins.columns or "image_path" not in df_pins.columns:
    raise SystemExit("pins.csv must contain columns: pin_id,image_path")
if "pin_id" not in df_clusters.columns or "cluster" not in df_clusters.columns:
    # try alternative column name
    if "cluster_id" in df_clusters.columns:
        df_clusters = df_clusters.rename(columns={"cluster_id":"cluster"})
    else:
        raise SystemExit("clusters CSV must contain columns: pin_id,cluster (or cluster_id)")

# Build mapping pin_id -> index in embeddings
# We assume embeddings.npy is in the same order as df_pins rows (most import). If not, try to match by per_pin files.
# Sanity check:
n_pins = len(df_pins)
if emb.shape[0] != n_pins:
    print(f"[WARN] embeddings rows ({emb.shape[0]}) != pins.csv rows ({n_pins}). Attempting to align by pin_id using per_pin files.")

# Attempt to build index map:
pin_to_index = {}
# prefer order in pins.csv: index -> pin_id mapping, then map to embedding index if equal length; else fallback
if emb.shape[0] == n_pins:
    for idx, pid in enumerate(df_pins["pin_id"].astype(str)):
        pin_to_index[str(pid)] = idx
else:
    # try to load per_pin folder to build order
    per_pin_dir = Path("data/embeddings/per_pin")
    if per_pin_dir.exists():
        per_files = sorted(list(per_pin_dir.glob("*.npy")))
        for idx, f in enumerate(per_files):
            pid = f.stem
            pin_to_index[str(pid)] = idx
    else:
        raise SystemExit("Can't align embeddings to pins. Make sure embeddings.npy rows correspond to pins.csv rows, or that data/embeddings/per_pin exists.")

# Merge cluster info with pins
df_clusters["pin_id"] = df_clusters["pin_id"].astype(str)
df = df_pins.merge(df_clusters[["pin_id","cluster"]], on="pin_id", how="left")
if df["cluster"].isna().any():
    print("[WARN] Some pins have no cluster assigned. They will be skipped.")

# Convert cluster to int if possible
try:
    df["cluster"] = df["cluster"].astype(int)
except:
    df["cluster"] = df["cluster"].astype(str)

clusters = sorted(df["cluster"].dropna().unique(), key=lambda x: int(x) if str(x).isdigit() else x)
print("Found clusters:", clusters)

def load_image_safe(path, size=IMAGE_SIZE):
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.LANCZOS)
        # ensure consistent size by padding if needed
        new_im = Image.new("RGB", size, (255,255,255))
        new_im.paste(im, ((size[0]-im.size[0])//2, (size[1]-im.size[1])//2))
        return new_im
    except Exception as e:
        # return a blank image with error text
        im = Image.new("RGB", size, (240,240,240))
        return im

results = {}

for c in clusters:
    c = int(c) if str(c).isdigit() else c
    rows = df[df["cluster"]==c].reset_index(drop=True)
    if len(rows) == 0:
        continue
    # get indices for this cluster, filtering out missing mapping
    indices = []
    pin_ids = []
    for pid in rows["pin_id"].astype(str):
        idx = pin_to_index.get(pid, None)
        if idx is not None and 0 <= idx < emb.shape[0]:
            indices.append(idx)
            pin_ids.append(pid)
    if len(indices) == 0:
        print(f"[WARN] No embeddings found for cluster {c}.")
        continue

    Xc = emb[indices]            # cluster embeddings
    centroid = Xc.mean(axis=0, keepdims=True)
    # compute distances
    dists = np.linalg.norm(Xc - centroid, axis=1)
    order = np.argsort(dists)   # nearest to centroid first

    # pick top NUM_PER_CLUSTER
    top_idx = order[:NUM_PER_CLUSTER]
    rep_pins = [pin_ids[i] for i in top_idx]
    results[c] = rep_pins

    # create grid image
    per_row = int(math.sqrt(NUM_PER_CLUSTER))
    fig_w = per_row * IMAGE_SIZE[0]
    fig_h = math.ceil(NUM_PER_CLUSTER / per_row) * IMAGE_SIZE[1]
    grid = Image.new("RGB", (fig_w, fig_h), (255,255,255))
    for i, pid in enumerate(rep_pins):
        img_path = df.loc[df["pin_id"]==pid, "image_path"].values[0]
        img = load_image_safe(img_path)
        x = (i % per_row) * IMAGE_SIZE[0]
        y = (i // per_row) * IMAGE_SIZE[1]
        grid.paste(img, (x,y))
    out_file = OUT_DIR / f"cluster_{c}_repr.jpg"
    grid.save(out_file)
    print(f"Cluster {c}: saved representative grid → {out_file}")
    print(" Representative pin_ids:", rep_pins)
    print(" Example image paths:")
    for pid in rep_pins[:5]:
        print("  ", df.loc[df["pin_id"]==pid, "image_path"].values[0])

print("\nAll done. Representative grids saved to:", OUT_DIR)
