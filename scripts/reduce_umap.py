import numpy as np
import umap
from pathlib import Path

# Load embeddings
emb_path = Path("data/embeddings/embeddings.npy")
embeddings = np.load(emb_path)

# Fit UMAP
print("Running UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
emb_2d = reducer.fit_transform(embeddings)

# Save
out_path = Path("data/embeddings/umap_2d.npy")
np.save(out_path, emb_2d)
print("Saved 2D UMAP projection:", emb_2d.shape)
