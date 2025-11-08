from pathlib import Path
import numpy as np

dims = {}
for f in Path("data/embeddings/per_pin").glob("*.npy"):
    emb = np.load(f)
    dims[emb.shape[0]] = dims.get(emb.shape[0], 0) + 1
print(dims)
