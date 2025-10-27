import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

umap_2d = np.load("data/embeddings/umap_2d.npy")
meta = pd.read_csv(sorted(Path("data/clusters").glob("*_clusters.csv"))[-1])

plt.figure(figsize=(8,6))
scatter = plt.scatter(umap_2d[:,0], umap_2d[:,1], c=meta['cluster_id'], cmap='tab10', s=15)
plt.colorbar(scatter, label='Cluster ID')
plt.title("Image Clusters (UMAP + KMeans)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()