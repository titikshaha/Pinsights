import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# Load embeddings
embeddings = np.load("data/embeddings/embeddings.npy")

# Choose k
k = 3  # experiment later with 3–10
print(f"Running KMeans with k={k}...")

model = KMeans(n_clusters=k, random_state=42, n_init='auto')
cluster_ids = model.fit_predict(embeddings)

# Load metadata
meta = pd.read_csv("data/metadata/pins.csv")

# Add cluster info
meta['cluster_id'] = cluster_ids

# Save cluster file
run_id = datetime.now().strftime("%Y%m%d_%H%M")
out_path = Path(f"data/clusters/{run_id}_clusters.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
meta.to_csv(out_path, index=False)
print(" Saved clusters:", out_path)

# Optional metrics
from sklearn.metrics import silhouette_score
score = silhouette_score(embeddings, cluster_ids)
print(f"Silhouette Score: {score:.4f}")
