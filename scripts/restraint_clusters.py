import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import joblib

# === Paths ===
embeddings_dir = Path("data/embeddings/per_pin")
output_dir = Path("data/models")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load all embeddings ===
embeddings = []
pin_ids = []

for file in embeddings_dir.glob("*.npy"):
    try:
        emb = np.load(file)
        if emb.shape[0] == 1024:  # Only keep 2048-dim
            embeddings.append(emb)
            pin_ids.append(file.stem)
    except Exception as e:
        print(f"Error loading {file}: {e}")

X = np.vstack(embeddings)
print(f"Loaded {len(X)} embeddings of dimension {X.shape[1]}")

# === Step 1: Apply PCA (optional but keeps clustering clean) ===
pca = PCA(n_components=256, random_state=42)
X_reduced = pca.fit_transform(X)
print("PCA done. Reduced shape:", X_reduced.shape)

# === Step 2: Train KMeans ===
best_k = 3  # based on your earlier tests
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_reduced)

score = silhouette_score(X_reduced, labels)
print(f"KMeans with k={best_k}, silhouette score={score:.4f}")

# === Step 3: Save models ===
joblib.dump(pca, output_dir / "pca_model.pkl")
joblib.dump(kmeans, output_dir / "kmeans_model.pkl")
print("✅ Models saved to data/models/")

# === Step 4: Save cluster assignments ===
cluster_data = pd.DataFrame({"pin_id": pin_ids, "cluster": labels})
cluster_data.to_csv("data/clusters/final_clusters.csv", index=False)
print("✅ Cluster assignments saved to data/clusters/final_clusters.csv")
