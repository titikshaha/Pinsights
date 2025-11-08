import torch
import clip
from PIL import Image
import numpy as np
import joblib

# Load saved PCA and KMeans models
pca = joblib.load("backend/pca_model.pkl")
kmeans = joblib.load("backend/kmeans_model.pkl")

# Load CLIP model (same as before)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to extract CLIP embedding
def extract_clip_embedding(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()

# Path to your test image
test_image_path = "data/test/test2.jpg"

# Extract embedding and predict cluster
embedding = extract_clip_embedding(test_image_path)
embedding_pca = pca.transform([embedding])
predicted_cluster = kmeans.predict(embedding_pca)[0]

print(f"🧠 Predicted cluster: {predicted_cluster}")
