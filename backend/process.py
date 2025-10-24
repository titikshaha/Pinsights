import os
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import clip
from PIL import Image
import numpy as np


def extract_features(image_paths, model, preprocess, device):
    """Extract visual embeddings using CLIP"""
    features = []
    for path in tqdm(image_paths, desc="Extracting CLIP features"):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu().numpy())
        except Exception as e:
            print(f"Error processing {path}: {e}")
            features.append(np.zeros((1, 512)))
    return np.vstack(features)


def process_data(input_csv, output_csv, n_clusters=8):
    # Load metadata
    df = pd.read_csv(input_csv)

    # Check if image_path column exists
    if "image_path" not in df.columns:
        raise ValueError("Missing 'image_path' column in input CSV")

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Extract visual features
    image_features = extract_features(df["image_path"], model, preprocess, device)

    # Dimensionality reduction
    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(image_features)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(reduced_features)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f" Saved clustered data to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Pinsights Data Processing")
    parser.add_argument("--input", default="data/metadata/pins.csv", help="Path to input metadata CSV")
    parser.add_argument("--output", default="data/processed/pins_clustered.csv", help="Path to save processed CSV")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters to form")
    args = parser.parse_args()

    process_data(args.input, args.output, args.clusters)


if __name__ == "__main__":
    main()
