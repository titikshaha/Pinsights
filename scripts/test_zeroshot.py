import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv()
from backend.ml.embedder import embed_images, embed_texts
import os
from pathlib import Path

LABELS_FILE = "data/labels.csv"
DATA_DIR = Path("data/pinterest_img")

def main():
    print("Loading labels...")
    df = pd.read_csv(LABELS_FILE)
    df = df[df['label'] != "Other/Unsure"].dropna()
    
    unique_classes = sorted(df['label'].unique().tolist())
    
    # We will format the classes into descriptive prompts for zero-shot
    class_prompts = [f"a photo of {cls.lower()} fashion" for cls in unique_classes]
    
    image_paths = [os.path.abspath(DATA_DIR / row['image_path']) for _, row in df.iterrows()]
    labels = [unique_classes.index(row['label']) for _, row in df.iterrows()]
    
    print(f"Embedding {len(image_paths)} images...")
    image_embeddings = embed_images(image_paths, batch_size=64)
    
    print("Embedding text prompts...")
    text_embeddings = embed_texts(class_prompts)
    
    # Compute cosine similarity (embeddings are already L2 normalized)
    # image_embeddings shape: (N, 512)
    # text_embeddings shape: (num_classes, 512)
    similarity = np.dot(image_embeddings, text_embeddings.T) # shape: (N, num_classes)
    
    preds = np.argmax(similarity, axis=1)
    
    acc = accuracy_score(labels, preds)
    print(f"\n--- Zero-Shot Results ---")
    print(f"Zero-Shot Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
