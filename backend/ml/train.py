import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Import our existing embedder
from backend.ml.embedder import embed_images

# Configuration
LABELS_FILE = "data/labels.csv"
DATA_DIR = Path("data/pinterest_img")
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "aesthetic_classifier.pt"
MAPPING_PATH = MODEL_DIR / "class_mapping.json"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

class AestheticClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_data():
    print("[1/4] Loading labels...")
    df = pd.read_csv(LABELS_FILE)
    
    # Filter out "Other/Unsure" or missing
    df = df[df['label'] != "Other/Unsure"]
    df = df.dropna()
    
    unique_classes = sorted(df['label'].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(unique_classes)}
    
    print(f"Found {len(unique_classes)} classes: {unique_classes}")
    
    # Save mapping
    with open(MAPPING_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)
        
    # Get absolute paths for embedder
    image_paths = [os.path.abspath(DATA_DIR / row['image_path']) for _, row in df.iterrows()]
    labels = [class_to_idx[row['label']] for _, row in df.iterrows()]
    
    print(f"[2/4] Embedding {len(image_paths)} images with CLIP...")
    embeddings = embed_images(image_paths, batch_size=64)
    
    return embeddings, labels, unique_classes

def train():
    device = get_device()
    print(f"Using device: {device}")
    
    embeddings, labels, unique_classes = load_data()
    num_classes = len(unique_classes)
    
    print("[3/4] Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AestheticClassifier(input_dim=512, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    print("[4/4] Training model...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_acc = accuracy_score(all_targets, all_preds)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            
    print(f"\nTraining complete! Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
