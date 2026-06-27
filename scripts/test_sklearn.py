import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv()
from backend.ml.embedder import embed_images
import os
from pathlib import Path

LABELS_FILE = "data/labels.csv"
DATA_DIR = Path("data/pinterest_img")

def main():
    print("Loading labels...")
    df = pd.read_csv(LABELS_FILE)
    df = df[df['label'] != "Other/Unsure"].dropna()
    
    unique_classes = sorted(df['label'].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    image_paths = [os.path.abspath(DATA_DIR / row['image_path']) for _, row in df.iterrows()]
    labels = [class_to_idx[row['label']] for _, row in df.iterrows()]
    
    print(f"Embedding {len(image_paths)} images...")
    embeddings = embed_images(image_paths, batch_size=64)
    
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("\n--- Model Results ---")
    
    # Logistic Regression
    clf = LogisticRegression(max_iter=1000, C=0.1) # C=0.1 for regularization
    clf.fit(X_train, y_train)
    print(f"Logistic Regression Acc: {accuracy_score(y_val, clf.predict(X_val)):.4f}")
    
    # SVM
    svm = SVC(kernel='linear', C=0.1)
    svm.fit(X_train, y_train)
    print(f"Linear SVM Acc: {accuracy_score(y_val, svm.predict(X_val)):.4f}")
    
    # KNN
    for k in [3, 5, 7, 11]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        print(f"KNN (k={k}) Acc: {accuracy_score(y_val, knn.predict(X_val)):.4f}")

if __name__ == "__main__":
    main()
