# Outfit pairing detection & rules
import os
import pandas as pd
from pathlib import Path
import argparse
import hashlib
from datetime import datetime

def md5(fname):
    """Generate md5 hash of a file to detect duplicates."""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ingest_folder(images_folder, csv_path):
    images_folder = Path(images_folder)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # load existing CSV if it exists
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        existing_paths = set(df_existing["image_path"].astype(str))
        print(f"Found existing CSV with {len(df_existing)} entries.")
    else:
        df_existing = pd.DataFrame(columns=["pin_id", "image_path", "caption", "board", "timestamp", "source_url", "cluster"])
        existing_paths = set()

    new_rows = []
    for img_file in images_folder.glob("**/*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        rel_path = f"data/images/{img_file.name}"
        if rel_path in existing_paths:
            continue  # skip if already in CSV

        pid = img_file.stem
        timestamp = datetime.now().isoformat()
        new_rows.append({
            "pin_id": pid,
            "image_path": rel_path,
            "caption": "",
            "board": "",
            "timestamp": timestamp,
            "source_url": "",
            "cluster": ""
        })

    if not new_rows:
        print("✅ No new images found.")
        return

    print(f"Adding {len(new_rows)} new images...")
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ Updated CSV saved at {csv_path} (total: {len(df_combined)} entries).")

def main():
    parser = argparse.ArgumentParser(description="Pinsights Data Ingestion")
    parser.add_argument("--source", choices=["folder"], required=True, help="Data source type")
    parser.add_argument("--csv", default="data/metadata/pins.csv", help="Path to CSV file")
    parser.add_argument("--folder", default="data/images", help="Folder with images")
    args = parser.parse_args()

    if args.source == "folder":
        ingest_folder(args.folder, args.csv)

if __name__ == "__main__":
    main()
