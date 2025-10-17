 # Pinterest API / manual export ingestion
from dotenv import load_dotenv
import argparse
import os
import pandas as pd
from datetime import datetime
import requests


load_dotenv()

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
META_DIR = os.path.join(DATA_DIR, "metadata")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def normalize_timestamp(ts):
    """Convert timestamps to ISO 8601 format if possible."""
    try:
        return datetime.fromisoformat(ts).isoformat()
    except Exception:
        try:
            return pd.to_datetime(ts).isoformat()
        except Exception:
            return None

def load_from_folder(csv_path):
    """Load existing pin metadata (manual mode)."""
    print(f" Loading manual data from {csv_path} ...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", engine="python")
    # Normalize timestamps
    df["timestamp"] = df["timestamp"].apply(normalize_timestamp)
    print(f" Loaded {len(df)} records.")
    return df

def load_from_unsplash(query="fashion", per_page=20, pages=1):
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        raise ValueError("Missing UNSPLASH_ACCESS_KEY in .env")

    print(f"Fetching Unsplash images for query='{query}' ...")
    all_data = []
    for page in range(1, pages + 1):
        url = f"https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "client_id": UNSPLASH_ACCESS_KEY,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])

        for r in results:
            pin_id = r["id"]
            image_url = r["urls"]["regular"]
            caption = r["alt_description"] or ""
            timestamp = r["created_at"]
            source_url = r["links"]["html"]

            image_path = os.path.join(IMAGES_DIR, f"{pin_id}.jpg")

            # Download image
            img_data = requests.get(image_url).content
            with open(image_path, "wb") as f:
                f.write(img_data)

            all_data.append({
                "pin_id": pin_id,
                "image_path": image_path,
                "caption": caption,
                "board": query,
                "timestamp": timestamp,
                "source_url": source_url,
            })

    df = pd.DataFrame(all_data)
    print(f" Fetched {len(df)} images from Unsplash.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Pinsights Data Ingestion")
    parser.add_argument("--source", choices=["folder", "unsplash"], required=True, help="Data source type")
    parser.add_argument("--csv", default="/data/metadata/pins.csv", help="Path to CSV for manual source")
    parser.add_argument("--query", help="Search query for Unsplash (e.g. 'street fashion')")
    args = parser.parse_args()

    os.makedirs("/data/metadata", exist_ok=True)

   
    if args.source == "folder":
        df = load_from_folder(args.csv)
    elif args.source == "unsplash":
        if not args.query:
            raise ValueError("--query is required when source=unsplash")
        df = load_from_unsplash(args.query)
    else:
        raise ValueError("Unknown source type")
    # Save canonical dataset
    output_path =os.path.join(META_DIR, "pins.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved canonical dataset to {output_path}")

if __name__ == "__main__":
    main()
