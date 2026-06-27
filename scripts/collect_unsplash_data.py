import os
import requests
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not UNSPLASH_ACCESS_KEY:
    raise ValueError("UNSPLASH_ACCESS_KEY not found in .env file")

OUTPUT_DIR = Path("data/pinterest_img")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define classes and their specific search queries
# We aim for ~50-100 images per class (60% full body, 20% detail, 20% contextual)
# We also want to include edge cases like neutral colors, summer/winter.
AESTHETIC_CLASSES = {
    "Luxury_Streetwear": [
        ("full_body", "person wearing luxury streetwear fashion outfit full body"),
        ("edge_neutral", "person wearing streetwear beige neutral outfit"),
        ("edge_winter", "person wearing luxury streetwear heavy winter coat"),
        ("detail", "close up person wearing streetwear fashion outfit fabric texture"),
        ("contextual", "person wearing luxury streetwear lifestyle street style")
    ],
    "Coastal_Grandmother": [
        ("full_body", "person wearing coastal grandmother fashion outfit full body"),
        ("edge_neutral", "person wearing coastal grandmother beige neutral outfit"),
        ("edge_summer", "person wearing coastal grandmother summer linen outfit"),
        ("detail", "close up person wearing coastal grandmother outfit fabric texture"),
        ("contextual", "person wearing coastal grandmother lifestyle outfit")
    ],
    "Y2K_Revival": [
        ("full_body", "person wearing Y2K fashion outfit full body"),
        ("edge_summer", "person wearing Y2K summer crop top outfit"),
        ("edge_neutral", "person wearing Y2K neutral color outfit"),
        ("detail", "close up person wearing Y2K fashion outfit accessory"),
        ("contextual", "person wearing Y2K fashion lifestyle street style")
    ],
    "Maximalism": [
        ("full_body", "person wearing maximalist fashion outfit full body"),
        ("edge_winter", "person wearing maximalist fashion heavy winter coat"),
        ("detail", "close up person wearing maximalist fashion outfit fabric texture"),
        ("contextual", "person wearing maximalist fashion lifestyle street style")
    ],
    "Kurti_Aesthetics": [
        ("full_body", "person wearing kurti fashion outfit full body"),
        ("edge_neutral", "person wearing kurti beige neutral outfit"),
        ("detail", "close up person wearing kurti outfit fabric embroidery detail"),
        ("contextual", "person wearing kurti fashion lifestyle outfit")
    ],
    "Saree_Styling": [
        ("full_body", "person wearing saree outfit full body"),
        ("edge_neutral", "person wearing saree beige neutral outfit"),
        ("detail", "close up person wearing saree outfit fabric embroidery detail"),
        ("contextual", "person wearing saree fashion lifestyle")
    ],
    "Traditional_Fusion": [
        ("full_body", "person wearing western ethnic fusion fashion outfit full body"),
        ("edge_fusion", "person wearing fusion tension western traditional outfit"),
        ("detail", "close up person wearing fusion fashion outfit fabric texture"),
        ("contextual", "person wearing fusion fashion lifestyle street style")
    ],
    "Contemporary_South_Asian": [
        ("full_body", "person wearing contemporary south asian fashion outfit full body"),
        ("edge_neutral", "person wearing contemporary south asian neutral outfit"),
        ("detail", "close up person wearing contemporary south asian outfit fabric detail"),
        ("contextual", "person wearing contemporary south asian fashion lifestyle")
    ],
    "Rock_Grunge": [
        ("full_body", "person wearing grunge rock fashion outfit full body"),
        ("edge_summer", "person wearing grunge rock summer outfit"),
        ("detail", "close up person wearing grunge rock fashion outfit texture hardware"),
        ("contextual", "person wearing grunge rock fashion lifestyle street style")
    ]
}

def search_unsplash(query, per_page=15):
    """Search Unsplash for photos matching the query."""
    url = "https://api.unsplash.com/search/photos"
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }
    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "portrait"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error fetching {query}: {response.status_code}")
        return []
    return response.json().get("results", [])

def download_image(url, filepath):
    """Download image from url and save it to filepath."""
    if filepath.exists():
        return
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

def main():
    print("Starting Unsplash data collection...")
    # To manage rate limits (50 requests/hour typically for demo),
    # we have 10 classes * 4-5 queries = ~48 requests.
    # We will fetch 15 images per query to get roughly 60-75 images per class.
    
    total_downloaded = 0
    for aesthetic, queries in AESTHETIC_CLASSES.items():
        print(f"\\nProcessing {aesthetic}...")
        for category, query in queries:
            print(f"  Searching for: {query}")
            results = search_unsplash(query, per_page=15)
            for i, photo in enumerate(results):
                # Using 'regular' url for good quality/size tradeoff
                img_url = photo["urls"]["regular"]
                img_id = photo["id"]
                # Prefix with aesthetic to implicitly label
                # Include category in name to help with the 60/20/20 split verification
                filename = f"{aesthetic}_{category}_{img_id}.jpg"
                filepath = OUTPUT_DIR / filename
                
                try:
                    download_image(img_url, filepath)
                    total_downloaded += 1
                except Exception as e:
                    print(f"    Failed to download {img_id}: {e}")
            
            # Sleep slightly to avoid hammering the API
            time.sleep(0.5)

    print(f"\\nData collection complete. Total images downloaded: {total_downloaded}")

if __name__ == "__main__":
    main()
