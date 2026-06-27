import os
import glob
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path

app = FastAPI()

# Mount the data directory to serve images statically
app.mount("/images", StaticFiles(directory="data/pinterest_img"), name="images")

LABELS_FILE = "data/labels.csv"
DATA_DIR = Path("data/pinterest_img")

CLASSES = [
    "90s Minimalism",
    "Luxury Streetwear",
    "Coastal Grandmother",
    "Y2K Revival",
    "Maximalism",
    "Kurti Aesthetics",
    "Saree Styling",
    "Traditional Fusion",
    "Contemporary South Asian",
    "Rock/Grunge",
    "Other/Unsure"
]

class LabelData(BaseModel):
    image_path: str
    label: str

def get_all_images():
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(DATA_DIR.rglob(ext))
    # Return relative paths to data/pinterest_img
    return [str(p.relative_to(DATA_DIR)).replace("\\", "/") for p in images]

def get_labeled_images():
    if not os.path.exists(LABELS_FILE):
        return set()
    df = pd.read_csv(LABELS_FILE)
    return set(df["image_path"].tolist())

@app.get("/", response_class=HTMLResponse)
def index():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pinsights Labeling UI</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0A0A0B; color: #F5F0E8; display: flex; flex-direction: column; align-items: center; padding: 20px; }}
            .container {{ display: flex; flex-direction: row; gap: 40px; max-width: 1200px; width: 100%; }}
            .image-container {{ flex: 1; display: flex; justify-content: center; align-items: center; background: #1a1a1c; border-radius: 8px; padding: 20px; height: 70vh; }}
            img {{ max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; }}
            .controls {{ flex: 1; display: flex; flex-direction: column; gap: 15px; }}
            .btn {{ background: #2a2a2c; border: 1px solid #4a4a4c; color: #F5F0E8; padding: 15px 20px; border-radius: 6px; cursor: pointer; font-size: 16px; text-align: left; transition: all 0.2s; }}
            .btn:hover {{ background: #C8A882; color: #0A0A0B; border-color: #C8A882; }}
            .header {{ width: 100%; max-width: 1200px; display: flex; justify-content: space-between; margin-bottom: 20px; border-bottom: 1px solid #2a2a2c; padding-bottom: 20px; }}
            .progress {{ font-size: 18px; color: #C8A882; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Pinsights v2 - Data Labeling</h2>
            <div class="progress" id="progress">Loading...</div>
        </div>
        <div class="container">
            <div class="image-container">
                <img id="current-image" src="" alt="Loading..." />
            </div>
            <div class="controls" id="buttons">
                <!-- Buttons injected by JS -->
            </div>
        </div>

        <script>
            let currentImagePath = "";
            
            const classes = {CLASSES};

            async function loadNext() {{
                const res = await fetch('/next');
                const data = await res.json();
                
                document.getElementById('progress').innerText = `Labeled: ${{data.labeled_count}} / ${{data.total_count}}`;
                
                if (data.image_path) {{
                    currentImagePath = data.image_path;
                    document.getElementById('current-image').src = '/images/' + data.image_path;
                }} else {{
                    document.getElementById('current-image').src = '';
                    document.getElementById('current-image').alt = 'All images labeled!';
                    document.getElementById('buttons').innerHTML = '';
                }}
            }}

            async function submitLabel(label) {{
                if (!currentImagePath) return;
                
                await fetch('/label', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ image_path: currentImagePath, label: label }})
                }});
                
                loadNext();
            }}

            function initButtons() {{
                const container = document.getElementById('buttons');
                classes.forEach(c => {{
                    const btn = document.createElement('button');
                    btn.className = 'btn';
                    btn.innerText = c;
                    btn.onclick = () => submitLabel(c);
                    container.appendChild(btn);
                }});
            }}

            initButtons();
            loadNext();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/next")
def get_next():
    all_images = get_all_images()
    labeled = get_labeled_images()
    
    unlabeled = [img for img in all_images if img not in labeled]
    
    return {
        "image_path": unlabeled[0] if unlabeled else None,
        "labeled_count": len(labeled),
        "total_count": len(all_images)
    }

@app.post("/label")
def save_label(data: LabelData):
    file_exists = os.path.exists(LABELS_FILE)
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("image_path,label\n")
        f.write(f"{data.image_path},{data.label}\n")
    
    return {"status": "success"}

if __name__ == "__main__":
    print(f"Starting Labeling UI on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
