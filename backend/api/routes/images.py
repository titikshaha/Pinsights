"""
images.py — Image serving and preset listing routes.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.api.models import PresetBoard

router = APIRouter(prefix="/images", tags=["images"])

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PRESET_DIR = DATA_DIR / "pinterest_img"


@router.get("/presets", response_model=List[PresetBoard])
async def list_presets():
    """List available preset boards from data/pinterest_img/."""
    if not PRESET_DIR.exists():
        return []

    boards = []
    display_names = {
        "minimal": "Minimal",
        "rock": "Rock & Edge",
        "streetwear": "Streetwear",
        "summer": "Summer Light",
        "winter": "Winter Layers",
    }

    for board_dir in sorted(PRESET_DIR.iterdir()):
        if not board_dir.is_dir():
            continue
        name = board_dir.name
        paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            paths.extend([str(p) for p in board_dir.glob(ext)])

        if not paths:
            continue

        boards.append(PresetBoard(
            name=name,
            display_name=display_names.get(name, name.title()),
            image_count=len(paths),
            thumbnail_paths=[f"/images/file/{name}/{Path(p).name}" for p in paths[:4]],
        ))

    return boards


@router.get("/file/{board}/{filename}")
async def serve_preset_image(board: str, filename: str):
    """Serve a preset board image."""
    img_path = PRESET_DIR / board / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))


@router.get("/session/{session_id}/{filename}")
async def serve_session_image(session_id: str, filename: str):
    """Serve an uploaded session image."""
    upload_dir = DATA_DIR / "uploads" / session_id
    img_path = upload_dir / filename

    # Also check data/images/ for existing dataset images
    if not img_path.exists():
        img_path = DATA_DIR / "images" / filename

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))


@router.get("/data/{filename}")
async def serve_data_image(filename: str):
    """Serve any image from data/images/."""
    img_path = DATA_DIR / "images" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))
