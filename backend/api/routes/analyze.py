"""
analyze.py — Analysis routes.

POST /analyze        → accept images or source config, start pipeline, return session_id
GET  /analyze/{id}/stream → SSE stream of pipeline progress + final result
GET  /analyze/{id}   → return stored result (after stream completes)
"""

from __future__ import annotations
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import httpx

from backend.api.models import AnalyzeStartResponse, AnalysisResult
from backend.pipeline.graph import run_pipeline_streaming
from backend.api import session_store

router = APIRouter(prefix="/analyze", tags=["analyze"])

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "data/sessions"))
PRESET_DIR = DATA_DIR / "pinterest_img"
UPLOAD_DIR = DATA_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Active streams ───────────────────────────────────────────────────────────
# Maps session_id → asyncio.Queue for SSE events
_active_streams: dict[str, asyncio.Queue] = {}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_preset_paths(board: str) -> List[str]:
    board_dir = PRESET_DIR / board
    if not board_dir.exists():
        raise HTTPException(status_code=404, detail=f"Preset board '{board}' not found")
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        paths.extend([str(p) for p in board_dir.glob(ext)])
    if not paths:
        raise HTTPException(status_code=404, detail=f"No images found in preset board '{board}'")
    return paths[:50]  # cap at 50 for speed


async def _fetch_unsplash_images(query: str, per_page: int = 20) -> List[str]:
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        raise HTTPException(status_code=500, detail="UNSPLASH_ACCESS_KEY not configured")

    unsplash_dir = DATA_DIR / "unsplash_cache" / query.replace(" ", "_")
    unsplash_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": per_page, "client_id": access_key},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

    paths = []
    async with httpx.AsyncClient(timeout=30) as client:
        for r in results:
            pin_id = r["id"]
            img_path = unsplash_dir / f"{pin_id}.jpg"
            if not img_path.exists():
                img_data = await client.get(r["urls"]["regular"])
                img_path.write_bytes(img_data.content)
            paths.append(str(img_path))

    return paths


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("", response_model=AnalyzeStartResponse)
async def start_analysis(
    background_tasks: BackgroundTasks,
    # Multipart upload fields
    source: str = Form(...),
    board: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    goal: str = Form("styling"),
    per_page: int = Form(20),
    files: List[UploadFile] = File(default=[]),
):
    """
    Start an analysis. Accepts:
      - source=upload   + files (multipart)
      - source=preset   + board (e.g. "minimal")
      - source=unsplash + query
    """
    session_id = str(uuid.uuid4())[:8]
    event_queue: asyncio.Queue = asyncio.Queue()
    _active_streams[session_id] = event_queue

    # Resolve image paths
    image_paths: List[str] = []

    if source == "upload":
        if not files:
            raise HTTPException(status_code=400, detail="No files provided for upload source")
        session_upload_dir = UPLOAD_DIR / session_id
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dest = session_upload_dir / (f.filename or f"image_{uuid.uuid4().hex[:6]}.jpg")
            dest.write_bytes(await f.read())
            image_paths.append(str(dest))

    elif source == "preset":
        if not board:
            raise HTTPException(status_code=400, detail="board is required for preset source")
        image_paths = _get_preset_paths(board)

    elif source == "unsplash":
        if not query:
            raise HTTPException(status_code=400, detail="query is required for unsplash source")
        image_paths = await _fetch_unsplash_images(query, per_page)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    # Run pipeline in background
    background_tasks.add_task(
        _run_pipeline_task, session_id, image_paths, goal, event_queue
    )

    return AnalyzeStartResponse(session_id=session_id)


async def _run_pipeline_task(
    session_id: str,
    image_paths: List[str],
    goal: str,
    event_queue: asyncio.Queue,
):
    """Background task: run pipeline and save result."""
    try:
        result = await run_pipeline_streaming(image_paths, session_id, goal, event_queue)
        if result:
            # Save to session store
            session_store.save_session(session_id, result.to_dict())
    except Exception as e:
        await event_queue.put({"type": "error", "message": str(e)})
        await event_queue.put(None)
    finally:
        # Clean up stream entry after a delay
        await asyncio.sleep(30)
        _active_streams.pop(session_id, None)


@router.get("/{session_id}/stream")
async def stream_analysis(session_id: str):
    """
    SSE endpoint. Streams progress events as JSON lines.
    Connect immediately after POST /analyze.

    Event types:
      { "type": "progress", "stage": "...", "progress": 0-100, "message": "..." }
      { "type": "result",   "data": { ...full analysis result... } }
      { "type": "error",    "message": "..." }
    """
    queue = _active_streams.get(session_id)
    if queue is None:
        # Try returning stored result
        stored = session_store.load_session(session_id)
        if stored:
            async def single_event():
                yield f"data: {json.dumps({'type': 'result', 'data': stored})}\n\n"
            return StreamingResponse(single_event(), media_type="text/event-stream")
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired")

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    yield "data: {\"type\": \"heartbeat\"}\n\n"
                    continue

                if event is None:  # sentinel
                    break

                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}", response_model=AnalysisResult)
async def get_analysis(session_id: str):
    """Return the stored result for a completed analysis."""
    stored = session_store.load_session(session_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return stored
