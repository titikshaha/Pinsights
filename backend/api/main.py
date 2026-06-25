"""
main.py — FastAPI application entry point.

Lifespan:
  - Loads CLIP model on startup (once, in memory)
  - Ensures Qdrant collections exist
  - Warns if corpus is empty

Routes:
  /analyze   — analysis pipeline
  /images    — image serving + preset listing
  /history   — session history + drift
  /health    — health check
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.ml.embedder import load_model
from backend.rag.qdrant_client import ensure_collections, collection_count, COLLECTION_FASHION_HISTORY, COLLECTION_AESTHETIC_EXECUTION
from backend.api.routes import analyze, images, history

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n=== Pinsights API Starting ===")

    # Load CLIP model
    model_name = os.getenv("CLIP_MODEL", "ViT-B-32")
    pretrained = os.getenv("CLIP_PRETRAINED", "openai")
    load_model(model_name, pretrained)

    # Ensure Qdrant collections
    try:
        ensure_collections()
        n_history = collection_count(COLLECTION_FASHION_HISTORY)
        n_execution = collection_count(COLLECTION_AESTHETIC_EXECUTION)
        print(f"[Qdrant] fashion_history: {n_history} chunks")
        print(f"[Qdrant] aesthetic_execution: {n_execution} chunks")
        if n_history == 0 or n_execution == 0:
            print("\n⚠️  WARNING: Qdrant collections are empty!")
            print("   Run: python -m backend.rag.ingest_corpus")
            print("   to embed and upsert the corpus.\n")
    except Exception as e:
        print(f"[Qdrant] ⚠️  Connection failed: {e}")
        print("   Make sure Qdrant is running: docker compose up -d\n")

    print("=== API Ready ===\n")
    yield
    print("\n=== Pinsights API Shutting Down ===")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pinsights API",
    description="Personal Aesthetic Intelligence System — multi-agent RAG pipeline for visual identity analysis",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────

app.include_router(analyze.router)
app.include_router(images.router)
app.include_router(history.router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/", tags=["meta"])
async def root():
    return {
        "name": "Pinsights API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /analyze",
            "stream": "GET /analyze/{session_id}/stream",
            "presets": "GET /images/presets",
            "history": "GET /history/sessions",
        },
    }
