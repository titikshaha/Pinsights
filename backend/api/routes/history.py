"""
history.py — Session history and drift tracking routes.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from backend.api.models import SessionSummary, DriftResult
from backend.api import session_store

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    """List all past analysis sessions, newest first."""
    sessions = session_store.list_sessions()
    summaries = []
    for s in sessions:
        meta = s.get("meta", {})
        dna = s.get("aesthetic_dna", {})
        clusters = s.get("clusters", [])
        thumbnails = []
        for c in clusters[:2]:
            thumbnails.extend(c.get("representative_paths", [])[:2])

        summaries.append(SessionSummary(
            session_id=meta.get("session_id", ""),
            created_at=s.get("created_at", ""),
            total_images=meta.get("total_images", 0),
            primary_world=dna.get("primary_world", "Unknown"),
            secondary_world=dna.get("secondary_world"),
            thumbnail_paths=thumbnails[:4],
        ))
    return summaries


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Return the full result for a past session."""
    data = session_store.load_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return data


@router.get("/drift/{session_a}/{session_b}", response_model=DriftResult)
async def compute_drift(session_a: str, session_b: str):
    """
    Compare two sessions and return a drift signal.

    Signals:
      consolidating — primary aesthetic is the same or closely related
      shifting       — primary aesthetic has changed to a distinct category
      contradicting  — aesthetics from sessions actively oppose each other
      insufficient_data — not enough sessions or images to determine
    """
    data_a = session_store.load_session(session_a)
    data_b = session_store.load_session(session_b)

    if not data_a:
        raise HTTPException(status_code=404, detail=f"Session '{session_a}' not found")
    if not data_b:
        raise HTTPException(status_code=404, detail=f"Session '{session_b}' not found")

    dna_a = data_a.get("aesthetic_dna", {})
    dna_b = data_b.get("aesthetic_dna", {})

    world_a = dna_a.get("primary_world", "").lower()
    world_b = dna_b.get("primary_world", "").lower()

    # Simple heuristic drift detection
    # In production, this would use embedding similarity
    if world_a == world_b:
        signal = "consolidating"
        description = f"Your aesthetic has consolidated around {dna_b.get('primary_world')} across both sessions. The visual vocabulary is becoming more coherent."
    elif any(word in world_b for word in world_a.split()) or any(word in world_a for word in world_b.split()):
        signal = "shifting"
        description = f"Your aesthetic has evolved from {dna_a.get('primary_world')} toward {dna_b.get('primary_world')}. There's directional movement — the trajectory is visible."
    else:
        # Check if they're oppositional (e.g. maximalism vs minimalism)
        opposites = [
            ({"minimalism", "minimal", "quiet"}, {"maximalism", "camp", "maximalist"}),
            ({"streetwear", "urban"}, {"romantic", "cottagecore", "coastal"}),
        ]
        is_opposing = False
        for set_a, set_b in opposites:
            if any(w in world_a for w in set_a) and any(w in world_b for w in set_b):
                is_opposing = True
            if any(w in world_b for w in set_a) and any(w in world_a for w in set_b):
                is_opposing = True

        if is_opposing:
            signal = "contradicting"
            description = f"Your collections are pulling in opposing directions — {dna_a.get('primary_world')} and {dna_b.get('primary_world')} represent different aesthetic philosophies. This tension is worth examining: it may indicate a transition, or it may indicate that neither aesthetic is fully owned."
        else:
            signal = "shifting"
            description = f"Your aesthetic has moved from {dna_a.get('primary_world')} to {dna_b.get('primary_world')}. These are distinct worlds — the shift suggests active aesthetic exploration."

    return DriftResult(
        session_a_id=session_a,
        session_b_id=session_b,
        signal=signal,
        description=description,
        primary_world_change=f"{dna_a.get('primary_world')} → {dna_b.get('primary_world')}" if world_a != world_b else None,
    )
