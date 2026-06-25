"""
models.py — Pydantic schemas for request/response validation.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ─── Request models ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    source: Literal["upload", "unsplash", "preset"]
    query: Optional[str] = None         # for unsplash
    board: Optional[str] = None         # for preset (e.g. "minimal")
    per_page: int = Field(default=20, ge=5, le=50)


class SessionCreate(BaseModel):
    name: Optional[str] = None


# ─── Response models ──────────────────────────────────────────────────────────

class AnalyzeStartResponse(BaseModel):
    session_id: str
    message: str = "Analysis started. Connect to /analyze/{session_id}/stream for live updates."


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    representative_paths: List[str]
    dominant_palette: List[str]
    palette_tags: List[str]
    aesthetic_name: str
    description: str
    visual_signals: List[str]
    aspiration_reading: str
    palette_story: str
    cultural_origin: str
    gaps: List[Dict[str, Any]]


class AestheticDNA(BaseModel):
    primary_world: str
    secondary_world: Optional[str]
    visual_tension: str
    overall_aspiration: str


class GapItem(BaseModel):
    aesthetic: str
    gap_name: str
    what_it_requires: str
    common_miss: str
    your_tell: List[str]
    gap_type: str
    severity: Literal["critical", "moderate", "minor"]
    actionable_step: str


class CulturalContextItem(BaseModel):
    claim: str
    because: str
    source_era: str
    cultural_code: str


class AnalysisResult(BaseModel):
    aesthetic_dna: AestheticDNA
    clusters: List[ClusterSummary]
    gaps: List[GapItem]
    primary_gap: Optional[GapItem]
    cultural_context: List[CulturalContextItem]
    drift_signal: Optional[str]
    meta: Dict[str, Any]


class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    total_images: int
    primary_world: str
    secondary_world: Optional[str]
    thumbnail_paths: List[str]


class DriftResult(BaseModel):
    session_a_id: str
    session_b_id: str
    signal: Literal["consolidating", "shifting", "contradicting", "insufficient_data"]
    description: str
    primary_world_change: Optional[str]


class PresetBoard(BaseModel):
    name: str
    display_name: str
    image_count: int
    thumbnail_paths: List[str]
