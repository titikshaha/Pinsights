"""
session_store.py — Simple JSON-file session persistence for drift tracking.

Stores one JSON file per session in data/sessions/{session_id}.json.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "data/sessions"))
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session_id: str, result: Dict[str, Any]) -> None:
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    path = SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> List[Dict[str, Any]]:
    """Return all sessions sorted by creation time, newest first."""
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                sessions.append(json.load(f))
        except Exception:
            continue
    return sessions


def delete_session(session_id: str) -> bool:
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
