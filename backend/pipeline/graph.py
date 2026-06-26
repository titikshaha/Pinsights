"""
graph.py — LangGraph pipeline orchestration.

Wires the four agents as a StateGraph:
  START → intake → identity → gap → narrator → END

State is a TypedDict passed between nodes. Each node is an async function
that takes the current state and returns a partial update.

SSE streaming is achieved by passing a progress_callback into each node
that emits JSON lines to the connected SSE response.
"""

from __future__ import annotations
import asyncio
import json
from typing import TypedDict, Optional, List, Any, Callable, Awaitable

from langgraph.graph import StateGraph, END

from backend.agents.intake_agent import run_intake, IntakeResult
from backend.agents.identity_agent import run_identity, IdentityResult
from backend.agents.gap_agent import run_gap, GapResult
from backend.agents.narrator_agent import run_narrator, NarratorResult


# ─── State ───────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    image_paths: List[str]
    session_id: str
    goal: str
    intake_result: Optional[IntakeResult]
    identity_result: Optional[IdentityResult]
    gap_result: Optional[GapResult]
    narrator_result: Optional[NarratorResult]
    error: Optional[str]
    progress_events: List[dict]  # accumulated SSE events


# ─── Progress event builder ───────────────────────────────────────────────────

def make_progress_event(stage: str, pct: float, message: str = "") -> dict:
    return {
        "type": "progress",
        "stage": stage,
        "progress": round(pct * 100),
        "message": message or _stage_messages.get(stage, stage),
    }


_stage_messages = {
    "embedding": "Embedding your images with CLIP...",
    "clustering": "Clustering your visual worlds...",
    "palette": "Extracting colour palettes...",
    "intake_complete": "Visual analysis complete",
    "identity": "Naming your aesthetic worlds...",
    "identity_synthesis": "Mapping tensions between worlds...",
    "gap_analysis": "Identifying execution gaps...",
    "narrating": "Writing your aesthetic profile...",
    "complete": "Analysis complete",
}


# ─── Graph nodes ──────────────────────────────────────────────────────────────

async def intake_node(state: PipelineState) -> dict:
    events = list(state.get("progress_events", []))

    async def callback(stage: str, pct: float):
        events.append(make_progress_event(stage, pct))

    try:
        result = await run_intake(state["image_paths"], progress_callback=callback)
        return {"intake_result": result, "progress_events": events}
    except Exception as e:
        return {"error": f"Intake failed: {e}", "progress_events": events}


async def identity_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    events = list(state.get("progress_events", []))

    async def callback(stage: str, pct: float):
        events.append(make_progress_event(stage, pct))

    try:
        result = await run_identity(state["intake_result"], progress_callback=callback)
        return {"identity_result": result, "progress_events": events}
    except Exception as e:
        return {"error": f"Identity failed: {e}", "progress_events": events}


async def gap_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    events = list(state.get("progress_events", []))

    async def callback(stage: str, pct: float):
        events.append(make_progress_event(stage, pct))

    try:
        result = await run_gap(
            state["intake_result"],
            state["identity_result"],
            progress_callback=callback,
        )
        return {"gap_result": result, "progress_events": events}
    except Exception as e:
        return {"error": f"Gap analysis failed: {e}", "progress_events": events}


async def narrator_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    events = list(state.get("progress_events", []))

    async def callback(stage: str, pct: float):
        events.append(make_progress_event(stage, pct))

    try:
        result = await run_narrator(
            state["intake_result"],
            state["identity_result"],
            state["gap_result"],
            session_id=state.get("session_id", ""),
            goal=state.get("goal", "styling"),
            progress_callback=callback,
        )
        return {"narrator_result": result, "progress_events": events}
    except Exception as e:
        return {"error": f"Narration failed: {e}", "progress_events": events}


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("intake", intake_node)
    g.add_node("identity", identity_node)
    g.add_node("gap", gap_node)
    g.add_node("narrator", narrator_node)

    g.set_entry_point("intake")
    g.add_edge("intake", "identity")
    g.add_edge("identity", "gap")
    g.add_edge("gap", "narrator")
    g.add_edge("narrator", END)

    return g.compile()


# ─── Streaming runner ─────────────────────────────────────────────────────────

async def run_pipeline_streaming(
    image_paths: List[str],
    session_id: str,
    goal: str,
    event_queue: asyncio.Queue,
) -> Optional[NarratorResult]:
    """
    Run the full pipeline and push SSE events to event_queue as they occur.

    The SSE route will read from event_queue and stream to the client.
    Final result is also pushed as a "result" event.
    """
    graph = build_graph()

    initial_state: PipelineState = {
        "image_paths": image_paths,
        "session_id": session_id,
        "goal": goal,
        "intake_result": None,
        "identity_result": None,
        "gap_result": None,
        "narrator_result": None,
        "error": None,
        "progress_events": [],
    }

    last_seen_events = 0
    final_state = None

    async for state_update in graph.astream(initial_state):
        # astream yields state snapshots from each node
        for node_name, node_state in state_update.items():
            if isinstance(node_state, dict):
                events = node_state.get("progress_events", [])
                # Push any new events
                for event in events[last_seen_events:]:
                    await event_queue.put(event)
                last_seen_events = len(events)

                if node_state.get("error"):
                    await event_queue.put({
                        "type": "error",
                        "message": node_state["error"],
                    })
                    await event_queue.put(None)  # sentinel
                    return None

                if node_state.get("narrator_result"):
                    final_state = node_state

    if final_state and final_state.get("narrator_result"):
        result: NarratorResult = final_state["narrator_result"]
        await event_queue.put({
            "type": "result",
            "data": result.to_dict(),
        })

    await event_queue.put(None)  # sentinel — stream done
    return final_state["narrator_result"] if final_state else None
