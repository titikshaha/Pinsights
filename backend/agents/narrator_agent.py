"""
narrator_agent.py — Narrator Agent: synthesises all upstream outputs into
structured final JSON for the frontend.

Hard constraints:
  - Every claim has a "because"
  - Output is strict JSON matching the frontend schema
  - Nothing generic — all insight must be grounded in retrieved chunks
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from backend.agents.intake_agent import IntakeResult
from backend.agents.identity_agent import IdentityResult
from backend.agents.gap_agent import GapResult


@dataclass
class NarratorResult:
    """Final structured output from the full pipeline."""

    # Aesthetic DNA
    primary_world: str
    secondary_world: Optional[str]
    visual_tension: str
    overall_aspiration: str

    # Per-cluster summaries
    clusters: List[Dict[str, Any]]

    # Gaps
    gaps: List[Dict[str, Any]]
    primary_gap: Optional[Dict[str, Any]]

    # Cultural context (retrieved + synthesised)
    cultural_context: List[Dict[str, Any]]

    # Drift signal (populated by history route, empty on first analysis)
    drift_signal: Optional[str] = None

    # Meta
    session_id: str = ""
    total_images: int = 0
    method: str = "hdbscan"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aesthetic_dna": {
                "primary_world": self.primary_world,
                "secondary_world": self.secondary_world,
                "visual_tension": self.visual_tension,
                "overall_aspiration": self.overall_aspiration,
            },
            "clusters": self.clusters,
            "gaps": self.gaps,
            "primary_gap": self.primary_gap,
            "cultural_context": self.cultural_context,
            "drift_signal": self.drift_signal,
            "meta": {
                "session_id": self.session_id,
                "total_images": self.total_images,
                "method": self.method,
            },
        }


def _build_narrator_prompt(
    identity_result: IdentityResult,
    gap_result: GapResult,
    intake_result: IntakeResult,
    goal: str,
) -> str:
    worlds_summary = "\n\n".join([
        f"AESTHETIC: {w.name}\nDESCRIPTION: {w.description}\nASPIRATION: {w.aspiration_reading}"
        for w in identity_result.aesthetic_worlds
    ])

    gaps_summary = "\n\n".join([
        f"GAP: {g.gap_name} ({g.severity})\nAESTHETIC: {g.aesthetic}\n"
        f"WHAT IT REQUIRES: {g.what_it_requires}\nCOMMON MISS: {g.common_miss}\n"
        f"ACTIONABLE STEP: {g.actionable_step}"
        for g in gap_result.all_gaps[:6]
    ])

    cultural_chunks = []
    for w in identity_result.aesthetic_worlds:
        for chunk in w.retrieved_chunks[:2]:
            cultural_chunks.append(
                f"Aesthetic: {chunk.get('aesthetic', '')}\n"
                f"Era: {chunk.get('era', '')}\n"
                f"Context: {chunk.get('text', '')[:300]}..."
            )

    cultural_context_text = "\n\n".join(cultural_chunks[:4])

    goal_instruction = {
        "styling": "Provide an array of execution_suggestions (specific styling tips, material choices, or silhouettes to nail this aesthetic authentically)",
        "shopping": "Provide an array of execution_suggestions detailing the exact key pieces/items to buy to build this wardrobe",
        "trends": "Provide an array of execution_suggestions explaining why this silhouette is trending and where it is evolving next"
    }.get(goal, "Provide an array of execution_suggestions (specific styling tips, material choices, or silhouettes to nail this aesthetic authentically)")

    return f"""You are synthesising a fashion analysis into a final, precise report. 

DETECTED AESTHETICS:
{worlds_summary}

VISUAL TENSION: {identity_result.visual_tension}
OVERALL ASPIRATION: {identity_result.overall_aspiration}

IDENTIFIED GAPS:
{gaps_summary}

CULTURAL CONTEXT FROM RETRIEVED SOURCES:
{cultural_context_text}

TASK: Write 3-4 cultural context statements that ground the analysis in fashion history. Each statement must:
1. Make a specific claim about what the images reveal
2. Follow it with "because [cultural/historical reason]"
3. Include a detailed_analysis section (exactly 2 highly effective sentences hitting the right keywords) exploring the origins, evolution, and cultural meaning of this aesthetic
4. {goal_instruction}
5. Cite the source era and context
6. Be specific to THIS person's collection — not generic aesthetic description

Format as JSON array:
[
  {{
    "claim": "<specific claim about what the images reveal>",
    "because": "<the cultural/historical reason this is what it is>",
    "detailed_analysis": "<2-3 sentences diving deep into the history and nuance of this aesthetic signal>",
    "execution_suggestions": ["<specific styling tip 1>", "<specific styling tip 2>"],
    "source_era": "<e.g. '1990s New York'>",
    "cultural_code": "<what this signal means in its cultural context>"
  }}
]"""


async def run_narrator(
    intake_result: IntakeResult,
    identity_result: IdentityResult,
    gap_result: GapResult,
    session_id: str = "",
    goal: str = "styling",
    progress_callback=None,
) -> NarratorResult:
    """
    Synthesise all upstream outputs into the final structured result.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4,  # slightly higher for narrative synthesis
        max_tokens=4096,
    )

    if progress_callback:
        await progress_callback("narrating", 0.94)

    # Build cultural context statements
    cultural_context: List[Dict[str, Any]] = []
    try:
        narrator_prompt = _build_narrator_prompt(identity_result, gap_result, intake_result, goal)
        response = llm.invoke(narrator_prompt)
        content = response.content.strip()
        print(f"[Narrator] RAW OUTPUT:\n{content}")

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        cultural_context = json.loads(content)
        if not isinstance(cultural_context, list):
            cultural_context = [cultural_context]
    except Exception as e:
        print(f"[Narrator] Error building cultural context: {e}")
        # Fallback: build from retrieved chunks directly
        for world in identity_result.aesthetic_worlds:
            for chunk in world.retrieved_chunks[:1]:
                cultural_context.append({
                    "claim": f"Your images signal {world.name}",
                    "because": chunk.get("text", "")[:200],
                    "detailed_analysis": chunk.get("text", "")[:400],
                    "execution_suggestions": ["Focus on authentic silhouettes", "Reference original designers from this era"],
                    "source_era": chunk.get("era", ""),
                    "cultural_code": ", ".join(chunk.get("tags", [])[:5]),
                })

    # Build cluster summaries for frontend
    clusters = []
    for cluster in intake_result.clusters:
        world = next(
            (w for w in identity_result.aesthetic_worlds if w.cluster_id == cluster.cluster_id),
            None
        )
        cluster_gaps = gap_result.gaps_by_cluster.get(cluster.cluster_id, [])

        clusters.append({
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "representative_paths": cluster.representative_paths,
            "dominant_palette": cluster.dominant_palette,
            "palette_tags": cluster.palette_tags,
            "aesthetic_name": world.name if world else f"Cluster {cluster.cluster_id}",
            "description": world.description if world else "",
            "visual_signals": world.visual_signals if world else [],
            "aspiration_reading": world.aspiration_reading if world else "",
            "palette_story": world.palette_story if world else "",
            "cultural_origin": world.cultural_origin if world else "",
            "gaps": [
                {
                    "gap_name": g.gap_name,
                    "what_it_requires": g.what_it_requires,
                    "common_miss": g.common_miss,
                    "your_tell": g.your_tell,
                    "gap_type": g.gap_type,
                    "severity": g.severity,
                    "actionable_step": g.actionable_step,
                }
                for g in cluster_gaps
            ],
        })

    # Build gap list
    gaps = [
        {
            "aesthetic": g.aesthetic,
            "gap_name": g.gap_name,
            "what_it_requires": g.what_it_requires,
            "common_miss": g.common_miss,
            "your_tell": g.your_tell,
            "gap_type": g.gap_type,
            "severity": g.severity,
            "actionable_step": g.actionable_step,
        }
        for g in gap_result.all_gaps
    ]

    primary_gap_dict = None
    if gap_result.primary_gap:
        g = gap_result.primary_gap
        primary_gap_dict = {
            "aesthetic": g.aesthetic,
            "gap_name": g.gap_name,
            "what_it_requires": g.what_it_requires,
            "common_miss": g.common_miss,
            "your_tell": g.your_tell,
            "gap_type": g.gap_type,
            "severity": g.severity,
            "actionable_step": g.actionable_step,
        }

    if progress_callback:
        await progress_callback("complete", 1.0)

    return NarratorResult(
        primary_world=identity_result.primary_world,
        secondary_world=identity_result.secondary_world,
        visual_tension=identity_result.visual_tension,
        overall_aspiration=identity_result.overall_aspiration,
        clusters=clusters,
        gaps=gaps,
        primary_gap=primary_gap_dict,
        cultural_context=cultural_context,
        session_id=session_id,
        total_images=intake_result.total_images,
        method=intake_result.method,
    )
