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
import re
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
    executive_summary: str

    # Per-cluster summaries
    clusters: List[Dict[str, Any]]

    # Gaps
    gaps: List[Dict[str, Any]]
    primary_gap: Optional[Dict[str, Any]]

    # Filter conclusion (replaces cultural context)
    filter_conclusion: List[str]

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
                "executive_summary": self.executive_summary,
            },
            "clusters": self.clusters,
            "gaps": self.gaps,
            "primary_gap": self.primary_gap,
            "filter_conclusion": self.filter_conclusion,
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
        "styling": "Provide 4-5 highly specific, actionable styling rules to nail this aesthetic authentically. Prepend an appropriate emoji to each point.",
        "shopping": "Provide a curated shopping list of 4-5 exact key pieces/items to buy to build this wardrobe. Prepend an appropriate emoji (like 🧥, 👜, etc.) to each point.",
        "trends": "Provide 4-5 bullet points explaining why this silhouette is trending and where it is evolving next. Prepend an appropriate emoji to each point."
    }.get(goal, "Provide 4-5 specific styling tips. Prepend an emoji to each.")

    return f"""You are synthesising a fashion analysis into a final, highly concise report. 
Do NOT write long paragraphs. Keep it punchy and direct.

DETECTED AESTHETICS:
{worlds_summary}

IDENTIFIED GAPS:
{gaps_summary}

CULTURAL CONTEXT FROM RETRIEVED SOURCES (For your background knowledge):
{cultural_context_text}

TASK: Generate a strictly structured JSON object containing an executive summary and a filter conclusion.

CRITICAL: Your recommendations in the filter_conclusion MUST be strictly grounded in the visual signals of the clothes actually seen in the images. Do not recommend unrelated garments just because they appear in the cultural context. If the images are of black lace dresses, recommend black lace pieces, not sarees or blazers unless they are actually present. Ensure your output is perfectly valid JSON.

1. "executive_summary": Write a single, punchy 2-3 sentence overview of this person's entire aesthetic DNA. What is the vibe?
2. "filter_conclusion": {goal_instruction}

Format strictly as this JSON object:
{{
  "executive_summary": "<2-3 sentences>",
  "filter_conclusion": [
    "<point 1>",
    "<point 2>",
    "<point 3>",
    "<point 4>"
  ]
}}"""


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

    # Build executive summary and filter conclusion
    executive_summary = identity_result.overall_aspiration
    filter_conclusion: List[str] = []
    try:
        narrator_prompt = _build_narrator_prompt(identity_result, gap_result, intake_result, goal)
        response = llm.invoke(narrator_prompt)
        content = response.content.strip()
        print(f"[Narrator] RAW OUTPUT:\n{content}")

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        else:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                content = match.group(0)

        parsed = json.loads(content)
        executive_summary = parsed.get("executive_summary", executive_summary)
        filter_conclusion = parsed.get("filter_conclusion", [])
    except Exception as e:
        print(f"[Narrator] Error building synthesis: {e}")
        filter_conclusion = ["Focus on authentic silhouettes", "Reference original designers from this era"]

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
            "aesthetic_name": world.name if world else f"Cluster {cluster.cluster_id}",
            "description": world.description if world else "",
            "palette_tags": cluster.palette_tags,
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
        executive_summary=executive_summary,
        clusters=clusters,
        gaps=gaps,
        primary_gap=primary_gap_dict,
        filter_conclusion=filter_conclusion,
        session_id=session_id,
        total_images=intake_result.total_images,
        method=intake_result.method,
    )
