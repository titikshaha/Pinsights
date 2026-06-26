"""
gap_agent.py — Gap Agent: retrieves execution criteria and surfaces specific gaps.

The novel feature. For each named aesthetic, queries the aesthetic_execution
Qdrant collection and prompts Groq to identify the specific delta between
what the images signal and what the aesthetic actually requires.

Output: GapResult with per-aesthetic, named, specific gaps.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from backend.agents.intake_agent import IntakeResult, ClusterObject
from backend.agents.identity_agent import IdentityResult, AestheticWorld
from backend.rag import qdrant_client as qc


@dataclass
class Gap:
    aesthetic: str
    gap_name: str                    # e.g. "Proportion discipline", "Fabric quality"
    what_it_requires: str            # what authentic execution requires
    common_miss: str                 # the typical failure mode
    your_tell: List[str]             # the specific signals in the images
    gap_type: str                    # "construction", "material", "styling", "color", etc.
    severity: str                    # "critical" | "moderate" | "minor"
    actionable_step: str             # one specific, concrete thing to do


@dataclass
class GapResult:
    gaps_by_cluster: Dict[int, List[Gap]]  # cluster_id → gaps
    all_gaps: List[Gap]
    primary_gap: Optional[Gap]             # the single most important gap across all clusters
    raw_llm_responses: List[str]


def _build_gap_prompt(
    world: AestheticWorld,
    cluster: ClusterObject,
    retrieved_execution_chunks: List[Dict],
) -> str:
    exec_text = "\n\n---\n\n".join([
        f"EXECUTION MARKER: {c.get('execution_marker', '')}\n"
        f"GAP TYPE: {c.get('gap_type', '')}\n\n"
        f"What it requires:\n{c.get('what_it_requires', '')}\n\n"
        f"Common miss:\n{c.get('common_miss', '')}\n\n"
        f"Tells: {', '.join(c.get('your_tell', []))}"
        for c in retrieved_execution_chunks
    ])

    palette_desc = ", ".join(cluster.palette_tags[:4]) if cluster.palette_tags else "not determined"

    return f"""You are a precise fashion analyst identifying the specific execution gaps in someone's aesthetic.

IDENTIFIED AESTHETIC: {world.name}
DESCRIPTION: {world.description}
ASPIRATION: {world.aspiration_reading}
PALETTE SIGNALS: {palette_desc}
PALETTE STORY: {world.palette_story}
IMAGE COUNT: {cluster.size}

RETRIEVED EXECUTION CRITERIA FOR THIS AESTHETIC:
{exec_text}

TASK: Identify the 2-3 most significant gaps between what this person's images signal and what authentic {world.name} execution actually requires.

Rules:
1. Be specific — not "you need better fit" but "your trousers break at the wrong point, revealing mid-shoe rather than the top of the shoe"
2. Base every gap on the retrieved execution criteria, not on generic fashion advice
3. Every gap needs a concrete, single actionable step — not "invest in quality" but "find one Jil Sander or COS trouser in wool crepe and let it show you what the right proportion feels like"
4. The severity assessment must be honest: a critical gap prevents the aesthetic from landing; a minor gap is a refinement
5. Use "because" in every diagnostic claim

Respond with a JSON array of 2-3 gap objects:
[
  {{
    "gap_name": "<specific name for this gap>",
    "what_it_requires": "<what authentic execution requires — 1-2 sentences>",
    "common_miss": "<the typical failure this collection shows — 1-2 sentences>",
    "your_tell": ["<signal 1>", "<signal 2>"],
    "gap_type": "<one of: construction, material, styling, color, proportion, fit, knowledge>",
    "severity": "<one of: critical, moderate, minor>",
    "actionable_step": "<one specific, concrete action — start with a verb>"
  }},
  ...
]"""


async def run_gap(
    intake_result: IntakeResult,
    identity_result: IdentityResult,
    progress_callback=None,
) -> GapResult:
    """
    For each aesthetic world, retrieve execution criteria from Qdrant
    and prompt Groq to surface specific, named gaps.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,  # slightly lower for precision
        max_tokens=2048,
    )

    gaps_by_cluster: Dict[int, List[Gap]] = {}
    all_gaps: List[Gap] = []
    raw_responses: List[str] = []

    for i, world in enumerate(identity_result.aesthetic_worlds):
        if progress_callback:
            pct = 0.88 + (i / len(identity_result.aesthetic_worlds)) * 0.06
            await progress_callback("gap_analysis", pct)

        # Find the corresponding cluster
        cluster = next(
            (c for c in intake_result.clusters if c.cluster_id == world.cluster_id),
            intake_result.clusters[0] if intake_result.clusters else None
        )
        if cluster is None:
            continue

        # Query aesthetic_execution with cluster centroid
        results = qc.search(
            collection_name=qc.COLLECTION_AESTHETIC_EXECUTION,
            query_vector=cluster.centroid,
            top_k=6,
        )
        retrieved_exec = [r.payload for r in results]

        # Also search with aesthetic name filter for precision
        name_filtered = qc.search(
            collection_name=qc.COLLECTION_AESTHETIC_EXECUTION,
            query_vector=cluster.centroid,
            top_k=4,
            aesthetic_filter=world.name,
        )
        if name_filtered:
            # Merge, deduplicate
            seen_ids = {r.get("id") for r in retrieved_exec}
            for r in name_filtered:
                if r.payload.get("id") not in seen_ids:
                    retrieved_exec.append(r.payload)
                    seen_ids.add(r.payload.get("id"))

        prompt = _build_gap_prompt(world, cluster, retrieved_exec[:6])

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            print(f"[Gap] RAW OUTPUT (Cluster {cluster.cluster_id}):\n{content}")
            raw_responses.append(content)

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            if not isinstance(parsed, list):
                parsed = [parsed]

            cluster_gaps = []
            for g in parsed:
                cluster_gaps.append(Gap(
                    aesthetic=world.name,
                    gap_name=g.get("gap_name", "Unnamed gap"),
                    what_it_requires=g.get("what_it_requires", ""),
                    common_miss=g.get("common_miss", ""),
                    your_tell=g.get("your_tell", []),
                    gap_type=g.get("gap_type", "styling"),
                    severity=g.get("severity", "moderate"),
                    actionable_step=g.get("actionable_step", ""),
                ))

            gaps_by_cluster[world.cluster_id] = cluster_gaps
            all_gaps.extend(cluster_gaps)

        except Exception as e:
            print(f"[Gap] Error parsing gap response for cluster {world.cluster_id}: {e}")
            gaps_by_cluster[world.cluster_id] = []

    # Select primary gap: the first "critical" severity gap, else first gap
    primary_gap = None
    critical_gaps = [g for g in all_gaps if g.severity == "critical"]
    if critical_gaps:
        primary_gap = critical_gaps[0]
    elif all_gaps:
        primary_gap = all_gaps[0]

    return GapResult(
        gaps_by_cluster=gaps_by_cluster,
        all_gaps=all_gaps,
        primary_gap=primary_gap,
        raw_llm_responses=raw_responses,
    )
