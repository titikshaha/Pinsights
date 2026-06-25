"""
identity_agent.py — Identity Agent: names aesthetic worlds per cluster via RAG + Groq.

For each cluster centroid embedding, retrieves relevant fashion history chunks
from Qdrant and prompts Groq (Llama 3.1 70B) to name and interpret the aesthetic.

Output: IdentityResult with named aesthetic worlds, visual tensions,
        aspiration readings, and retrieved sources.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from backend.agents.intake_agent import IntakeResult, ClusterObject
from backend.rag import qdrant_client as qc


@dataclass
class AestheticWorld:
    cluster_id: int
    name: str                          # e.g. "90s Minimalism" or "Quiet Luxury Edge"
    description: str                   # 2-3 sentence interpretation
    visual_signals: List[str]          # key visual signals detected
    cultural_origin: str               # retrieved cultural context
    aspiration_reading: str            # what this cluster reveals about aspiration
    palette_story: str                 # what the palette communicates
    retrieved_chunks: List[Dict]       # source chunks from fashion_history


@dataclass
class IdentityResult:
    aesthetic_worlds: List[AestheticWorld]
    primary_world: str                 # name of the largest / most coherent cluster
    secondary_world: Optional[str]     # second cluster name, if exists
    visual_tension: str                # the tension between worlds (if multiple)
    overall_aspiration: str            # cross-cluster aspiration reading
    raw_llm_responses: List[str]       # for debugging


def _build_identity_prompt(
    cluster: ClusterObject,
    retrieved_chunks: List[Dict],
) -> str:
    chunks_text = "\n\n---\n\n".join([
        f"Aesthetic: {c.get('aesthetic', '')}\nEra: {c.get('era', '')}\n"
        f"Tags: {', '.join(c.get('tags', []))}\n\n{c.get('text', '')}"
        for c in retrieved_chunks
    ])

    palette_desc = ", ".join(cluster.palette_tags) if cluster.palette_tags else "unclear"

    return f"""You are an expert fashion analyst and cultural historian. Analyse this cluster of fashion images and name its aesthetic world.

CLUSTER DATA:
- Number of images: {cluster.size}
- Dominant colour palette: {", ".join(cluster.dominant_palette[:4])} (described as: {palette_desc})

RETRIEVED FASHION HISTORY CONTEXT:
{chunks_text}

TASK: Based on the visual data and retrieved context, provide a precise aesthetic analysis.

Rules:
1. Every claim MUST be followed by a "because" — no assertion without evidence
2. Be specific, not generic. "minimalism" is not enough; "90s Helmut Lang minimalism with proportion-discipline as the organising principle" is
3. Name the aesthetic world precisely — 2-5 words maximum
4. Surface the aspiration reading: what is the person reaching toward, not just what are they currently doing
5. Be honest about tensions within the cluster if they exist

Respond in this exact JSON format:
{{
  "name": "<precise aesthetic name, 2-5 words>",
  "description": "<2-3 sentence interpretation that uses 'because' at least once>",
  "visual_signals": ["<signal 1>", "<signal 2>", "<signal 3>"],
  "cultural_origin": "<1-2 sentences on cultural/historical origin of this aesthetic>",
  "aspiration_reading": "<what this cluster reveals about aspiration, not just current state>",
  "palette_story": "<what this specific palette communicates within the aesthetic — 1-2 sentences>"
}}"""


def _build_tension_prompt(worlds: List[AestheticWorld]) -> str:
    world_summaries = "\n\n".join([
        f"CLUSTER {w.cluster_id} — {w.name}:\n{w.description}\nAspiration: {w.aspiration_reading}"
        for w in worlds
    ])

    return f"""You are a fashion analyst synthesising multiple aesthetic clusters from the same person's image collection.

AESTHETIC WORLDS DETECTED:
{world_summaries}

TASK: Identify the visual tension between these aesthetics and the overall aspiration signal they produce together.

Rules:
1. The tension must be specific — not "you like both casual and formal" but "you're drawn to the restraint of minimalism but keep the proportions of casual dressing, which prevents either aesthetic from landing"
2. The overall aspiration should reveal something about identity, not just style preference
3. Use "because" in every evaluative claim

Respond in this exact JSON format:
{{
  "visual_tension": "<specific tension between the detected aesthetics — 2-3 sentences>",
  "overall_aspiration": "<what the full collection reveals about aesthetic aspiration — 2-3 sentences>"
}}"""


async def run_identity(
    intake_result: IntakeResult,
    progress_callback=None,
) -> IdentityResult:
    """
    For each cluster, retrieve from fashion_history Qdrant collection
    and prompt Groq to name and interpret the aesthetic.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=1024,
    )

    worlds: List[AestheticWorld] = []
    raw_responses: List[str] = []

    for i, cluster in enumerate(intake_result.clusters):
        if progress_callback:
            pct = 0.80 + (i / len(intake_result.clusters)) * 0.08
            await progress_callback("identity", pct)

        # Query Qdrant with cluster centroid
        results = qc.search(
            collection_name=qc.COLLECTION_FASHION_HISTORY,
            query_vector=cluster.centroid,
            top_k=5,
        )
        retrieved = [r.payload for r in results]

        # Prompt Groq
        prompt = _build_identity_prompt(cluster, retrieved)

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            raw_responses.append(content)

            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            worlds.append(AestheticWorld(
                cluster_id=cluster.cluster_id,
                name=parsed.get("name", f"Aesthetic World {cluster.cluster_id}"),
                description=parsed.get("description", ""),
                visual_signals=parsed.get("visual_signals", []),
                cultural_origin=parsed.get("cultural_origin", ""),
                aspiration_reading=parsed.get("aspiration_reading", ""),
                palette_story=parsed.get("palette_story", ""),
                retrieved_chunks=retrieved,
            ))
        except Exception as e:
            print(f"[Identity] Error parsing response for cluster {cluster.cluster_id}: {e}")
            worlds.append(AestheticWorld(
                cluster_id=cluster.cluster_id,
                name=f"Aesthetic World {cluster.cluster_id}",
                description="Analysis unavailable.",
                visual_signals=[],
                cultural_origin="",
                aspiration_reading="",
                palette_story="",
                retrieved_chunks=retrieved,
            ))

    # Synthesise tension (only if multiple worlds)
    visual_tension = ""
    overall_aspiration = ""

    if len(worlds) > 1:
        if progress_callback:
            await progress_callback("identity_synthesis", 0.87)

        tension_prompt = _build_tension_prompt(worlds)
        try:
            tension_resp = llm.invoke(tension_prompt)
            tension_content = tension_resp.content.strip()
            if "```json" in tension_content:
                tension_content = tension_content.split("```json")[1].split("```")[0].strip()
            elif "```" in tension_content:
                tension_content = tension_content.split("```")[1].split("```")[0].strip()

            tension_parsed = json.loads(tension_content)
            visual_tension = tension_parsed.get("visual_tension", "")
            overall_aspiration = tension_parsed.get("overall_aspiration", "")
            raw_responses.append(tension_resp.content)
        except Exception as e:
            print(f"[Identity] Error parsing tension response: {e}")
            visual_tension = "Multiple aesthetic directions detected — see individual cluster analyses."
            overall_aspiration = worlds[0].aspiration_reading if worlds else ""
    else:
        overall_aspiration = worlds[0].aspiration_reading if worlds else ""

    # Determine primary and secondary world by cluster size
    sorted_worlds = sorted(worlds, key=lambda w: intake_result.clusters[w.cluster_id].size if w.cluster_id < len(intake_result.clusters) else 0, reverse=True)
    primary = sorted_worlds[0].name if sorted_worlds else "Unknown"
    secondary = sorted_worlds[1].name if len(sorted_worlds) > 1 else None

    return IdentityResult(
        aesthetic_worlds=worlds,
        primary_world=primary,
        secondary_world=secondary,
        visual_tension=visual_tension,
        overall_aspiration=overall_aspiration,
        raw_llm_responses=raw_responses,
    )
