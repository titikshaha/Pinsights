# Pinsights v2 — Implementation Plan
## Personal Aesthetic Intelligence System

A multi-agent RAG system that analyses personal fashion image collections to produce aesthetic identity profiles and gap analysis — grounded in a curated fashion history knowledge base.

---

## What We're Building

**Input**: Images (upload, Unsplash query, or load preset board)  
**Output**: Aesthetic DNA profile + Gap Analysis + Cultural Context — specific, reasoned, not generic

### The 4 Features
| Feature | What it does |
|---|---|
| **Aesthetic DNA** | CLIP embed → HDBSCAN cluster → each cluster named + interpreted via fashion history RAG. Output: 2–3 aesthetic worlds, visual tensions, aspiration vs reality |
| **Gap Analysis** | Compares profile against execution criteria knowledge base. Surfaces the specific delta between drawn-to and building-toward. The novel feature |
| **Cultural Context** | Every insight grounded in retrieved history — origin, era, cultural codes, how it evolved |
| **Drift Tracking** | Upload images over time, system tracks whether aesthetic is consolidating, shifting, or contradicting |

---

## Existing Assets (Don't Rebuild)

> [!IMPORTANT]
> The project already has substantial work done. Preserve and migrate:

| Asset | Location | Status | Action |
|---|---|---|---|
| ~1,500 fashion images | `data/images/` + `data/pinterest_img/` | ✅ Ready | Use as demo dataset |
| CLIP embeddings (6MB) | `data/embeddings/embeddings.npy` | ✅ Done | **Migrate to Qdrant** (re-embed with open-clip for consistency) |
| Cluster assignments | `data/clusters/final_clusters.csv` | ✅ 3 clusters | Replace with HDBSCAN |
| Per-pin embeddings | `data/embeddings/per_pin/*.npy` | ✅ Done | Re-embed with open-clip-torch |
| Manifest CSV | `data/embeddings/manifest.csv` | ✅ Done | Update after re-embed |
| Pinterest subsets | `data/pinterest_img/{minimal,rock,streetwear,summer,winter}/` | ✅ 5 categories | Treat as labeled demo boards |
| Metadata CSV | `data/metadata/pins.csv` (1,505 rows) | ✅ Ready | Fix absolute paths → relative |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND (React/Vite)                │
│  Upload → Cluster Map (D3) → Insight Cards → Timeline   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (REST + SSE for streaming)
┌────────────────────▼────────────────────────────────────┐
│                   BACKEND (FastAPI)                      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              LangGraph Orchestration             │   │
│  │                                                  │   │
│  │  IntakeAgent → IdentityAgent → GapAgent          │   │
│  │                              ↓                   │   │
│  │                         NarratorAgent            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  open-clip-torch (ViT-B/32)  │  HDBSCAN  │  PIL KMeans │
└──────────────┬───────────────┴─────────────────────────-┘
               │
        ┌──────▼──────┐
        │   Qdrant    │
        │  (Docker)   │
        │             │
        │ fashion_    │
        │ history     │
        │             │
        │ aesthetic_  │
        │ execution   │
        └─────────────┘
```

---

## Proposed Monorepo Structure

```
pinsights/
├── backend/
│   ├── agents/
│   │   ├── intake_agent.py        # CLIP embed + HDBSCAN + palette
│   │   ├── identity_agent.py      # RAG against fashion_history
│   │   ├── gap_agent.py           # RAG against aesthetic_execution
│   │   └── narrator_agent.py      # Groq synthesis
│   ├── pipeline/
│   │   └── graph.py               # LangGraph wiring
│   ├── rag/
│   │   ├── qdrant_client.py       # Qdrant setup + collection management
│   │   ├── corpus/
│   │   │   ├── fashion_history.json   # ~60 curated chunks
│   │   │   └── aesthetic_execution.json  # execution criteria (secret weapon)
│   │   └── ingest_corpus.py       # embed + upsert to Qdrant
│   ├── ml/
│   │   ├── embedder.py            # open-clip-torch wrapper
│   │   ├── clusterer.py           # HDBSCAN + fallback KMeans
│   │   └── palette.py             # KMeans color extraction
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   ├── routes/
│   │   │   ├── analyze.py         # POST /analyze, GET /analyze/{id}/stream
│   │   │   ├── images.py          # POST /images/upload, GET /images/{id}
│   │   │   └── history.py         # GET /sessions, drift tracking
│   │   └── models.py              # Pydantic schemas
│   ├── data/                      # (existing, keep as-is)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Upload/            # react-dropzone upload zone
│   │   │   ├── ClusterMap/        # D3 force/scatter viz
│   │   │   ├── InsightCard/       # DNA + gap + context cards
│   │   │   ├── DriftTimeline/     # Timeline view
│   │   │   └── ui/                # shared primitives
│   │   ├── pages/
│   │   │   ├── Landing.tsx
│   │   │   ├── Analysis.tsx
│   │   │   └── History.tsx
│   │   ├── hooks/
│   │   │   ├── useAnalysis.ts     # SSE streaming hook
│   │   │   └── useSession.ts
│   │   ├── lib/
│   │   │   └── api.ts             # typed API client
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── scripts/                       # (existing, keep migration scripts)
├── data/                          # (existing dataset)
└── docker-compose.yml             # Qdrant local dev
```

---

## Proposed Changes

### Phase 1 — Backend ML & Ingest Pipeline

#### [MODIFY] Backend structure
Restructure `backend/` into the modular layout above.

#### [NEW] `backend/ml/embedder.py`
- Switch from HuggingFace CLIP to `open-clip-torch` (runs locally, no API dependency)
- Async batch embedding
- Normalize vectors before storing

#### [NEW] `backend/ml/clusterer.py`
- **HDBSCAN** as primary (handles noise points naturally, no need to specify k)
- `min_cluster_size=5`, `min_samples=3`, `metric='euclidean'`
- Fallback to KMeans if fewer than 10 images uploaded
- Returns: cluster labels, representative images (closest to centroid), noise flag

#### [NEW] `backend/ml/palette.py`
- PIL pixel array → KMeans (k=5 colors per image)
- Returns top palette colors as hex strings + frequency weights
- Cluster-level palette: aggregate individual palettes, weighted by frequency

---

### Phase 2 — Qdrant RAG Corpus

> [!IMPORTANT]
> This is the highest-leverage thing to get right. The quality of the RAG corpus determines output quality. Each chunk needs: `text`, `aesthetic`, `era`, `tags`, `chunk_type`.

#### [NEW] `backend/rag/corpus/fashion_history.json`
~60 hand-written chunks covering:
- **90s Minimalism** — Calvin Klein, Jil Sander, origin codes, fabric signatures
- **Streetwear** — 80s NY roots, Supreme era, sportswear crossover signals
- **Dark Academia** — literary aesthetic, European university codes, layering logic
- **Coastal Grandmother** — linen language, quiet luxury signals, relaxed proportion
- **Quiet Luxury** — The Row, Brunello Cucinelli, what signals wealth vs. wanting to signal it
- **Y2K Revival** — low-rise codes, logo saturation, construction notes
- **Workwear/Utilitarian** — Carhartt heritage vs. fashion adoption, construction details
- **Romantic/Cottagecore** — pre-Raphaelite visual vocabulary, Prairie codes
- And ~10 more aesthetic categories

#### [NEW] `backend/rag/corpus/aesthetic_execution.json`
~40 execution criteria chunks (the secret weapon). Format per chunk:
```
aesthetic: "90s Minimalism"
execution_marker: "Proportion discipline"
what_it_requires: "Trousers with a high, clean rise — not high-waisted, specifically a rise that elongates without bunching. Jacket shoulders that land exactly at the edge. No collar gaps."
common_miss: "People buy neutral colours but keep the proportions of fast fashion — boxy tees, mid-rise everything, oversized but unsculpted."
tells: ["rise point", "shoulder seam position", "hem break on trousers", "absence of decorative stitching"]
```

#### [NEW] `backend/rag/qdrant_client.py`
- Two collections: `fashion_history`, `aesthetic_execution`
- Embedding dim: 512 (CLIP ViT-B/32 text encoder for corpus chunks)
- Cosine similarity search
- Helper: `search(collection, query_vector, top_k=5)`

#### [NEW] `backend/rag/ingest_corpus.py`
- Loads JSON corpus files
- CLIP-encodes each chunk text
- Upserts to Qdrant with payload metadata

---

### Phase 3 — Agent Pipeline (LangGraph)

#### [NEW] `backend/agents/intake_agent.py`
**Input**: list of image paths or uploaded files  
**Output**: `IntakeResult` — list of cluster objects, each with:
- `cluster_id`, `size`, `representative_images` (top 5 closest to centroid)
- `dominant_palette` (hex list)
- `embedding_centroid` (vector, for RAG query)
- `noise_pins` (HDBSCAN -1 labels, flagged separately)

#### [NEW] `backend/agents/identity_agent.py`
**Input**: `IntakeResult`  
**Process**:
- For each cluster centroid, query `fashion_history` with top-k=5
- Groq (Llama 3.1 70B): "Given these retrieved chunks and this cluster's visual palette, name this aesthetic world and describe its internal logic"
- Hard prompt constraint: **every claim requires a because**  
**Output**: Named aesthetic worlds, dominant themes, tension between clusters, aspiration reading

#### [NEW] `backend/agents/gap_agent.py`
**Input**: `IdentityResult` (named aesthetics)  
**Process**:
- For each named aesthetic, query `aesthetic_execution` with aesthetic name + palette tags
- Groq: "The user's images signal X aesthetic. Retrieved execution criteria state Y. What specifically is the delta?"  
**Output**: Per-aesthetic gap list — specific named gaps with `what_it_requires` and `common_miss`

#### [NEW] `backend/agents/narrator_agent.py`
**Input**: `IntakeResult` + `IdentityResult` + `GapResult`  
**Process**: Final synthesis prompt — assembles structured JSON with hard schema enforcement  
**Output**:
```json
{
  "aesthetic_dna": {
    "primary_world": "...",
    "secondary_world": "...",
    "tension": "...",
    "aspiration_reading": "..."
  },
  "gaps": [
    {
      "aesthetic": "...",
      "gap_name": "...",
      "what_it_requires": "...",
      "common_miss": "...",
      "your_tell": "..."
    }
  ],
  "cultural_context": [
    {
      "claim": "...",
      "because": "...",
      "source_era": "...",
      "cultural_code": "..."
    }
  ],
  "clusters": [...]
}
```

#### [NEW] `backend/pipeline/graph.py`
LangGraph `StateGraph`:
```
START → intake_node → identity_node → gap_node → narrator_node → END
```
- State: `PipelineState` TypedDict
- Each node is async, streams partial results via SSE

---

### Phase 4 — FastAPI Backend

#### [NEW] `backend/api/main.py`
- FastAPI app with CORS for `localhost:5173` (Vite dev) and Vercel prod domain
- Startup: connect Qdrant, load CLIP model into memory (singleton)
- Lifespan context manager

#### [NEW] `backend/api/routes/analyze.py`
- `POST /analyze` — accepts multipart form (images) or `{"source": "unsplash", "query": "..."}` or `{"source": "preset", "board": "minimal"}`
- Returns `{"session_id": "..."}` immediately
- `GET /analyze/{session_id}/stream` — SSE endpoint, streams pipeline state updates as JSON lines

#### [NEW] `backend/api/routes/images.py`
- `POST /images/upload` — validates, saves, returns `image_id` list
- `GET /images/{image_id}` — serves image file (for frontend cluster map thumbnails)

#### [NEW] `backend/api/routes/history.py`
- `GET /sessions` — returns past analysis sessions
- `GET /sessions/{id}` — returns full result
- `POST /sessions/{id}/drift` — compares two sessions, returns drift signal

---

### Phase 5 — Frontend (React + TypeScript + Vite)

#### [NEW] `frontend/` — Full Vite + React + TypeScript app

**Design Direction**: Dark mode, editorial — like a high-end magazine crossed with a research interface. Not a tool. An experience.
- Color palette: near-black `#0A0A0B`, warm off-white `#F5F0E8`, accent `#C8A882` (warm gold)
- Typography: `Playfair Display` for headings (editorial), `Inter` for body
- Animations: Framer Motion — cluster dots animate in, cards slide up, transitions feel considered

**Key Components**:

`Upload/` — Full-screen drop zone with drag-over state. Shows image previews as they're added. "Analyse" CTA appears when ≥5 images present.

`ClusterMap/` — D3 force-directed graph. Each node = one image, colored by cluster. Nodes drift into clusters as analysis streams in. Click node = highlight cluster. Click cluster = load InsightCard.

`InsightCard/` — The main output component:
- Cluster representative images (3x3 grid)
- Dominant palette swatches
- Aesthetic name (large, editorial)
- Cultural context section with "because" chain
- Gap list with specific named gaps
- Visual tension note (when multiple clusters)

`DriftTimeline/` — Horizontal timeline, each session is a point. Cluster makeup shown as stacked bars. Hover = mini insight.

---

### Phase 6 — Infrastructure

#### [NEW] `docker-compose.yml`
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

#### [MODIFY] `.env`
```
UNSPLASH_ACCESS_KEY=...
GROQ_API_KEY=...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=          # empty for local dev
CLIP_MODEL=ViT-B-32
```

---

## User Review Required

> [!IMPORTANT]
> **Groq API Key**: You'll need a Groq API key (free tier at console.groq.com). Do you have one, or should I build with a placeholder and a mock LLM mode for local dev?

> [!IMPORTANT]
> **Docker**: The Qdrant vector store requires Docker for local dev. Do you have Docker Desktop installed on this machine?

> [!WARNING]
> **Corpus writing**: The `fashion_history.json` and `aesthetic_execution.json` chunks are the highest-leverage part of the system — they determine whether the output sounds generic or genuinely insightful. I'll write a solid first draft (~60 chunks) but you should review and extend these, especially the `aesthetic_execution` chunks which are your "secret weapon." I'll flag these clearly.

> [!NOTE]
> **Re-embedding**: The existing `embeddings.npy` was created with HuggingFace `CLIPProcessor`. The new stack uses `open-clip-torch`. These are compatible (same ViT-B/32 weights), but to be safe I'll add a migration check — if existing embeddings are the right dimension (512), we reuse them; otherwise re-embed.

---

## Open Questions

1. **Demo board**: The `data/pinterest_img/` subfolders (`minimal`, `rock`, `streetwear`, `summer`, `winter`) — should I wire these as selectable "preset boards" in the UI? This gives a great demo moment without needing to upload.

2. **Session persistence**: For drift tracking, sessions need to be persisted. Simple JSON files in `data/sessions/` is enough for v1, or do you want SQLite?

3. **Deployment timing**: Should I build with Railway + Vercel deployment config from the start, or focus on making local dev excellent first and add deployment config at the end?

4. **Corpus depth**: Do you want me to write the full ~60+40 corpus chunks from scratch (I'll make them genuinely good — sourced from fashion history knowledge), or do you have existing notes/references you'd like incorporated?

---

## Build Order

```
Week 1 — ML Pipeline + Qdrant
  Day 1-2: Project scaffold, Docker/Qdrant setup, open-clip embedder
  Day 3-4: HDBSCAN clusterer, palette extractor, corpus JSON first draft
  Day 5-7: Ingest corpus to Qdrant, test retrieval precision on 10 queries

Week 2 — Agent Pipeline
  Day 8-9: IntakeAgent + IdentityAgent, wired to Qdrant
  Day 10-11: GapAgent, NarratorAgent, full Groq prompt engineering
  Day 12-14: LangGraph wiring, test full pipeline end-to-end on existing dataset

Week 3 — FastAPI + SSE Streaming
  Day 15-16: FastAPI app, upload route, image serving
  Day 17-18: SSE streaming, session management, history routes
  Day 19-21: Integration test backend with Postman / CLI

Week 4 — Frontend
  Day 22-23: Vite setup, design system (tokens, typography, dark mode)
  Day 24-25: Upload component, ClusterMap D3 viz
  Day 26-27: InsightCard, streaming integration
  Day 28: DriftTimeline, polish, demo prep
```

---

## Verification Plan

### Automated
- `pytest backend/` — unit tests for embedder, clusterer, palette, each agent in isolation
- `pytest backend/api/` — FastAPI test client for all routes
- Retrieval precision: for 10 manually labelled queries, verify top-1 Qdrant result is correct aesthetic

### Manual
- End-to-end demo: upload the `data/pinterest_img/minimal/` preset, verify output names the aesthetic correctly and surfaces at least 2 specific, named gaps
- Demo under 10s with pre-computed embeddings for the preset boards
- Interview demo moment: upload → cluster map animates in → click cluster → insight card loads with specific cultural context and gaps
