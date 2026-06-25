# Pinsights v2 — Personal Aesthetic Intelligence System

> Not what you like. **Who you are.**

Pinsights is a multi-agent RAG system that analyses your saved fashion images and produces: an **Aesthetic DNA profile**, a precise **Gap Analysis** between aspiration and execution, and **grounded cultural context** — all specific, all reasoned, none generic.

## Architecture

```
Images → CLIP Embedder → HDBSCAN Clusterer → Qdrant RAG → Groq LLM
                                                    ↓
                               Intake → Identity → Gap → Narrator
                                         (LangGraph pipeline)
                                                    ↓
                                        FastAPI (SSE streaming)
                                                    ↓
                                        React/Vite/D3 frontend
```

### The Four Agents

| Agent | Function |
|-------|----------|
| **Intake** | CLIP embeds images → HDBSCAN clusters → palette extraction |
| **Identity** | Queries `fashion_history` Qdrant collection → names aesthetic worlds with Groq |
| **Gap** | Queries `aesthetic_execution` Qdrant collection → surfaces specific execution gaps |
| **Narrator** | Synthesises all outputs → structured JSON with cultural context statements |

### RAG Collections

| Collection | Chunks | Purpose |
|------------|--------|---------|
| `fashion_history` | 60 | Cultural origin, era, visual codes, fabric language for 15+ aesthetics |
| `aesthetic_execution` | 40 | Specific execution criteria, common misses, actionable tells |

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop (for Qdrant)
- Groq API key

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Python environment

```bash
cd backend
pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in:
#   GROQ_API_KEY=your_key
#   UNSPLASH_ACCESS_KEY=your_key (optional)
```

### 4. Ingest the RAG corpus

```bash
python -m backend.rag.ingest_corpus
```

This embeds the 100 fashion history and execution chunks into Qdrant. Run once.
Use `--force` to re-ingest after editing corpus JSON files.

### 5. Start the backend

```bash
uvicorn backend.api.main:app --reload --port 8000
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

## Data

### Preset boards

Pre-existing boards in `data/pinterest_img/`:
- `minimal/` — Minimalism and quiet luxury
- `rock/` — Rock and edge aesthetics
- `streetwear/` — Streetwear
- `summer/` — Summer light
- `winter/` — Winter layers

### Upload your own

Drag and drop any collection of images (min. 5, max. 50). JPG, PNG, WebP.

## API Reference

```
POST /analyze                    — Start analysis (multipart form)
GET  /analyze/{id}/stream        — SSE progress stream → final result
GET  /analyze/{id}               — Get stored result
GET  /images/presets             — List preset boards
GET  /history/sessions           — List past sessions
GET  /history/drift/{a}/{b}      — Compare two sessions
GET  /health                     — Health check
```

Full interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure

```
pinsights/
├── backend/
│   ├── agents/
│   │   ├── intake_agent.py       # CLIP + HDBSCAN + palette
│   │   ├── identity_agent.py     # Qdrant RAG → Groq → aesthetic names
│   │   ├── gap_agent.py          # Qdrant RAG → Groq → gap analysis
│   │   └── narrator_agent.py     # Synthesis → final JSON
│   ├── ml/
│   │   ├── embedder.py           # open-clip-torch wrapper
│   │   ├── clusterer.py          # HDBSCAN + KMeans fallback
│   │   └── palette.py            # KMeans color extraction
│   ├── pipeline/
│   │   └── graph.py              # LangGraph StateGraph + SSE streaming
│   ├── rag/
│   │   ├── corpus/
│   │   │   ├── fashion_history.json     # 60 cultural chunks
│   │   │   └── aesthetic_execution.json # 40 execution criteria chunks
│   │   ├── qdrant_client.py      # Vector store connection + search
│   │   └── ingest_corpus.py      # Embed + upsert corpus
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analyze.py        # Analysis pipeline routes
│   │   │   ├── images.py         # Image serving
│   │   │   └── history.py        # Session history + drift
│   │   ├── models.py             # Pydantic schemas
│   │   ├── session_store.py      # JSON file persistence
│   │   └── main.py               # FastAPI app
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Upload/UploadZone.tsx    # Dropzone + preset selector
│       │   ├── ClusterMap/ClusterMap.tsx # D3 force-directed image map
│       │   └── InsightCard/InsightCard.tsx # Expandable cluster card
│       ├── pages/
│       │   ├── Landing.tsx
│       │   └── Analysis.tsx
│       ├── hooks/useAnalysis.ts         # Analysis state + SSE lifecycle
│       ├── lib/api.ts                   # Typed API client
│       └── index.css                    # Design system
├── data/
│   └── pinterest_img/
│       ├── minimal/
│       ├── rock/
│       ├── streetwear/
│       ├── summer/
│       └── winter/
├── docker-compose.yml
└── .env.example
```
