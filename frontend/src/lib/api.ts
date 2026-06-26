// API client — typed requests to the FastAPI backend

const BASE = ''  // proxied via Vite

export interface AnalyzeStartResponse {
  session_id: string
  message: string
}

export type ProgressEvent = {
  type: 'progress'
  stage: string
  progress: number   // 0-100
  message: string
}

export type ResultEvent = {
  type: 'result'
  data: AnalysisResult
}

export type ErrorEvent = {
  type: 'error'
  message: string
}

export type HeartbeatEvent = { type: 'heartbeat' }

export type StreamEvent = ProgressEvent | ResultEvent | ErrorEvent | HeartbeatEvent

// ─── Analysis result types ────────────────────────────────────────────────────

export interface AestheticDNA {
  primary_world: string
  secondary_world: string | null
  visual_tension: string
  overall_aspiration: string
}

export interface Gap {
  aesthetic: string
  gap_name: string
  what_it_requires: string
  common_miss: string
  your_tell: string[]
  gap_type: string
  severity: 'critical' | 'moderate' | 'minor'
  actionable_step: string
}

export interface ClusterSummary {
  cluster_id: number
  size: number
  representative_paths: string[]
  dominant_palette: string[]
  palette_tags: string[]
  aesthetic_name: string
  description: string
  visual_signals: string[]
  aspiration_reading: string
  palette_story: string
  cultural_origin: string
  gaps: Gap[]
}

export interface CulturalContextItem {
  claim: string
  because: string
  detailed_analysis: string
  execution_suggestions: string[]
  source_era: string
  cultural_code: string
}

export interface AnalysisResult {
  aesthetic_dna: AestheticDNA
  clusters: ClusterSummary[]
  gaps: Gap[]
  primary_gap: Gap | null
  cultural_context: CulturalContextItem[]
  drift_signal: string | null
  meta: {
    session_id: string
    total_images: number
    method: string
  }
}

export interface SessionSummary {
  session_id: string
  created_at: string
  total_images: number
  primary_world: string
  secondary_world: string | null
  thumbnail_paths: string[]
}

export interface PresetBoard {
  name: string
  display_name: string
  image_count: number
  thumbnail_paths: string[]
}

export interface DriftResult {
  session_a_id: string
  session_b_id: string
  signal: 'consolidating' | 'shifting' | 'contradicting' | 'insufficient_data'
  description: string
  primary_world_change: string | null
}

// ─── API calls ────────────────────────────────────────────────────────────────

export async function startAnalysis(
  source: 'upload' | 'preset' | 'unsplash',
  options: {
    files?: File[]
    board?: string
    query?: string
    goal?: string
  }
): Promise<string> {
  const form = new FormData()
  form.append('source', source)

  if (source === 'upload' && options.files) {
    options.files.forEach(f => form.append('files', f))
  }
  if (source === 'preset' && options.board) {
    form.append('board', options.board)
  }
  if (source === 'unsplash' && options.query) {
    form.append('query', options.query)
  }
  if (options.goal) {
    form.append('goal', options.goal)
  }

  const res = await fetch(`${BASE}/analyze`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Analysis failed to start')
  }
  const data: AnalyzeStartResponse = await res.json()
  return data.session_id
}

export function subscribeToStream(
  sessionId: string,
  onEvent: (event: StreamEvent) => void,
  onDone: () => void
): () => void {
  const es = new EventSource(`${BASE}/analyze/${sessionId}/stream`)

  es.onmessage = (e) => {
    try {
      const event: StreamEvent = JSON.parse(e.data)
      onEvent(event)
      if (event.type === 'result' || event.type === 'error') {
        es.close()
        onDone()
      }
    } catch {}
  }

  es.onerror = () => {
    es.close()
    onDone()
  }

  return () => es.close()
}

export async function fetchPresets(): Promise<PresetBoard[]> {
  const res = await fetch(`${BASE}/images/presets`)
  if (!res.ok) return []
  return res.json()
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${BASE}/history/sessions`)
  if (!res.ok) return []
  return res.json()
}

export async function fetchSession(id: string): Promise<AnalysisResult | null> {
  const res = await fetch(`${BASE}/analyze/${id}`)
  if (!res.ok) return null
  return res.json()
}

export async function fetchDrift(idA: string, idB: string): Promise<DriftResult | null> {
  const res = await fetch(`${BASE}/history/drift/${idA}/${idB}`)
  if (!res.ok) return null
  return res.json()
}

export function imageUrl(path: string): string {
  // Convert local absolute paths to API image endpoints
  if (path.startsWith('http')) return path
  
  // Normalize path separators to forward slash for easier parsing
  const normalizedPath = path.replace(/\\/g, '/')
  
  if (normalizedPath.includes('uploads/')) {
    // Extract session_id and filename
    const parts = normalizedPath.split('uploads/')
    if (parts.length > 1) {
      const subpath = parts[1] // session_id/filename.jpg
      return `${BASE}/images/session/${subpath}`
    }
  } else if (normalizedPath.includes('pinterest_img/')) {
    // Extract board and filename
    const parts = normalizedPath.split('pinterest_img/')
    if (parts.length > 1) {
      const subpath = parts[1] // board/filename.jpg
      return `${BASE}/images/file/${subpath}`
    }
  } else if (normalizedPath.includes('unsplash_cache/')) {
    // Fallback to data endpoint for unsplash images
    const filename = normalizedPath.split('/').pop() || normalizedPath
    return `${BASE}/images/data/${filename}`
  }

  // Generic fallback
  const filename = normalizedPath.split('/').pop() || normalizedPath
  return `${BASE}/images/data/${filename}`
}
