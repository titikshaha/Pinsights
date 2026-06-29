import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { type AnalysisState } from '../hooks/useAnalysis'
import { type ClusterSummary, imageUrl } from '../lib/api'
import { RotateCcw } from 'lucide-react'

// ─── Plan image mapping ───────────────────────────────────────────────────────
const PLAN_IMAGES = [
  'public/images/plan/image2.png',  // dress forms → wardrobe
  'public/images/plan/image.png',   // measuring tape → fit & proportion
  'public/images/plan/image1.png',  // meditative figure → identity
  'public/images/plan/image3.png',  // hand + lipstick → detail & finish
  'public/images/plan/image4.png',  // megaphone → statement
]

// ─── Plan section ─────────────────────────────────────────────────────────────
function PlanSection({ items }: { items: string[] }) {
  if (!items.length) return null
  return (
    <section className="plan-section">
      <div className="plan-section__header">
        <div className="section-label">Your Custom Plan</div>
        <h2 className="plan-section__title">What to do next</h2>
      </div>

      <div className="plan-folder-stack">
        {items.map((item, i) => (
          <motion.div
            key={i}
            className="plan-folder"
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.5, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
          >
            {/* Folder tab */}
            <div className="plan-folder__tab">
              <span className="plan-folder__tab-n">{String(i + 1).padStart(2, '0')}</span>
            </div>

            {/* Folder body */}
            <div className="plan-folder__body">
              {/* Image clipped on left */}
              <div className="plan-folder__img-wrap">
                <img
                  src={PLAN_IMAGES[i % PLAN_IMAGES.length]}
                  alt=""
                  className="plan-folder__img"
                />
              </div>

              {/* Text */}
              <div className="plan-folder__text">
                <p className="plan-folder__item">{item}</p>
              </div>

              {/* Accent stamp */}
              <div className="plan-folder__stamp">PINSIGHTS</div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  )
}

interface AnalysisPageProps {
  state: AnalysisState
  onReset: () => void
}

// ─── Progress view ────────────────────────────────────────────────────────────
function ProgressView({ state }: { state: AnalysisState }) {
  return (
    <div className="progress-view">
      <div className="progress-view__inner">
        <motion.div
          className="progress-view__spinner"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        />
        <div className="progress-view__text">
          <p className="progress-view__stage">{state.message}</p>
          <div className="progress-bar" style={{ width: 260 }}>
            <motion.div
              className="progress-bar__fill"
              animate={{ width: `${state.progress}%` }}
              transition={{ duration: 0.4 }}
            />
          </div>
          <p className="progress-view__pct">{state.progress}%</p>
        </div>
      </div>
    </div>
  )
}

// ─── Polaroid card ────────────────────────────────────────────────────────────
function PolaroidCard({
  src,
  label,
  isHero,
  rotation,
}: {
  src: string
  label: string
  isHero?: boolean
  rotation?: number
}) {
  return (
    <div
      className={`polaroid ${isHero ? 'polaroid--hero' : ''}`}
      style={{ '--rot': `${rotation ?? 0}deg` } as React.CSSProperties}
    >
      <div className="polaroid__img-wrap">
        <img src={src} alt={label} />
      </div>
      <div className="polaroid__label">{label}</div>
    </div>
  )
}

// ─── Expanded cluster panel ────────────────────────────────────────────────────
function ClusterPanel({ cluster, index }: { cluster: ClusterSummary; index: number }) {
  const images = cluster.representative_paths
  const [heroIdx, setHeroIdx] = useState(0)
  const heroSrc = imageUrl(images[heroIdx] ?? images[0])

  const rotations = [-3, 2, -1.5, 3, -2, 1]

  const gap = cluster.gaps[0] ?? null

  return (
    <div className="cluster-panel">
      {/* Left: polaroid scatter */}
      <div className="cluster-panel__left">
        <div className="cluster-panel__board">
          {/* Hero polaroid — large centred */}
          <motion.div
            key={heroIdx}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.35 }}
            className="cluster-panel__hero-wrap"
          >
            <PolaroidCard src={heroSrc} label={`LOOK ${String(index + 1).padStart(2, '0')}`} isHero />
          </motion.div>

          {/* Scattered small polaroids */}
          <div className="cluster-panel__scatter">
            {images.map((path, i) => (
              <button
                key={i}
                className={`scatter-thumb ${i === heroIdx ? 'scatter-thumb--active' : ''}`}
                onClick={() => setHeroIdx(i)}
                title={`Detail ${i + 1}`}
              >
                <PolaroidCard
                  src={imageUrl(path)}
                  label={`DETAIL ${i + 1}`}
                  rotation={rotations[i % rotations.length]}
                />
              </button>
            ))}
          </div>
        </div>

        {/* Colour palette row */}
        <div className="cluster-panel__palette-row">
          {cluster.dominant_palette.slice(0, 8).map((hex, i) => (
            <div key={i} className="cluster-panel__swatch" style={{ background: hex }} title={hex} />
          ))}
        </div>
      </div>

      {/* Right: analysis detail */}
      <div className="cluster-panel__right">
        {/* Cluster number eyebrow */}
        <div className="cluster-panel__eyebrow">
          Aesthetic World {String(index + 1).padStart(2, '0')}
        </div>

        <h2 className="cluster-panel__name">{cluster.aesthetic_name}</h2>

        {cluster.description && (
          <p className="cluster-panel__desc">{cluster.description}</p>
        )}

        <div className="cluster-panel__divider" />

        {/* Palette tags */}
        {cluster.palette_tags.length > 0 && (
          <div className="cluster-panel__tags">
            {cluster.palette_tags.map((tag, i) => (
              <span key={i} className="cluster-tag">{tag}</span>
            ))}
          </div>
        )}

        {/* Gap analysis */}
        {gap && (
          <div className="cluster-panel__gap">
            <div className="cluster-panel__gap-label">
              <span className={`badge badge--${gap.severity}`}>
                {gap.severity === 'critical' ? 'Key Guide' : gap.severity === 'moderate' ? 'Suggestion' : 'Refinement'}
              </span>
              <span className="cluster-panel__gap-name">{gap.gap_name.replace('Gap', 'Guide').replace('gap', 'guide')}</span>
            </div>

            <p className="cluster-panel__gap-body">{gap.what_it_requires}</p>

            {gap.actionable_step && (
              <div className="cluster-panel__gap-action">
                → {gap.actionable_step}
              </div>
            )}

            {gap.your_tell?.length > 0 && (
              <div className="cluster-panel__tells">
                {gap.your_tell.map((tell, i) => (
                  <span key={i} className="cluster-tell">· {tell}</span>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="cluster-panel__meta">
          <span>{cluster.size} images</span>
        </div>
      </div>
    </div>
  )
}

// ─── Cluster thumbnail (left sidebar card) ─────────────────────────────────────
function ClusterThumb({
  cluster,
  index,
  isSelected,
  onClick,
}: {
  cluster: ClusterSummary
  index: number
  isSelected: boolean
  onClick: () => void
}) {
  return (
    <button
      className={`cluster-thumb ${isSelected ? 'cluster-thumb--active' : ''}`}
      onClick={onClick}
    >
      <div className="cluster-thumb__imgs">
        {cluster.representative_paths.slice(0, 3).map((p, i) => (
          <img key={i} src={imageUrl(p)} alt="" />
        ))}
      </div>
      <div className="cluster-thumb__info">
        <span className="cluster-thumb__n">{String(index + 1).padStart(2, '0')}</span>
        <span className="cluster-thumb__name">{cluster.aesthetic_name}</span>
        <span className="cluster-thumb__count">{cluster.size} images</span>
      </div>
      <div className={`cluster-thumb__dot ${isSelected ? 'cluster-thumb__dot--active' : ''}`} />
    </button>
  )
}

// ─── Main Analysis Page ────────────────────────────────────────────────────────
export default function Analysis({ state, onReset }: AnalysisPageProps) {
  const [selectedIdx, setSelectedIdx] = useState<number>(0)
  const result = state.result

  if (state.status === 'running') {
    return (
      <div className="container" style={{ paddingTop: 60 }}>
        <ProgressView state={state} />
        <style>{`
          .progress-view {
            display: flex; align-items: center; justify-content: center; min-height: 65vh;
          }
          .progress-view__inner {
            display: flex; flex-direction: column; align-items: center; gap: var(--space-6);
          }
          .progress-view__spinner {
            width: 52px; height: 52px; border-radius: 50%;
            border: 1.5px solid var(--color-border); border-top-color: var(--color-accent);
          }
          .progress-view__text { display: flex; flex-direction: column; align-items: center; gap: var(--space-3); }
          .progress-view__stage { color: var(--color-text-secondary); font-size: 0.9rem; }
          .progress-view__pct { color: var(--color-text-muted); font-size: 0.8125rem; font-variant-numeric: tabular-nums; }
        `}</style>
      </div>
    )
  }

  if (state.status === 'error') {
    return (
      <div className="analysis-error-wrap container" style={{ paddingTop: 60 }}>
        <div className="analysis-error">
          <div className="analysis-error__icon">!</div>
          <h3>Analysis failed</h3>
          <p>{state.error}</p>
          <button className="btn btn--ghost" onClick={onReset}>Try again</button>
        </div>
        <style>{`
          .analysis-error-wrap { min-height: 70vh; display: flex; align-items: center; justify-content: center; }
          .analysis-error { display: flex; flex-direction: column; align-items: center; gap: var(--space-4); text-align: center; }
          .analysis-error__icon { width: 48px; height: 48px; border-radius: 50%; background: rgba(192,57,43,0.08); border: 1px solid rgba(192,57,43,0.2); color: var(--color-critical); display: flex; align-items: center; justify-content: center; font-size: 1.25rem; font-weight: 700; }
        `}</style>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="empty-state container">
        <div className="empty-state__inner">
          <div className="empty-state__icon">🔍</div>
          <h2 className="empty-state__title">No analysis yet</h2>
          <p className="empty-state__desc">Upload your inspiration board or select a preset to uncover your aesthetic DNA.</p>
          <button className="btn btn--primary" onClick={onReset}>
            Start Analysis
          </button>
        </div>
        <style>{`
          .empty-state { min-height: 70vh; display: flex; align-items: center; justify-content: center; padding-top: 60px; }
          .empty-state__inner { display: flex; flex-direction: column; align-items: center; text-align: center; gap: var(--space-4); max-width: 400px; }
          .empty-state__icon { font-size: 2.5rem; margin-bottom: var(--space-2); }
          .empty-state__title { font-family: var(--font-display); font-size: 2rem; color: var(--color-text-primary); }
          .empty-state__desc { color: var(--color-text-secondary); line-height: 1.6; margin-bottom: var(--space-2); }
        `}</style>
      </div>
    )
  }

  const dna = result.aesthetic_dna
  const clusters = result.clusters
  const activeCluster = clusters[selectedIdx]

  return (
    <div className="ap" style={{ paddingTop: 56 }}>

      {/* ── Top header bar ── */}
      <div className="ap__header">
        <div className="ap__header-left">
          <div className="ap__eyebrow">Your Aesthetic DNA</div>
          <h1 className="ap__title">
            {dna.primary_world}
            {dna.secondary_world && (
              <><span className="ap__sep"> & </span>{dna.secondary_world}</>
            )}
          </h1>
        </div>
        <div className="ap__header-right">
          {dna.executive_summary && (
            <p className="ap__summary">{dna.executive_summary}</p>
          )}
          <div className="ap__meta">
            <span className="badge badge--neutral">{result.meta.total_images} images</span>
            <span className="badge badge--neutral">{clusters.length} worlds</span>
            <button className="btn btn--ghost" onClick={onReset} style={{ marginLeft: 'auto' }}>
              <RotateCcw size={13} /> New analysis
            </button>
          </div>
        </div>
      </div>

      {/* ── Body: sidebar + main panel ── */}
      <div className="ap__body">

        {/* Left sidebar: cluster list */}
        <aside className="ap__sidebar">
          <div className="ap__sidebar-label">Visual Clusters</div>
          <div className="ap__cluster-list">
            {clusters.map((cluster, i) => (
              <motion.div
                key={cluster.cluster_id}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.07 }}
              >
                <ClusterThumb
                  cluster={cluster}
                  index={i}
                  isSelected={selectedIdx === i}
                  onClick={() => setSelectedIdx(i)}
                />
              </motion.div>
            ))}
          </div>
        </aside>

        {/* Main: expanded cluster panel */}
        <main className="ap__main">
          <AnimatePresence mode="wait">
            {activeCluster && (
              <motion.div
                key={activeCluster.cluster_id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
              >
                <ClusterPanel cluster={activeCluster} index={selectedIdx} />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* ── Plan section below everything ── */}
      {result.filter_conclusion?.length > 0 && (
        <PlanSection items={result.filter_conclusion} />
      )}

      <style>{`
        /* ── Page shell ── */
        .ap {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background: var(--color-bg);
        }

        /* ── Header ── */
        .ap__header {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-6);
          padding: var(--space-7) var(--space-7) var(--space-5);
          border-bottom: 1px solid var(--color-border);
          background: var(--color-surface);
        }
        .ap__eyebrow {
          font-size: 0.6rem;
          font-weight: 600;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--color-accent);
          margin-bottom: var(--space-2);
        }
        .ap__title {
          font-family: var(--font-display);
          font-size: clamp(1.75rem, 4vw, 2.75rem);
          letter-spacing: -0.025em;
          line-height: 1.1;
          color: var(--color-text-primary);
        }
        .ap__sep { color: var(--color-accent); }
        .ap__header-right {
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          gap: var(--space-3);
        }
        .ap__summary {
          font-size: 0.875rem;
          line-height: 1.7;
          color: var(--color-text-secondary);
          max-width: 480px;
        }
        .ap__meta {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          flex-wrap: wrap;
        }

        /* ── Body layout ── */
        .ap__body {
          display: grid;
          grid-template-columns: 280px 1fr;
          flex: 1;
          min-height: 0;
          overflow: hidden;
        }

        /* ── Sidebar ── */
        .ap__sidebar {
          border-right: 1px solid var(--color-border);
          overflow-y: auto;
          padding: var(--space-5) var(--space-4);
          background: var(--color-surface);
          display: flex;
          flex-direction: column;
          gap: var(--space-5);
        }
        .ap__sidebar-label {
          font-size: 0.6rem;
          font-weight: 600;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--color-text-muted);
        }
        .ap__cluster-list { display: flex; flex-direction: column; gap: var(--space-2); }

        /* ── Cluster thumbnail ── */
        .cluster-thumb {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3);
          background: transparent;
          border: 1px solid transparent;
          border-radius: var(--radius-md);
          cursor: pointer;
          width: 100%;
          text-align: left;
          transition: all var(--duration-base) var(--ease-out);
          position: relative;
        }
        .cluster-thumb:hover {
          background: var(--color-surface-2);
          border-color: var(--color-border);
        }
        .cluster-thumb--active {
          background: var(--color-accent-dim);
          border-color: rgba(200, 48, 90, 0.2);
        }
        .cluster-thumb__imgs {
          display: flex;
          width: 52px;
          height: 36px;
          flex-shrink: 0;
          position: relative;
        }
        .cluster-thumb__imgs img {
          position: absolute;
          width: 28px;
          height: 28px;
          object-fit: cover;
          border-radius: 3px;
          border: 1.5px solid var(--color-surface);
        }
        .cluster-thumb__imgs img:nth-child(1) { left: 0; top: 4px; z-index: 1; }
        .cluster-thumb__imgs img:nth-child(2) { left: 12px; top: 0; z-index: 2; }
        .cluster-thumb__imgs img:nth-child(3) { left: 24px; top: 4px; z-index: 3; }
        .cluster-thumb__info {
          display: flex;
          flex-direction: column;
          gap: 2px;
          flex: 1;
          min-width: 0;
        }
        .cluster-thumb__n {
          font-size: 0.55rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          color: var(--color-accent);
        }
        .cluster-thumb__name {
          font-size: 0.8125rem;
          font-weight: 500;
          color: var(--color-text-primary);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .cluster-thumb__count {
          font-size: 0.6875rem;
          color: var(--color-text-muted);
        }
        .cluster-thumb__dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--color-border-hover);
          flex-shrink: 0;
          transition: background var(--duration-fast);
        }
        .cluster-thumb__dot--active { background: var(--color-accent); }

        /* ── Plan ── */
        .ap__plan {
          margin-top: auto;
          padding-top: var(--space-5);
          border-top: 1px solid var(--color-border);
        }
        .ap__plan-label {
          font-size: 0.6rem;
          font-weight: 600;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--color-text-muted);
          margin-bottom: var(--space-3);
        }
        .ap__plan-item {
          font-size: 0.8125rem;
          color: var(--color-text-secondary);
          line-height: 1.6;
          margin-bottom: var(--space-2);
        }

        /* ── Main panel ── */
        .ap__main {
          overflow-y: auto;
          padding: var(--space-6) var(--space-7);
          background: var(--color-bg);
        }

        /* ── Cluster panel ── */
        .cluster-panel {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-8);
          min-height: 100%;
        }

        /* ── Left: polaroid board ── */
        .cluster-panel__left {
          display: flex;
          flex-direction: column;
          gap: var(--space-5);
        }

        .cluster-panel__board {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: var(--space-4);
        }

        .cluster-panel__hero-wrap {
          align-self: center;
        }

        /* ── Polaroid ── */
        .polaroid {
          background: #fff;
          padding: 10px 10px 32px;
          box-shadow: 0 3px 14px rgba(15,14,12,0.12), 0 1px 3px rgba(15,14,12,0.06);
          transform: rotate(var(--rot, 0deg));
          display: inline-block;
          flex-shrink: 0;
        }
        .polaroid--hero {
          padding: 12px 12px 44px;
          box-shadow: 0 8px 32px rgba(15,14,12,0.15), 0 2px 8px rgba(15,14,12,0.08);
          transform: none;
        }
        .polaroid__img-wrap { overflow: hidden; }
        .polaroid__img-wrap img {
          display: block;
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        .polaroid--hero .polaroid__img-wrap {
          width: 300px;
          height: 360px;
        }
        .polaroid:not(.polaroid--hero) .polaroid__img-wrap {
          width: 80px;
          height: 80px;
        }
        .polaroid__label {
          text-align: center;
          font-size: 0.5rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          color: var(--color-text-muted);
          margin-top: 8px;
          font-family: var(--font-body);
        }

        /* ── Scatter row ── */
        .cluster-panel__scatter {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-4);
          justify-content: center;
          padding: var(--space-3) 0;
        }
        .scatter-thumb {
          background: none;
          border: none;
          cursor: pointer;
          transition: transform var(--duration-base) var(--ease-out);
          padding: 0;
        }
        .scatter-thumb:hover { transform: translateY(-4px) scale(1.05); }
        .scatter-thumb--active .polaroid {
          box-shadow: 0 3px 14px rgba(200,48,90,0.25), 0 0 0 2px var(--color-accent);
        }

        /* ── Palette row ── */
        .cluster-panel__palette-row {
          display: flex;
          gap: 6px;
          justify-content: center;
        }
        .cluster-panel__swatch {
          width: 22px;
          height: 22px;
          border-radius: 50%;
          box-shadow: 0 0 0 1.5px rgba(15,14,12,0.10);
          flex-shrink: 0;
        }

        /* ── Right: detail ── */
        .cluster-panel__right {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
          padding-top: var(--space-2);
        }
        .cluster-panel__eyebrow {
          font-size: 0.6rem;
          font-weight: 600;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--color-accent);
        }
        .cluster-panel__name {
          font-family: var(--font-display);
          font-size: clamp(1.75rem, 3vw, 2.5rem);
          letter-spacing: -0.025em;
          line-height: 1.1;
          color: var(--color-text-primary);
        }
        .cluster-panel__desc {
          font-size: 0.9375rem;
          line-height: 1.75;
          color: var(--color-text-secondary);
        }
        .cluster-panel__divider {
          height: 1px;
          background: var(--color-border);
        }
        .cluster-panel__tags {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-2);
        }
        .cluster-tag {
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          color: var(--color-text-secondary);
          padding: 3px var(--space-3);
          border-radius: 2px;
          font-size: 0.6875rem;
          font-weight: 500;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        .cluster-panel__gap {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: var(--space-5);
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }
        .cluster-panel__gap-label {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }
        .cluster-panel__gap-name {
          font-size: 0.875rem;
          font-weight: 600;
          color: var(--color-text-primary);
        }
        .cluster-panel__gap-body {
          font-size: 0.875rem;
          line-height: 1.7;
          color: var(--color-text-secondary);
        }
        .cluster-panel__gap-action {
          font-size: 0.875rem;
          font-weight: 500;
          color: var(--color-accent);
          padding-top: var(--space-2);
          border-top: 1px solid var(--color-border);
        }
        .cluster-panel__tells {
          display: flex;
          flex-direction: column;
          gap: var(--space-1);
        }
        .cluster-tell {
          font-size: 0.8125rem;
          color: var(--color-text-muted);
          font-style: italic;
        }
        .cluster-panel__meta {
          margin-top: auto;
          padding-top: var(--space-4);
          font-size: 0.75rem;
          color: var(--color-text-muted);
          letter-spacing: 0.06em;
          text-transform: uppercase;
        }

        /* ── Responsive ── */
        @media (max-width: 900px) {
          .ap__header { grid-template-columns: 1fr; }
          .ap__body { grid-template-columns: 1fr; }
          .ap__sidebar { border-right: none; border-bottom: 1px solid var(--color-border); }
          .cluster-panel { grid-template-columns: 1fr; }
          .plan-folder-stack { grid-template-columns: 1fr; }
        }

        /* ── Plan Section ── */
        .plan-section {
          border-top: 1px solid var(--color-border);
          background: var(--color-surface-2);
          padding: var(--space-9) var(--space-7);
        }
        .plan-section__header {
          margin-bottom: var(--space-8);
        }
        .plan-section__title {
          font-family: var(--font-display);
          font-size: clamp(1.75rem, 3.5vw, 2.75rem);
          letter-spacing: -0.025em;
          line-height: 1.1;
          margin-top: var(--space-2);
          color: var(--color-text-primary);
        }

        .plan-folder-stack {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
          gap: var(--space-4);
        }

        .plan-folder {
          background: #F3EFE6;
          border-radius: 3px 12px 3px 3px;
          overflow: hidden;
          position: relative;
          box-shadow: 0 4px 18px rgba(15,14,12,0.09), 0 1px 4px rgba(15,14,12,0.05);
          border: 1px solid rgba(15,14,12,0.07);
          transition: transform var(--duration-base) var(--ease-out), box-shadow var(--duration-base);
        }
        .plan-folder:hover {
          transform: translateY(-3px);
          box-shadow: 0 10px 32px rgba(15,14,12,0.13), 0 2px 6px rgba(15,14,12,0.06);
        }

        .plan-folder__tab {
          height: 28px;
          background: #D4C9B4;
          border-bottom: 1px solid rgba(15,14,12,0.08);
          display: flex;
          align-items: center;
          padding: 0 var(--space-4);
          border-radius: 0 10px 0 0;
          position: relative;
          width: 120px;
          margin-left: var(--space-5);
          margin-bottom: -1px;
        }
        .plan-folder__tab-n {
          font-size: 0.6rem;
          font-weight: 700;
          letter-spacing: 0.14em;
          color: rgba(15,14,12,0.45);
          text-transform: uppercase;
        }

        .plan-folder__body {
          display: flex;
          align-items: center;
          gap: 0;
          min-height: 130px;
          border-top: 1px solid rgba(15,14,12,0.08);
          position: relative;
          overflow: hidden;
        }

        .plan-folder__img-wrap {
          width: 110px;
          min-width: 110px;
          height: 130px;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255,255,255,0.4);
          border-right: 1px solid rgba(15,14,12,0.07);
          flex-shrink: 0;
        }
        .plan-folder__img {
          width: 90px;
          height: 110px;
          object-fit: contain;
          mix-blend-mode: multiply;
          display: block;
        }

        .plan-folder__text {
          flex: 1;
          padding: var(--space-5) var(--space-5);
        }
        .plan-folder__item {
          font-size: 0.9375rem;
          line-height: 1.65;
          color: rgba(15,14,12,0.78);
          font-weight: 400;
        }

        .plan-folder__stamp {
          position: absolute;
          bottom: var(--space-3);
          right: var(--space-4);
          font-size: 0.5rem;
          font-weight: 700;
          letter-spacing: 0.18em;
          color: var(--color-accent);
          opacity: 0.5;
          text-transform: uppercase;
        }
      `}</style>
    </div>
  )
}
