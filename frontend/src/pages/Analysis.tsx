import { useState } from 'react'
import { motion } from 'framer-motion'
import { type AnalysisState } from '../hooks/useAnalysis'
import ClusterMap from '../components/ClusterMap/ClusterMap'
import InsightCard from '../components/InsightCard/InsightCard'
import { type CulturalContextItem } from '../lib/api'

interface AnalysisPageProps {
  state: AnalysisState
  onReset: () => void
}

function ProgressView({ state }: { state: AnalysisState }) {
  return (
    <div className="progress-view">
      <div className="progress-view__inner">
        <motion.div
          className="progress-view__orb"
          animate={{ scale: [1, 1.08, 1], opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />
        <div className="progress-view__text">
          <p className="progress-view__stage">{state.message}</p>
          <div className="progress-bar" style={{ width: 280 }}>
            <motion.div
              className="progress-bar__fill"
              animate={{ width: `${state.progress}%` }}
              transition={{ duration: 0.4 }}
            />
          </div>
          <p className="progress-view__pct">{state.progress}%</p>
        </div>
      </div>
      <style>{`
        .progress-view { display: flex; align-items: center; justify-content: center; min-height: 60vh; }
        .progress-view__inner { display: flex; flex-direction: column; align-items: center; gap: var(--space-6); }
        .progress-view__orb {
          width: 80px; height: 80px; border-radius: 50%;
          background: radial-gradient(circle, rgba(200,168,130,0.4) 0%, transparent 70%);
          border: 1px solid rgba(200,168,130,0.3);
        }
        .progress-view__text { display: flex; flex-direction: column; align-items: center; gap: var(--space-3); }
        .progress-view__stage { color: var(--color-text-secondary); font-size: 0.9375rem; }
        .progress-view__pct { color: var(--color-text-muted); font-size: 0.8125rem; }
      `}</style>
    </div>
  )
}

function CulturalContext({ items }: { items: CulturalContextItem[] }) {
  return (
    <div className="cultural-context">
      <div className="section-label">Cultural Context & Execution Insights</div>
      <div className="cultural-items">
        {items.map((item, i) => (
          <motion.div
            key={i}
            className="cultural-item"
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <p className="cultural-item__claim">{item.claim}</p>
            <div className="because">{item.because}</div>
            
            {item.detailed_analysis && (
              <div className="cultural-item__analysis">
                {item.detailed_analysis}
              </div>
            )}
            
            {item.execution_suggestions && item.execution_suggestions.length > 0 && (
              <div className="cultural-item__suggestions">
                <strong>Execution Suggestions:</strong>
                <ul>
                  {item.execution_suggestions.map((suggestion, j) => (
                    <li key={j}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="cultural-item__meta">
              <span>{item.source_era}</span>
              {item.cultural_code && <span>· {item.cultural_code}</span>}
            </div>
          </motion.div>
        ))}
      </div>
      <style>{`
        .cultural-context { margin-top: var(--space-6); }
        .cultural-items { display: flex; flex-direction: column; gap: var(--space-4); margin-top: var(--space-3); }
        .cultural-item { padding: var(--space-4); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); }
        .because { font-style: italic; color: var(--color-text-secondary); margin-bottom: var(--space-4); font-size: 0.9375rem; }
        .cultural-item__analysis { margin-bottom: var(--space-4); font-size: 0.9375rem; line-height: 1.6; color: var(--color-text-secondary); border-left: 2px solid var(--color-border); padding-left: var(--space-3); }
        .cultural-item__suggestions { margin-bottom: var(--space-3); font-size: 0.9375rem; }
        .cultural-item__suggestions strong { display: block; margin-bottom: var(--space-2); color: var(--color-text-primary); font-weight: 500; }
        .cultural-item__suggestions ul { padding-left: var(--space-4); margin: 0; color: var(--color-text-secondary); }
        .cultural-item__suggestions li { margin-bottom: var(--space-1); }
        .cultural-item__meta { display: flex; gap: var(--space-2); margin-top: var(--space-2); font-size: 0.75rem; color: var(--color-text-muted); }
      `}</style>
    </div>
  )
}

export default function Analysis({ state, onReset }: AnalysisPageProps) {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  const result = state.result

  if (state.status === 'running') {
    return (
      <div className="container" style={{ paddingTop: 80 }}>
        <ProgressView state={state} />
      </div>
    )
  }

  if (state.status === 'error') {
    return (
      <div className="container analysis-error" style={{ paddingTop: 80 }}>
        <h2>Analysis failed</h2>
        <p>{state.error}</p>
        <button className="btn btn--ghost" onClick={onReset}>Try again</button>
        <style>{`
          .analysis-error { padding-top: 120px; display: flex; flex-direction: column; align-items: center; gap: var(--space-4); text-align: center; }
        `}</style>
      </div>
    )
  }

  if (!result) return null

  const dna = result.aesthetic_dna

  return (
    <div className="analysis-page" style={{ paddingTop: 80 }}>
      <div className="container">

        {/* DNA header */}
        <motion.div
          className="dna-header"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="section-label">Your aesthetic DNA</div>
          <h1 className="dna-title">
            {dna.primary_world}
            {dna.secondary_world && <> <span className="dna-separator">&</span> {dna.secondary_world}</>}
          </h1>
          {dna.visual_tension && (
            <div className="because dna-tension">{dna.visual_tension}</div>
          )}
          {dna.overall_aspiration && (
            <p className="dna-aspiration">{dna.overall_aspiration}</p>
          )}
          <div className="dna-meta">
            <span className="badge badge--neutral">{result.meta.total_images} images analysed</span>
            <span className="badge badge--neutral">{result.clusters.length} aesthetic worlds</span>
            <button className="btn btn--ghost" style={{ marginLeft: 'auto' }} onClick={onReset}>
              New analysis
            </button>
          </div>
        </motion.div>

        {/* Cluster map */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <ClusterMap
            clusters={result.clusters}
            onSelectCluster={setSelectedCluster}
            selectedClusterId={selectedCluster}
          />
          <p className="map-hint">Click a cluster to explore its aesthetic world</p>
        </motion.div>

        {/* Insight cards */}
        <div className="insight-grid">
          {result.clusters.map((cluster, i) => (
            <motion.div
              key={cluster.cluster_id}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.1 }}
            >
              <InsightCard
                cluster={cluster}
                isSelected={selectedCluster === cluster.cluster_id}
                onClick={() => setSelectedCluster(
                  selectedCluster === cluster.cluster_id ? null : cluster.cluster_id
                )}
              />
            </motion.div>
          ))}
        </div>

        {/* Cultural context */}
        {result.cultural_context.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <CulturalContext items={result.cultural_context} />
          </motion.div>
        )}

        {/* Primary gap callout */}
        {result.primary_gap && (
          <motion.div
            className="primary-gap-callout"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            <div className="primary-gap-callout__header">
              <span className={`badge badge--${result.primary_gap.severity}`}>{result.primary_gap.severity} gap</span>
              <span className="primary-gap-callout__title">{result.primary_gap.gap_name}</span>
            </div>
            <p className="primary-gap-callout__desc">{result.primary_gap.what_it_requires}</p>
            {result.primary_gap.actionable_step && (
              <p className="primary-gap-callout__action">→ {result.primary_gap.actionable_step}</p>
            )}
          </motion.div>
        )}

      </div>

      <style>{`
        .analysis-page { min-height: 100vh; padding-bottom: var(--space-9); }
        .dna-header { padding: var(--space-7) 0 var(--space-6); }
        .dna-title { font-size: clamp(2rem, 5vw, 3.5rem); margin: var(--space-3) 0; }
        .dna-separator { color: var(--color-accent); }
        .dna-tension { max-width: 680px; margin: var(--space-3) 0; }
        .dna-aspiration { max-width: 680px; font-size: 1rem; margin-top: var(--space-3); color: var(--color-text-secondary); }
        .dna-meta { display: flex; align-items: center; gap: var(--space-3); margin-top: var(--space-5); flex-wrap: wrap; }
        .map-hint { text-align: center; font-size: 0.8125rem; color: var(--color-text-muted); margin-top: var(--space-3); margin-bottom: var(--space-6); }
        .insight-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: var(--space-4); margin-bottom: var(--space-7); }
        .primary-gap-callout {
          margin-top: var(--space-7);
          padding: var(--space-5);
          background: linear-gradient(135deg, var(--color-surface) 0%, rgba(200,92,92,0.05) 100%);
          border: 1px solid rgba(224,92,92,0.2);
          border-radius: var(--radius-lg);
        }
        .primary-gap-callout__header { display: flex; align-items: center; gap: var(--space-3); margin-bottom: var(--space-3); }
        .primary-gap-callout__title { font-size: 1rem; font-weight: 600; color: var(--color-text-primary); }
        .primary-gap-callout__desc { font-size: 0.9375rem; color: var(--color-text-secondary); margin-bottom: var(--space-3); }
        .primary-gap-callout__action { font-size: 0.9375rem; color: var(--color-accent); font-weight: 500; }
      `}</style>
    </div>
  )
}
