import { useState } from 'react'
import { motion } from 'framer-motion'
import { type AnalysisState } from '../hooks/useAnalysis'
import ClusterMap from '../components/ClusterMap/ClusterMap'
import InsightCard from '../components/InsightCard/InsightCard'
import { type CulturalContextItem } from '../lib/api'
import { RotateCcw } from 'lucide-react'

interface AnalysisPageProps {
  state: AnalysisState
  onReset: () => void
}

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
      <style>{`
        .progress-view {
          display: flex; align-items: center; justify-content: center;
          min-height: 65vh;
        }
        .progress-view__inner {
          display: flex; flex-direction: column; align-items: center;
          gap: var(--space-6);
        }
        .progress-view__spinner {
          width: 52px; height: 52px;
          border-radius: 50%;
          border: 1.5px solid var(--color-border);
          border-top-color: var(--color-accent);
        }
        .progress-view__text { display: flex; flex-direction: column; align-items: center; gap: var(--space-3); }
        .progress-view__stage { color: var(--color-text-secondary); font-size: 0.9rem; }
        .progress-view__pct { color: var(--color-text-muted); font-size: 0.8125rem; font-variant-numeric: tabular-nums; }
      `}</style>
    </div>
  )
}

function CulturalContext({ items }: { items: CulturalContextItem[] }) {
  return (
    <div className="cultural-context">
      <div className="section-label">Cultural Context & Insights</div>
      <div className="cultural-items">
        {items.map((item, i) => (
          <motion.div
            key={i}
            className="cultural-item"
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08 }}
          >
            {item.source_era && (
              <div className="cultural-item__era">{item.source_era}</div>
            )}
            <p className="cultural-item__claim">{item.claim}</p>
            <div className="because">{item.because}</div>

            {item.detailed_analysis && (
              <div className="cultural-item__analysis">
                {item.detailed_analysis}
              </div>
            )}

            {item.execution_suggestions && item.execution_suggestions.length > 0 && (
              <div className="cultural-item__suggestions">
                <div className="cultural-item__suggestions-label">Execution suggestions</div>
                <ul>
                  {item.execution_suggestions.map((s, j) => (
                    <li key={j}>{s}</li>
                  ))}
                </ul>
              </div>
            )}

            {item.cultural_code && (
              <div className="cultural-item__code">
                <span>{item.cultural_code}</span>
              </div>
            )}
          </motion.div>
        ))}
      </div>
      <style>{`
        .cultural-context { margin-top: var(--space-7); }
        .cultural-items { display: flex; flex-direction: column; gap: var(--space-4); margin-top: var(--space-4); }
        .cultural-item {
          padding: var(--space-5);
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          box-shadow: var(--shadow-sm);
        }
        .cultural-item__era {
          font-size: 0.6875rem;
          font-weight: 600;
          letter-spacing: 0.10em;
          text-transform: uppercase;
          color: var(--color-accent);
          margin-bottom: var(--space-2);
        }
        .cultural-item__claim {
          font-size: 1rem;
          font-weight: 500;
          color: var(--color-text-primary);
          line-height: 1.5;
          margin-bottom: var(--space-2);
        }
        .cultural-item__analysis {
          margin: var(--space-3) 0;
          font-size: 0.9rem;
          line-height: 1.7;
          color: var(--color-text-secondary);
          border-left: 2px solid var(--color-border);
          padding-left: var(--space-3);
        }
        .cultural-item__suggestions { margin-top: var(--space-4); }
        .cultural-item__suggestions-label {
          font-size: 0.6875rem;
          font-weight: 600;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--color-text-muted);
          margin-bottom: var(--space-2);
        }
        .cultural-item__suggestions ul {
          padding-left: var(--space-5);
          color: var(--color-text-secondary);
          display: flex;
          flex-direction: column;
          gap: var(--space-1);
        }
        .cultural-item__suggestions li { font-size: 0.875rem; line-height: 1.55; }
        .cultural-item__code {
          margin-top: var(--space-3);
          padding-top: var(--space-3);
          border-top: 1px solid var(--color-border);
        }
        .cultural-item__code span {
          font-size: 0.75rem;
          color: var(--color-text-muted);
          font-style: italic;
        }
      `}</style>
    </div>
  )
}

export default function Analysis({ state, onReset }: AnalysisPageProps) {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  const result = state.result

  if (state.status === 'running') {
    return (
      <div className="container" style={{ paddingTop: 60 }}>
        <ProgressView state={state} />
      </div>
    )
  }

  if (state.status === 'error') {
    return (
      <div className="container" style={{ paddingTop: 60 }}>
        <div className="analysis-error">
          <div className="analysis-error__icon">!</div>
          <h3>Analysis failed</h3>
          <p>{state.error}</p>
          <button className="btn btn--ghost" onClick={onReset}>Try again</button>
        </div>
        <style>{`
          .analysis-error {
            display: flex; flex-direction: column; align-items: center;
            gap: var(--space-4); text-align: center; padding-top: 120px;
          }
          .analysis-error__icon {
            width: 48px; height: 48px; border-radius: 50%;
            background: rgba(192,57,43,0.08);
            border: 1px solid rgba(192,57,43,0.2);
            color: var(--color-critical);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.25rem; font-weight: 700;
          }
        `}</style>
      </div>
    )
  }

  if (!result) return null

  const dna = result.aesthetic_dna

  return (
    <div className="analysis-page" style={{ paddingTop: 60 }}>
      <div className="container">

        {/* DNA Header */}
        <motion.div
          className="dna-header"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="section-label">Your aesthetic DNA</div>
          <h1 className="dna-title">
            {dna.primary_world}
            {dna.secondary_world && (
              <> <span className="dna-separator">&</span> {dna.secondary_world}</>
            )}
          </h1>
          {dna.visual_tension && (
            <div className="because dna-tension">{dna.visual_tension}</div>
          )}
          {dna.overall_aspiration && (
            <p className="dna-aspiration">{dna.overall_aspiration}</p>
          )}
          <div className="dna-meta">
            <span className="badge badge--neutral">{result.meta.total_images} images</span>
            <span className="badge badge--neutral">{result.clusters.length} aesthetic worlds</span>
            <button
              className="btn btn--ghost dna-reset-btn"
              onClick={onReset}
            >
              <RotateCcw size={13} />
              New analysis
            </button>
          </div>
        </motion.div>

        {/* Cluster map */}
        <motion.div
          className="cluster-section"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15 }}
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
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 + i * 0.08 }}
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

        {/* Primary gap callout */}
        {result.primary_gap && (
          <motion.div
            className="primary-gap"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <div className="primary-gap__header">
              <span className={`badge badge--${result.primary_gap.severity}`}>
                {result.primary_gap.severity} gap
              </span>
              <span className="primary-gap__title">{result.primary_gap.gap_name}</span>
            </div>
            <p className="primary-gap__desc">{result.primary_gap.what_it_requires}</p>
            {result.primary_gap.actionable_step && (
              <p className="primary-gap__action">→ {result.primary_gap.actionable_step}</p>
            )}
          </motion.div>
        )}

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

      </div>

      <style>{`
        .analysis-page { min-height: 100vh; padding-bottom: var(--space-9); }

        .dna-header { padding: var(--space-8) 0 var(--space-6); border-bottom: 1px solid var(--color-border); margin-bottom: var(--space-7); }
        .dna-title {
          font-size: clamp(2rem, 5vw, 3.25rem);
          line-height: 1.08;
          letter-spacing: -0.02em;
          margin: var(--space-3) 0;
        }
        .dna-separator { color: var(--color-accent); }
        .dna-tension { max-width: 640px; margin: var(--space-3) 0; }
        .dna-aspiration { max-width: 640px; font-size: 1rem; margin-top: var(--space-3); color: var(--color-text-secondary); }
        .dna-meta { display: flex; align-items: center; gap: var(--space-3); margin-top: var(--space-5); flex-wrap: wrap; }
        .dna-reset-btn { margin-left: auto; font-size: 0.8125rem; }

        .cluster-section { margin-bottom: var(--space-4); }
        .map-hint { text-align: center; font-size: 0.8125rem; color: var(--color-text-muted); margin-top: var(--space-3); margin-bottom: var(--space-6); }

        .insight-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: var(--space-4); margin-bottom: var(--space-7); }

        .primary-gap {
          padding: var(--space-5) var(--space-6);
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          box-shadow: var(--shadow-sm);
          margin-top: var(--space-6);
        }
        .primary-gap__header { display: flex; align-items: center; gap: var(--space-3); margin-bottom: var(--space-3); }
        .primary-gap__title { font-size: 0.9375rem; font-weight: 600; color: var(--color-text-primary); }
        .primary-gap__desc { font-size: 0.9rem; color: var(--color-text-secondary); margin-bottom: var(--space-3); }
        .primary-gap__action { font-size: 0.875rem; color: var(--color-accent); font-weight: 500; }
      `}</style>
    </div>
  )
}
