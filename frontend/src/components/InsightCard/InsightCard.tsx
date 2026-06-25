import { motion, AnimatePresence } from 'framer-motion'
import { ClusterSummary, Gap } from '../../lib/api'
import { imageUrl } from '../../lib/api'
import { ChevronDown, ChevronUp, ArrowRight } from 'lucide-react'
import { useState } from 'react'

interface InsightCardProps {
  cluster: ClusterSummary
  isSelected: boolean
  onClick: () => void
}

function GapItem({ gap }: { gap: Gap }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="gap-item">
      <div className="gap-item__header" onClick={() => setOpen(o => !o)}>
        <div className="gap-item__title-row">
          <span className={`badge badge--${gap.severity}`}>{gap.severity}</span>
          <span className="gap-item__name">{gap.gap_name}</span>
        </div>
        <button className="gap-item__toggle">
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="gap-item__body"
          >
            <div className="gap-detail">
              <div className="section-label">What it requires</div>
              <p>{gap.what_it_requires}</p>
            </div>
            <div className="gap-detail">
              <div className="section-label">The common miss</div>
              <p>{gap.common_miss}</p>
            </div>
            {gap.actionable_step && (
              <div className="gap-action">
                <ArrowRight size={14} />
                <span>{gap.actionable_step}</span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        .gap-item { border: 1px solid var(--color-border); border-radius: var(--radius-md); overflow: hidden; }
        .gap-item__header { display: flex; justify-content: space-between; align-items: center; padding: var(--space-3) var(--space-4); cursor: pointer; gap: var(--space-3); }
        .gap-item__header:hover { background: var(--color-surface-2); }
        .gap-item__title-row { display: flex; align-items: center; gap: var(--space-2); }
        .gap-item__name { font-size: 0.875rem; font-weight: 500; color: var(--color-text-primary); }
        .gap-item__toggle { background: none; border: none; color: var(--color-text-muted); cursor: pointer; }
        .gap-item__body { overflow: hidden; }
        .gap-detail { padding: var(--space-3) var(--space-4); border-top: 1px solid var(--color-border); }
        .gap-detail p { font-size: 0.875rem; }
        .gap-action { display: flex; align-items: flex-start; gap: var(--space-2); padding: var(--space-3) var(--space-4); border-top: 1px solid var(--color-border); color: var(--color-accent); font-size: 0.875rem; font-weight: 500; }
      `}</style>
    </div>
  )
}

export default function InsightCard({ cluster, isSelected, onClick }: InsightCardProps) {
  return (
    <motion.div
      className={`insight-card ${isSelected ? 'insight-card--selected' : ''}`}
      onClick={onClick}
      layout
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Representative image grid */}
      <div className="insight-card__images">
        {cluster.representative_paths.slice(0, 3).map((path, i) => (
          <img key={i} src={imageUrl(path)} alt="" />
        ))}
      </div>

      {/* Palette */}
      <div className="insight-card__palette">
        {cluster.dominant_palette.slice(0, 6).map((hex, i) => (
          <div key={i} className="insight-card__swatch" style={{ background: hex }} title={hex} />
        ))}
      </div>

      {/* Aesthetic name */}
      <h3 className="insight-card__name">{cluster.aesthetic_name}</h3>

      {/* Description */}
      {cluster.description && (
        <p className="insight-card__description">{cluster.description}</p>
      )}

      <AnimatePresence>
        {isSelected && (
          <motion.div
            className="insight-card__expanded"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <hr className="divider" />

            {/* Aspiration reading */}
            {cluster.aspiration_reading && (
              <div className="insight-section">
                <div className="section-label">Aspiration reading</div>
                <p>{cluster.aspiration_reading}</p>
              </div>
            )}

            {/* Palette story */}
            {cluster.palette_story && (
              <div className="insight-section">
                <div className="section-label">What your palette says</div>
                <p>{cluster.palette_story}</p>
              </div>
            )}

            {/* Cultural origin */}
            {cluster.cultural_origin && (
              <div className="because">{cluster.cultural_origin}</div>
            )}

            {/* Visual signals */}
            {cluster.visual_signals.length > 0 && (
              <div className="insight-section">
                <div className="section-label">Visual signals detected</div>
                <div className="insight-tags">
                  {cluster.visual_signals.map((s, i) => (
                    <span key={i} className="insight-tag">{s}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Gaps */}
            {cluster.gaps.length > 0 && (
              <div className="insight-section">
                <div className="section-label">Execution gaps</div>
                <div className="insight-gaps">
                  {cluster.gaps.map((g, i) => (
                    <GapItem key={i} gap={g} />
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        .insight-card {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          overflow: hidden;
          cursor: pointer;
          transition: border-color var(--duration-base), box-shadow var(--duration-base);
        }
        .insight-card:hover {
          border-color: var(--color-border-hover);
        }
        .insight-card--selected {
          border-color: rgba(200,168,130,0.4);
          box-shadow: 0 0 0 1px rgba(200,168,130,0.2);
        }
        .insight-card__images {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 2px;
          background: var(--color-surface-3);
        }
        .insight-card__images img {
          width: 100%;
          aspect-ratio: 1;
          object-fit: cover;
          display: block;
        }
        .insight-card__palette {
          display: flex;
          gap: var(--space-1);
          padding: var(--space-3) var(--space-4) 0;
        }
        .insight-card__swatch {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          flex-shrink: 0;
          box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
        }
        .insight-card__name {
          font-family: var(--font-display);
          font-size: 1.25rem;
          padding: var(--space-2) var(--space-4) 0;
        }
        .insight-card__description {
          font-size: 0.875rem;
          padding: var(--space-2) var(--space-4) var(--space-4);
          color: var(--color-text-secondary);
        }
        .insight-card__expanded { padding: 0 var(--space-4) var(--space-4); overflow: hidden; }
        .insight-section { margin-bottom: var(--space-4); }
        .insight-section:last-child { margin-bottom: 0; }
        .insight-section p { font-size: 0.875rem; }
        .insight-tags { display: flex; flex-wrap: wrap; gap: var(--space-2); margin-top: var(--space-2); }
        .insight-tag { background: var(--color-surface-3); color: var(--color-text-secondary); padding: 2px var(--space-3); border-radius: 100px; font-size: 0.75rem; }
        .insight-gaps { display: flex; flex-direction: column; gap: var(--space-2); margin-top: var(--space-2); }
      `}</style>
    </motion.div>
  )
}
