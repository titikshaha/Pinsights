import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Sparkles } from 'lucide-react'
import { PresetBoard, fetchPresets } from '../lib/api'
import UploadZone from '../components/Upload/UploadZone'

interface LandingProps {
  onAnalysisStarted: () => void
  onAnalyse: (source: 'upload' | 'preset' | 'unsplash', options: any) => void
  isRunning: boolean
}

export default function Landing({ onAnalysisStarted, onAnalyse, isRunning }: LandingProps) {
  const [presets, setPresets] = useState<PresetBoard[]>([])

  useEffect(() => {
    fetchPresets().then(setPresets)
  }, [])

  const handleAnalyse = async (
    source: 'upload' | 'preset' | 'unsplash',
    options: any
  ) => {
    onAnalyse(source, options)
    onAnalysisStarted()
  }

  return (
    <div className="landing">
      {/* Hero */}
      <section className="landing__hero container">
        <motion.div
          className="landing__eyebrow"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Sparkles size={14} />
          Personal Aesthetic Intelligence
        </motion.div>

        <motion.h1
          className="landing__title"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
        >
          Not what you like.
          <br />
          <em>Who you are.</em>
        </motion.h1>

        <motion.p
          className="landing__subtitle"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
        >
          Pinsights analyses your saved fashion images and tells you your aesthetic DNA,
          what you're reaching toward, and exactly what's standing between you and that vision.
          Grounded in fashion history. Specific, not generic.
        </motion.p>

        <motion.div
          className="landing__features"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
        >
          {[
            { label: 'Aesthetic DNA', desc: 'Your visual worlds, named and interpreted' },
            { label: 'Gap Analysis', desc: 'The specific delta between aspiration and execution' },
            { label: 'Cultural Context', desc: 'Every insight grounded in fashion history' },
            { label: 'Drift Tracking', desc: 'Watch your aesthetic evolve over time' },
          ].map(f => (
            <div key={f.label} className="feature-pill">
              <span className="feature-pill__label">{f.label}</span>
              <span className="feature-pill__desc">{f.desc}</span>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Upload section */}
      <section className="landing__upload container">
        <motion.div
          className="upload-section"
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
        >
          <div className="upload-section__header">
            <h2>Start your analysis</h2>
            <p>Load a preset board or upload your own images — 10 seconds to insight.</p>
          </div>
          <UploadZone
            presets={presets}
            onAnalyse={handleAnalyse}
            disabled={isRunning}
          />
        </motion.div>
      </section>

      <style>{`
        .landing { padding-top: 80px; }
        .landing__hero { padding-top: var(--space-9); padding-bottom: var(--space-7); }
        .landing__eyebrow {
          display: inline-flex; align-items: center; gap: var(--space-2);
          font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
          color: var(--color-accent); margin-bottom: var(--space-5);
          padding: var(--space-2) var(--space-3);
          background: var(--color-accent-dim); border-radius: 100px;
        }
        .landing__title {
          font-size: clamp(3rem, 7vw, 5.5rem);
          line-height: 1.05;
          letter-spacing: -0.02em;
          margin-bottom: var(--space-5);
        }
        .landing__title em { color: var(--color-accent); font-style: italic; }
        .landing__subtitle {
          font-size: clamp(1rem, 1.5vw, 1.125rem);
          max-width: 600px;
          line-height: 1.7;
          margin-bottom: var(--space-6);
        }
        .landing__features { display: flex; flex-wrap: wrap; gap: var(--space-3); }
        .feature-pill {
          display: flex; flex-direction: column; gap: 2px;
          padding: var(--space-3) var(--space-4);
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
        }
        .feature-pill__label { font-size: 0.8125rem; font-weight: 600; color: var(--color-text-primary); }
        .feature-pill__desc { font-size: 0.75rem; color: var(--color-text-muted); }
        .landing__upload { padding-bottom: var(--space-9); }
        .upload-section {
          max-width: 680px;
          margin: 0 auto;
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-xl);
          padding: var(--space-7);
        }
        .upload-section__header { margin-bottom: var(--space-6); }
        .upload-section__header h2 { font-size: 1.75rem; margin-bottom: var(--space-2); }
        .upload-section__header p { font-size: 0.9375rem; }
        @media (max-width: 600px) { .upload-section { padding: var(--space-5); } }
      `}</style>
    </div>
  )
}
