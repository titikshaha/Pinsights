import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, X, Sparkles, Ruler, ShoppingBag, TrendingUp } from 'lucide-react'
import { type PresetBoard } from '../../lib/api'

interface UploadZoneProps {
  presets: PresetBoard[]
  onAnalyse: (
    source: 'upload' | 'preset' | 'unsplash',
    options: { files?: File[]; board?: string; query?: string; goal?: string }
  ) => void
  disabled?: boolean
}

const GOALS = [
  { id: 'styling' as const, label: 'Styling Rules', icon: Ruler, desc: 'How to wear & combine' },
  { id: 'shopping' as const, label: 'Shopping Guide', icon: ShoppingBag, desc: 'What to buy next' },
  { id: 'trends' as const, label: 'Trend Forecast', icon: TrendingUp, desc: 'Where this is heading' },
]

export default function UploadZone({ presets, onAnalyse, disabled }: UploadZoneProps) {
  const [files, setFiles] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [mode, setMode] = useState<'upload' | 'preset'>('preset')
  const [selectedPreset, setSelectedPreset] = useState<string | null>(presets[0]?.name ?? null)
  const [goal, setGoal] = useState<'styling' | 'shopping' | 'trends'>('styling')

  const onDrop = useCallback((accepted: File[]) => {
    setFiles(prev => [...prev, ...accepted].slice(0, 50))
    accepted.forEach(f => {
      const url = URL.createObjectURL(f)
      setPreviews(prev => [...prev, url].slice(0, 50))
    })
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    multiple: true,
    disabled,
  })

  const removeFile = (i: number) => {
    URL.revokeObjectURL(previews[i])
    setFiles(f => f.filter((_, idx) => idx !== i))
    setPreviews(p => p.filter((_, idx) => idx !== i))
  }

  const canAnalyse = mode === 'preset' ? !!selectedPreset : files.length >= 5

  const handleAnalyse = () => {
    if (mode === 'preset' && selectedPreset) {
      onAnalyse('preset', { board: selectedPreset, goal })
    } else if (mode === 'upload' && files.length >= 5) {
      onAnalyse('upload', { files, goal })
    }
  }

  return (
    <div className="upload-zone">
      {/* Mode tabs */}
      <div className="upload-zone__tabs">
        {(['preset', 'upload'] as const).map(m => (
          <button
            key={m}
            className={`uz-tab ${mode === m ? 'uz-tab--active' : ''}`}
            onClick={() => setMode(m)}
            disabled={disabled}
          >
            {m === 'preset' ? 'Preset board' : 'Upload images'}
          </button>
        ))}
      </div>

      {/* Content panel */}
      <AnimatePresence mode="wait">
        {mode === 'preset' ? (
          <motion.div
            key="preset"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.18 }}
            className="preset-grid"
          >
            {presets.map(board => (
              <motion.button
                key={board.name}
                className={`preset-card ${selectedPreset === board.name ? 'preset-card--selected' : ''}`}
                onClick={() => setSelectedPreset(board.name)}
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                disabled={disabled}
              >
                <div className="preset-card__images">
                  {board.thumbnail_paths.slice(0, 4).map((p, i) => (
                    <img key={i} src={p} alt="" />
                  ))}
                </div>
                <div className="preset-card__info">
                  <span className="preset-card__name">{board.display_name}</span>
                  <span className="preset-card__count">{board.image_count} images</span>
                </div>
                {selectedPreset === board.name && (
                  <div className="preset-card__check">✓</div>
                )}
              </motion.button>
            ))}
          </motion.div>
        ) : (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.18 }}
          >
            <div
              {...getRootProps()}
              className={`dropzone ${isDragActive ? 'dropzone--active' : ''} ${disabled ? 'dropzone--disabled' : ''}`}
            >
              <input {...getInputProps()} />
              <div className="dropzone__icon">
                <Upload size={24} strokeWidth={1.5} />
              </div>
              <p className="dropzone__title">
                {isDragActive ? 'Drop your images here' : 'Drag images here, or click to browse'}
              </p>
              <p className="dropzone__hint">
                {files.length > 0
                  ? `${files.length} selected — add more or analyse`
                  : 'Minimum 5 images · JPG, PNG, WebP'}
              </p>
            </div>

            {previews.length > 0 && (
              <div className="upload-previews">
                {previews.map((src, i) => (
                  <div key={i} className="upload-preview">
                    <img src={src} alt="" />
                    <button
                      className="upload-preview__remove"
                      onClick={() => removeFile(i)}
                      disabled={disabled}
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {mode === 'upload' && files.length < 5 && files.length > 0 && (
              <p className="upload-zone__hint-text">
                Add {5 - files.length} more image{5 - files.length !== 1 ? 's' : ''} to continue
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Goal selection */}
      <div className="goal-section">
        <div className="goal-section__label">What do you want to infer?</div>
        <div className="goal-section__row">
          {GOALS.map(g => {
            const Icon = g.icon
            return (
              <button
                key={g.id}
                className={`goal-btn ${goal === g.id ? 'goal-btn--active' : ''}`}
                onClick={() => setGoal(g.id)}
                disabled={disabled}
              >
                <Icon size={14} strokeWidth={1.75} />
                <div className="goal-btn__text">
                  <span className="goal-btn__label">{g.label}</span>
                  <span className="goal-btn__desc">{g.desc}</span>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* CTA */}
      <motion.button
        className={`btn btn--primary btn--large analyse-btn ${!canAnalyse ? 'analyse-btn--disabled' : ''}`}
        onClick={handleAnalyse}
        disabled={!canAnalyse || disabled}
        whileHover={canAnalyse && !disabled ? { scale: 1.015 } : {}}
        whileTap={canAnalyse && !disabled ? { scale: 0.985 } : {}}
      >
        <Sparkles size={16} />
        Analyse my aesthetic
      </motion.button>

      <style>{`
        .upload-zone { display: flex; flex-direction: column; gap: var(--space-5); }

        /* Mode tabs */
        .upload-zone__tabs {
          display: flex;
          gap: 2px;
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          padding: 3px;
        }
        .uz-tab {
          flex: 1;
          padding: var(--space-2) var(--space-4);
          border-radius: 6px;
          border: none;
          background: transparent;
          color: var(--color-text-muted);
          font-family: var(--font-body);
          font-size: 0.8125rem;
          font-weight: 500;
          cursor: pointer;
          transition: all var(--duration-base);
        }
        .uz-tab--active {
          background: var(--color-surface);
          color: var(--color-text-primary);
          box-shadow: var(--shadow-sm);
        }
        .uz-tab:hover:not(.uz-tab--active):not(:disabled) {
          color: var(--color-text-secondary);
        }

        /* Preset grid */
        .preset-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
          gap: var(--space-3);
        }
        .preset-card {
          position: relative;
          border: 1.5px solid var(--color-border);
          border-radius: var(--radius-md);
          overflow: hidden;
          cursor: pointer;
          background: var(--color-surface);
          transition: all var(--duration-base);
          box-shadow: var(--shadow-sm);
          text-align: left;
        }
        .preset-card:hover { box-shadow: var(--shadow-md); border-color: var(--color-border-hover); }
        .preset-card--selected {
          border-color: var(--color-accent);
          box-shadow: 0 0 0 2px rgba(156,123,90,0.18);
        }
        .preset-card__images {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1px;
          aspect-ratio: 1;
          background: var(--color-surface-3);
        }
        .preset-card__images img { width: 100%; height: 100%; object-fit: cover; }
        .preset-card__info {
          padding: var(--space-2) var(--space-3);
          display: flex;
          flex-direction: column;
          gap: 1px;
        }
        .preset-card__name { font-size: 0.8125rem; font-weight: 500; color: var(--color-text-primary); }
        .preset-card__count { font-size: 0.6875rem; color: var(--color-text-muted); }
        .preset-card__check {
          position: absolute;
          top: var(--space-2);
          right: var(--space-2);
          background: var(--color-accent);
          color: #fff;
          width: 18px; height: 18px;
          border-radius: 50%;
          display: flex; align-items: center; justify-content: center;
          font-size: 0.625rem;
          font-weight: 700;
        }

        /* Dropzone */
        .dropzone {
          border: 1.5px dashed var(--color-border);
          border-radius: var(--radius-lg);
          padding: var(--space-8) var(--space-5);
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: var(--space-3);
          cursor: pointer;
          transition: all var(--duration-base);
          text-align: center;
        }
        .dropzone:hover, .dropzone--active {
          border-color: var(--color-accent);
          background: var(--color-accent-glow);
        }
        .dropzone--disabled { opacity: 0.45; cursor: not-allowed; }
        .dropzone__icon {
          width: 48px; height: 48px;
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          display: flex; align-items: center; justify-content: center;
          color: var(--color-text-muted);
        }
        .dropzone__title { font-size: 0.9375rem; font-weight: 500; color: var(--color-text-secondary); }
        .dropzone__hint { font-size: 0.8125rem; color: var(--color-text-muted); }

        /* Previews */
        .upload-previews {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(64px, 1fr));
          gap: var(--space-2);
          margin-top: var(--space-3);
        }
        .upload-preview { position: relative; aspect-ratio: 1; border-radius: var(--radius-sm); overflow: hidden; }
        .upload-preview img { width: 100%; height: 100%; object-fit: cover; }
        .upload-preview__remove {
          position: absolute; top: 2px; right: 2px;
          background: rgba(28,25,19,0.65);
          border: none; color: #fff;
          width: 16px; height: 16px;
          border-radius: 50%; cursor: pointer;
          display: flex; align-items: center; justify-content: center;
        }

        /* Goal section */
        .goal-section {
          padding-top: var(--space-4);
          border-top: 1px solid var(--color-border);
        }
        .goal-section__label {
          font-size: 0.6875rem;
          font-weight: 600;
          letter-spacing: 0.10em;
          text-transform: uppercase;
          color: var(--color-text-muted);
          margin-bottom: var(--space-3);
        }
        .goal-section__row { display: flex; gap: var(--space-2); flex-wrap: wrap; }
        .goal-btn {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-2) var(--space-3);
          border: 1.5px solid var(--color-border);
          background: var(--color-surface);
          border-radius: var(--radius-md);
          font-family: var(--font-body);
          font-size: 0.8125rem;
          cursor: pointer;
          transition: all var(--duration-base);
          color: var(--color-text-secondary);
          flex: 1;
          min-width: 120px;
          box-shadow: var(--shadow-sm);
        }
        .goal-btn:hover:not(:disabled) {
          border-color: var(--color-border-hover);
          background: var(--color-surface-2);
          color: var(--color-text-primary);
        }
        .goal-btn--active {
          border-color: var(--color-accent);
          background: var(--color-accent-dim);
          color: var(--color-text-primary);
        }
        .goal-btn__text { display: flex; flex-direction: column; gap: 1px; text-align: left; }
        .goal-btn__label { font-weight: 600; font-size: 0.8125rem; color: inherit; }
        .goal-btn__desc { font-size: 0.6875rem; color: var(--color-text-muted); }

        /* CTA */
        .analyse-btn { width: 100%; justify-content: center; letter-spacing: 0.06em; }
        .analyse-btn--disabled { opacity: 0.35; cursor: not-allowed; }
        .upload-zone__hint-text { text-align: center; font-size: 0.8125rem; color: var(--color-text-muted); }
      `}</style>
    </div>
  )
}
