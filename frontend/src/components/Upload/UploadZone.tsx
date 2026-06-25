import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, X, Sparkles } from 'lucide-react'
import { type PresetBoard } from '../../lib/api'

interface UploadZoneProps {
  presets: PresetBoard[]
  onAnalyse: (
    source: 'upload' | 'preset' | 'unsplash',
    options: { files?: File[]; board?: string; query?: string }
  ) => void
  disabled?: boolean
}

export default function UploadZone({ presets, onAnalyse, disabled }: UploadZoneProps) {
  const [files, setFiles] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [mode, setMode] = useState<'upload' | 'preset'>('preset')
  const [selectedPreset, setSelectedPreset] = useState<string | null>(presets[0]?.name ?? null)

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
      onAnalyse('preset', { board: selectedPreset })
    } else if (mode === 'upload' && files.length >= 5) {
      onAnalyse('upload', { files })
    }
  }

  return (
    <div className="upload-zone">
      {/* Mode toggle */}
      <div className="upload-zone__tabs">
        {(['preset', 'upload'] as const).map(m => (
          <button
            key={m}
            className={`upload-zone__tab ${mode === m ? 'upload-zone__tab--active' : ''}`}
            onClick={() => setMode(m)}
            disabled={disabled}
          >
            {m === 'preset' ? 'Use a preset board' : 'Upload your images'}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {mode === 'preset' ? (
          <motion.div
            key="preset"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="upload-zone__presets"
          >
            {presets.map(board => (
              <motion.button
                key={board.name}
                className={`preset-card ${selectedPreset === board.name ? 'preset-card--selected' : ''}`}
                onClick={() => setSelectedPreset(board.name)}
                whileHover={{ scale: 1.02 }}
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
            transition={{ duration: 0.2 }}
          >
            <div
              {...getRootProps()}
              className={`dropzone ${isDragActive ? 'dropzone--active' : ''} ${disabled ? 'dropzone--disabled' : ''}`}
            >
              <input {...getInputProps()} />
              <Upload size={32} strokeWidth={1.5} />
              <p className="dropzone__title">
                {isDragActive ? 'Drop your images here' : 'Drag images here, or click to select'}
              </p>
              <p className="dropzone__hint">
                {files.length > 0 ? `${files.length} selected — add more or analyse` : 'Minimum 5 images. JPG, PNG, WebP accepted.'}
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
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* CTA */}
      <motion.button
        className={`btn btn--primary btn--large analyse-btn ${!canAnalyse ? 'analyse-btn--disabled' : ''}`}
        onClick={handleAnalyse}
        disabled={!canAnalyse || disabled}
        whileHover={canAnalyse && !disabled ? { scale: 1.02 } : {}}
        whileTap={canAnalyse && !disabled ? { scale: 0.98 } : {}}
      >
        <Sparkles size={18} />
        Analyse my aesthetic
      </motion.button>

      {mode === 'upload' && files.length < 5 && files.length > 0 && (
        <p className="upload-zone__hint-text">Add {5 - files.length} more image{5 - files.length !== 1 ? 's' : ''} to continue</p>
      )}

      <style>{`
        .upload-zone { display: flex; flex-direction: column; gap: var(--space-5); }
        .upload-zone__tabs { display: flex; gap: var(--space-2); padding: 4px; background: var(--color-surface-2); border-radius: var(--radius-md); }
        .upload-zone__tab { flex: 1; padding: var(--space-3) var(--space-4); border-radius: 6px; border: none; background: transparent; color: var(--color-text-muted); font-family: var(--font-body); font-size: 0.875rem; cursor: pointer; transition: all var(--duration-base); }
        .upload-zone__tab--active { background: var(--color-surface-3); color: var(--color-text-primary); }
        .upload-zone__presets { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: var(--space-3); }
        .preset-card { position: relative; border: 1px solid var(--color-border); border-radius: var(--radius-md); overflow: hidden; cursor: pointer; background: var(--color-surface); transition: all var(--duration-base); }
        .preset-card--selected { border-color: var(--color-accent); box-shadow: 0 0 0 1px var(--color-accent); }
        .preset-card__images { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; aspect-ratio: 1; }
        .preset-card__images img { width: 100%; height: 100%; object-fit: cover; }
        .preset-card__info { padding: var(--space-2) var(--space-3); display: flex; flex-direction: column; gap: 2px; }
        .preset-card__name { font-size: 0.875rem; font-weight: 500; color: var(--color-text-primary); }
        .preset-card__count { font-size: 0.75rem; color: var(--color-text-muted); }
        .preset-card__check { position: absolute; top: var(--space-2); right: var(--space-2); background: var(--color-accent); color: var(--color-bg); width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; }
        .dropzone { border: 1.5px dashed var(--color-border); border-radius: var(--radius-lg); padding: var(--space-8) var(--space-5); display: flex; flex-direction: column; align-items: center; gap: var(--space-3); cursor: pointer; transition: all var(--duration-base); color: var(--color-text-muted); }
        .dropzone:hover, .dropzone--active { border-color: var(--color-accent); color: var(--color-accent); background: var(--color-accent-glow); }
        .dropzone--disabled { opacity: 0.5; cursor: not-allowed; }
        .dropzone__title { font-size: 1rem; font-weight: 500; color: var(--color-text-secondary); }
        .dropzone__hint { font-size: 0.8125rem; color: var(--color-text-muted); }
        .upload-previews { display: grid; grid-template-columns: repeat(auto-fill, minmax(72px, 1fr)); gap: var(--space-2); margin-top: var(--space-3); }
        .upload-preview { position: relative; aspect-ratio: 1; border-radius: var(--radius-sm); overflow: hidden; }
        .upload-preview img { width: 100%; height: 100%; object-fit: cover; }
        .upload-preview__remove { position: absolute; top: 2px; right: 2px; background: rgba(0,0,0,0.75); border: none; color: white; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; }
        .analyse-btn { width: 100%; justify-content: center; }
        .analyse-btn--disabled { opacity: 0.4; cursor: not-allowed; }
        .upload-zone__hint-text { text-align: center; font-size: 0.8125rem; color: var(--color-text-muted); }
      `}</style>
    </div>
  )
}
