import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { type PresetBoard, fetchPresets } from '../lib/api'
import UploadZone from '../components/Upload/UploadZone'

interface LandingProps {
  onAnalysisStarted: () => void
  onAnalyse: (source: 'upload' | 'preset' | 'unsplash', options: { files?: File[]; board?: string; query?: string; goal?: string }) => void
  isRunning: boolean
}

// Ticker items
const TICKER_ITEMS = [
  'Aesthetic DNA',
  'Gap Analysis',
  'Cultural Context',
  'Styling Rules',
  'Shopping Guide',
  'Trend Forecast',
  'Drift Tracking',
  'Pinterest Intelligence',
]

const HERO_IMAGES = [
  { src: '/f1.png', label: 'Look 01' },
  { src: '/f4.png', label: 'Look 03' },
  { src: '/f3.png', label: 'Look 04' },
]

export default function Landing({ onAnalysisStarted, onAnalyse, isRunning }: LandingProps) {
  const [presets, setPresets] = useState<PresetBoard[]>([])
  useEffect(() => {
    fetchPresets().then(setPresets)
  }, [])

  const handleAnalyse = (
    source: 'upload' | 'preset' | 'unsplash',
    options: { files?: File[]; board?: string; query?: string; goal?: string }
  ) => {
    onAnalyse(source, options)
    onAnalysisStarted()
  }

  return (
    <div className="landing">

      {/* ── Hero ── */}
      <section className="hero">
        {/* Background Large Text */}
        <motion.div 
          className="hero__bg-text"
          initial="hidden"
          animate="visible"
          variants={{
            hidden: { opacity: 1 },
            visible: {
              opacity: 1,
              transition: { staggerChildren: 0.15, delayChildren: 0.3 }
            }
          }}
        >
          {"Pinsights".split('').map((char, i) => (
            <motion.span
              key={i}
              variants={{
                hidden: { opacity: 0 },
                visible: { opacity: 1 }
              }}
            >
              {char}
            </motion.span>
          ))}
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 1, 1, 0] }}
            transition={{ repeat: Infinity, duration: 0.9, times: [0, 0.1, 0.5, 0.6] }}
            style={{ fontWeight: 400, marginLeft: '0.05em', color: 'var(--color-accent)' }}
          >
            |
          </motion.span>
        </motion.div>

        {/* Foreground Model Images */}
        <div className="hero__models">
          {HERO_IMAGES.map((img, i) => (
            <motion.img
              key={img.src}
              src={img.src}
              alt={img.label}
              className={`hero__model hero__model--${i + 1}`}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 + i * 0.1 }}
            />
          ))}
        </div>

        {/* Overlay CTA */}
        <motion.div
          className="hero__overlay"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <a
            href="#analyse"
            className="hero__cta"
            onClick={e => { e.preventDefault(); document.getElementById('analyse')?.scrollIntoView({ behavior: 'smooth' }) }}
          >
            START ANALYSIS
          </a>
        </motion.div>
      </section>

      {/* ── Marquee ticker ── */}
      <div className="ticker">
        <div className="ticker__track">
          {[...TICKER_ITEMS, ...TICKER_ITEMS].map((item, i) => (
            <span key={i} className="ticker__item">
              {item} <span className="ticker__dot">·</span>
            </span>
          ))}
        </div>
      </div>

      {/* ── How it works strip ── */}
      <section className="how-it-works container">
        <div className="how-it-works__label">How it works</div>
        <div className="how-it-works__grid">
          {[
            { n: '01', title: 'Upload', body: 'Drop your Pinterest screenshots or saved images. Minimum 5.' },
            { n: '02', title: 'Choose your goal', body: 'Styling rules, shopping guide, or trend forecast.' },
            { n: '03', title: 'Get your DNA', body: 'Your aesthetic worlds are named, historicised, and decoded.' },
          ].map(step => (
            <div key={step.n} className="how-step">
              <div className="how-step__n">{step.n}</div>
              <div className="how-step__title">{step.title}</div>
              <p className="how-step__body">{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── GIF section ── */}
      <section className="gif-section container">
        <div className="gif-section__inner">
          <div className="gif-section__text">
            <div className="section-label">Analysis preview</div>
            <h2 className="gif-section__heading">From mood board<br /><em>to meaning.</em></h2>
            <p className="gif-section__desc">
              Pinsights groups your images into aesthetic clusters, maps their cultural history, 
              and tells you exactly what you're reaching for, and what's stopping you from getting there.
            </p>
          </div>
          <div className="gif-section__visual">
            <img src="/fashion.gif" alt="Fashion analysis preview" className="gif-section__gif" />
          </div>
        </div>
      </section>

      {/* ── Upload section ── */}
      <section className="analyse-section" id="analyse">
        <div className="analyse-section__header container">
          <div className="section-label">Start now</div>
          <h2 className="analyse-section__heading">Your aesthetic,<em> decoded.</em></h2>
        </div>
        <div className="container analyse-section__body">
          <div className="analyse-section__blueprint">
            <div className="blueprint-frame">
              {/* Corner crosshairs */}
              <div className="crosshair top-left" />
              <div className="crosshair top-right" />
              <div className="crosshair bottom-left" />
              <div className="crosshair bottom-right" />
              
              {/* Frame labels */}
              <div className="frame-label frame-label--top">
                <span>CAM</span><span className="line"></span><span>ANALYZE</span>
              </div>
              <div className="frame-label frame-label--bottom">
                PINSIGHTS
              </div>

              <motion.img 
                src="/f2.png" 
                alt="Model" 
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
              />

              {/* Bounding boxes mimicking ML vision detection */}
              <div className="bounding-box box-1">
                <div className="box-label">LV/01</div>
              </div>
              
              <div className="bounding-box box-2">
                <div className="box-label">A.P./04</div>
              </div>

              <div className="bounding-box box-3">
                <div className="box-label">LV/02</div>
              </div>
            </div>
          </div>
          <div className="upload-card">
            <UploadZone
              presets={presets}
              onAnalyse={handleAnalyse}
              disabled={isRunning}
            />
          </div>
        </div>
      </section>

      <footer className="landing-footer container">
        <span className="landing-footer__logo">Pin<em>sights</em></span>
        <span className="landing-footer__copy">Personal Aesthetic Intelligence · 2026</span>
      </footer>

      <style>{`
        .landing { padding-top: 20px; }

        /* ── Hero ── */
        .hero {
          position: relative;
          width: 100%;
          min-height: calc(100vh - 56px);
          max-height: 850px;
          background: #E8E4DA; /* A slightly darker beige background to contrast text */
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          border-bottom: 1x solid var(--color-border);
        }

        .hero__bg-text {
          position: absolute;
          z-index: 10;
          font-family: var(--font-display);
          font-style: italic;
          font-size: clamp(6rem, 20vw, 22rem);
          color: #F9F8F5; /* white/off-white */
          white-space: nowrap;
          pointer-events: none;
          user-select: none;
          letter-spacing: -0.05em;
          line-height: 1;
        }

        .hero__models {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
        }

        .hero__model {
          position: absolute;
          bottom: 0;
          max-height: 90%;
          object-fit: contain;
        }
        .hero__model--1 { left: -3%; height: 110%; z-index: 1; }
        .hero__model--2 { left: 29%; transform: translateX(-50%); height: 65%; z-index: 15; }
        .hero__model--3 { left: 82.4%; height: 75%; z-index: 15; }

        .hero__overlay {
          position: absolute;
          bottom: var(--space-8);
          z-index: 20;
          display: flex;
          justify-content: center;
          width: 100%;
          pointer-events: auto;
        }

        .hero__cta {
          background: rgba(249, 248, 245, 0.9);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          color: var(--color-text-primary);
          padding: 16px 48px;
          font-size: 0.8125rem;
          font-weight: 600;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          text-decoration: none;
          border-radius: 2px;
          box-shadow: var(--shadow-md);
          transition: all var(--duration-base);
          border: 1px solid var(--color-border);
        }
        .hero__cta:hover {
          background: #fff;
          transform: translateY(-2px);
          box-shadow: var(--shadow-lg);
          opacity: 1;
        }

        /* ── Ticker ── */
        .ticker {
          border-top: 1px solid var(--color-border);
          border-bottom: 1px solid var(--color-border);
          background: var(--color-text-primary);
          overflow: hidden;
          padding: 10px 0;
        }
        .ticker__track {
          display: flex;
          white-space: nowrap;
          animation: marquee 24s linear infinite;
          width: max-content;
        }
        .ticker__item {
          font-size: 0.6875rem;
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: rgba(249,248,245,0.7);
          padding: 0 var(--space-4);
        }
        .ticker__dot { color: var(--color-accent); }

        /* ── How it works ── */
        .how-it-works {
          padding: var(--space-8) 0;
          border-bottom: 1px solid var(--color-border);
        }
        .how-it-works__label {
          font-size: 0.825rem;
          font-weight: 600;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--color-text-muted);
          margin-bottom: var(--space-6);
        }
        .how-it-works__grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 0;
          border: 1px solid var(--color-border);
        }
        .how-step {
          padding: var(--space-6);
          border-right: 1px solid var(--color-border);
        }
        .how-step:last-child { border-right: none; }
        .how-step__n {
          font-size: 0.825rem;
          font-weight: 600;
          letter-spacing: 0.14em;
          color: var(--color-accent);
          margin-bottom: var(--space-3);
        }
        .how-step__title {
          font-family: var(--font-display);
          font-size: 1.55rem;
          font-weight: 600;
          color: var(--color-text-primary);
          margin-bottom: var(--space-3);
          letter-spacing: -0.01em;
        }
        .how-step__body {
          font-size: 0.9125rem;
          line-height: 1.65;
          color: var(--color-text-muted);
        }

        /* ── GIF section ── */
        .gif-section {
          padding: var(--space-9) 0;
          border-bottom: 1px solid var(--color-border);
        }
        .gif-section__inner {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-8);
          align-items: center;
        }
        .gif-section__heading {
          font-size: clamp(2rem, 4vw, 3.25rem);
          line-height: 1.1;
          letter-spacing: -0.025em;
          margin: var(--space-4) 0 var(--space-5);
        }
        .gif-section__heading em { color: var(--color-accent); font-style: italic; }
        .gif-section__desc {
          font-size: 0.9rem;
          line-height: 1.75;
          color: var(--color-text-secondary);
          max-width: 380px;
        }
        .gif-section__visual {
          display: flex;
          justify-content: center;
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          overflow: hidden;
        }
        .gif-section__gif {
          width: 100%;
          display: block;
          object-fit: cover;
          max-height: 420px;
        }

        /* ── Analyse section ── */
        .analyse-section {
          padding: var(--space-9) 0;
          border-bottom: 1px solid var(--color-border);
        }
        .analyse-section__header {
          margin-bottom: var(--space-7);
        }
        .analyse-section__heading {
          font-size: clamp(2.25rem, 4.5vw, 3.75rem);
          line-height: 1.08;
          letter-spacing: -0.025em;
          margin-top: var(--space-4);
        }
        .analyse-section__heading em { color: var(--color-accent); font-style: italic; }

        .analyse-section__body {
          display: flex;
          justify-content: center;
          align-items: flex-start;
          gap: var(--space-8);
          padding-top: var(--space-4);
        }

        .analyse-section__blueprint {
          flex: 1;
          display: flex;
          justify-content: center;
          align-items: flex-start;
        }

        .blueprint-frame {
          position: relative;
          padding: var(--space-7) var(--space-6);
          border: 1px solid rgba(15, 14, 12, 0.15);
          display: inline-block;
          background: var(--color-bg);
        }

        .crosshair {
          position: absolute;
          width: 16px;
          height: 16px;
        }
        .crosshair::before, .crosshair::after {
          content: '';
          position: absolute;
          background: rgba(15, 14, 12, 0.5);
        }
        .crosshair::before { width: 100%; height: 1px; top: 50%; left: 0; }
        .crosshair::after { width: 1px; height: 100%; left: 50%; top: 0; }
        
        .top-left { top: -8px; left: -8px; }
        .top-right { top: -8px; right: -8px; }
        .bottom-left { bottom: -8px; left: -8px; }
        .bottom-right { bottom: -8px; right: -8px; }

        .frame-label {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          font-size: 0.55rem;
          font-weight: 600;
          letter-spacing: 0.15em;
          color: var(--color-text-muted);
          display: flex;
          align-items: center;
          gap: var(--space-3);
          background: var(--color-bg);
          padding: 0 var(--space-4);
        }
        .frame-label--top { top: -6px; }
        .frame-label--bottom { bottom: -6px; }
        .frame-label .line { width: 24px; height: 1px; background: var(--color-border); }

        .blueprint-frame img {
          max-height: 520px;
          width: auto;
          object-fit: contain;
          display: block;
        }

        .bounding-box {
          position: absolute;
          border: 1px solid var(--color-text-primary);
          background: rgba(15, 14, 12, 0.02);
        }
        .box-label {
          position: absolute;
          top: -10px;
          left: -1px;
          background: var(--color-text-primary);
          color: var(--color-surface);
          font-size: 0.55rem;
          font-weight: 700;
          padding: 2px 6px;
          letter-spacing: 0.1em;
          white-space: nowrap;
        }

        /* Adjust these percentages slightly if f2.png proportions change */
        .box-1 { top: 25%; left: 15%; width: 35%; height: 25%; }
        .box-2 { top: 55%; right: 15%; width: 28%; height: 35%; }
        .box-3 { bottom: 8%; left: 25%; width: 22%; height: 12%; }

        .upload-card {
          width: 100%;
          max-width: 600px;
          flex-shrink: 0;
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-xl);
          padding: var(--space-7);
          box-shadow: var(--shadow-lg);
        }

        /* ── Footer ── */
        .landing-footer {
          padding: var(--space-6) 0;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .landing-footer__logo {
          font-family: var(--font-display);
          font-size: 1rem;
          font-weight: 700;
          color: var(--color-text-primary);
          letter-spacing: -0.01em;
        }
        .landing-footer__logo em { color: var(--color-accent); font-style: italic; }
        .landing-footer__copy {
          font-size: 0.6875rem;
          color: var(--color-text-muted);
          letter-spacing: 0.06em;
          text-transform: uppercase;
        }

        /* ── Responsive ── */
        @media (max-width: 860px) {
          .hero__model--1 { left: -10%; height: 65%; }
          .hero__model--2 { height: 85%; }
          .hero__model--3 { right: -10%; height: 70%; }
          .gif-section__inner { grid-template-columns: 1fr; }
          .how-it-works__grid { grid-template-columns: 1fr; }
          .how-step { border-right: none; border-bottom: 1px solid var(--color-border); }
          .how-step:last-child { border-bottom: none; }
          
          .analyse-section__body { flex-direction: column; align-items: center; }
          .analyse-section__blueprint { display: none; } /* Hide blueprint on mobile to save space */
          .upload-card { padding: var(--space-5); max-width: 100%; }
        }
      `}</style>
    </div>
  )
}
