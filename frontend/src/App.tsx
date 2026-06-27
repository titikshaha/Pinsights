import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useAnalysis } from './hooks/useAnalysis'
import Landing from './pages/Landing'
import Analysis from './pages/Analysis'

type Page = 'landing' | 'analysis'

function Nav({ page, onNavigate }: { page: Page; onNavigate: (p: Page) => void }) {
  return (
    <nav className="nav">
      <span
        className="nav__logo"
        onClick={() => onNavigate('landing')}
        style={{ cursor: 'pointer' }}
      >
        Pin<em>sights</em>
      </span>
      <div className="nav__links">
        <span
          className={`nav__link ${page === 'landing' ? 'nav__link--active' : ''}`}
          onClick={() => onNavigate('landing')}
        >
          Analyse
        </span>
        <span
          className={`nav__link ${page === 'analysis' ? 'nav__link--active' : ''}`}
          onClick={() => onNavigate('analysis')}
        >
          Results
        </span>
      </div>
    </nav>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>('landing')
  const { state, analyse, reset } = useAnalysis()

  const handleAnalysisStarted = () => {
    setPage('analysis')
  }

  const handleReset = () => {
    reset()
    setPage('landing')
  }

  return (
    <>
      <Nav page={page} onNavigate={setPage} />
      <AnimatePresence mode="wait">
        {page === 'landing' ? (
          <motion.div
            key="landing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <Landing
              onAnalysisStarted={handleAnalysisStarted}
              onAnalyse={analyse}
              isRunning={state.status === 'running'}
            />
          </motion.div>
        ) : (
          <motion.div
            key="analysis"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <Analysis state={state} onReset={handleReset} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
