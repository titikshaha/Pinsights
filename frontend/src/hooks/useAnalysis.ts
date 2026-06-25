import { useState, useCallback, useRef } from 'react'
import { StreamEvent, AnalysisResult, startAnalysis, subscribeToStream, PresetBoard } from '../../lib/api'

export interface AnalysisState {
  status: 'idle' | 'running' | 'done' | 'error'
  progress: number
  stage: string
  message: string
  result: AnalysisResult | null
  error: string | null
  sessionId: string | null
}

const initial: AnalysisState = {
  status: 'idle',
  progress: 0,
  stage: '',
  message: '',
  result: null,
  error: null,
  sessionId: null,
}

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>(initial)
  const unsubRef = useRef<(() => void) | null>(null)

  const analyse = useCallback(async (
    source: 'upload' | 'preset' | 'unsplash',
    options: { files?: File[]; board?: string; query?: string }
  ) => {
    // Cancel any existing stream
    unsubRef.current?.()

    setState({ ...initial, status: 'running', message: 'Starting analysis...' })

    try {
      const sessionId = await startAnalysis(source, options)
      setState(s => ({ ...s, sessionId }))

      const unsub = subscribeToStream(
        sessionId,
        (event: StreamEvent) => {
          if (event.type === 'progress') {
            setState(s => ({
              ...s,
              progress: event.progress,
              stage: event.stage,
              message: event.message,
            }))
          } else if (event.type === 'result') {
            setState(s => ({
              ...s,
              status: 'done',
              progress: 100,
              message: 'Analysis complete',
              result: event.data,
            }))
          } else if (event.type === 'error') {
            setState(s => ({
              ...s,
              status: 'error',
              error: event.message,
              message: 'Analysis failed',
            }))
          }
        },
        () => {
          // Stream closed
          setState(s => {
            if (s.status === 'running') {
              return { ...s, status: 'error', error: 'Stream disconnected unexpectedly' }
            }
            return s
          })
        }
      )
      unsubRef.current = unsub
    } catch (err: any) {
      setState(s => ({
        ...s,
        status: 'error',
        error: err.message || 'Unknown error',
        message: 'Failed to start analysis',
      }))
    }
  }, [])

  const reset = useCallback(() => {
    unsubRef.current?.()
    setState(initial)
  }, [])

  return { state, analyse, reset }
}
