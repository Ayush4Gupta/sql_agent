import { useState } from 'react'
import { Send, Loader2, Table2, Clock, AlertCircle } from 'lucide-react'
import { submitQuery } from '@/hooks/useApi'
import type { QueryResponse } from '@/types/api'
import TraceTimeline from '@/components/AgentTrace/TraceTimeline'
import { cn } from '@/lib/utils'

export default function QueryPanel() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!question.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await submitQuery(question.trim())
      setResult(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="space-y-6">
      {/* Input card */}
      <div className="bg-bg-card rounded-xl border border-border p-6 shadow-lg">
        <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
          Ask a question
        </label>
        <div className="relative">
          <textarea
            id="query-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. What are the top 10 products by total sales?"
            rows={3}
            className="w-full bg-bg-secondary border border-border rounded-lg px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent resize-none transition-all"
          />
          <button
            id="submit-query"
            onClick={handleSubmit}
            disabled={!question.trim() || loading}
            className={cn(
              'absolute right-3 bottom-3 p-2 rounded-lg transition-all duration-200',
              question.trim() && !loading
                ? 'bg-accent hover:bg-accent-hover text-white shadow-lg shadow-accent/20 cursor-pointer'
                : 'bg-bg-card text-text-muted cursor-not-allowed'
            )}
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        <p className="mt-2 text-xs text-text-muted">
          Press <kbd className="px-1.5 py-0.5 rounded bg-bg-secondary border border-border text-text-secondary text-[10px] font-mono">Ctrl+Enter</kbd> to submit
        </p>
      </div>

      {/* Loading skeleton */}
      {loading && (
        <div className="bg-bg-card rounded-xl border border-border p-6 shadow-lg animate-pulse">
          <div className="flex items-center gap-3 mb-4">
            <Loader2 className="w-5 h-5 text-accent animate-spin" />
            <span className="text-sm text-text-secondary">Processing your query...</span>
          </div>
          <div className="space-y-2">
            <div className="h-4 bg-bg-secondary rounded w-3/4" />
            <div className="h-4 bg-bg-secondary rounded w-1/2" />
            <div className="h-4 bg-bg-secondary rounded w-5/6" />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-error/10 border border-error/20 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-error flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-error">Query failed</p>
            <p className="text-sm text-text-secondary mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          {/* Answer card */}
          <div className="bg-bg-card rounded-xl border border-border p-6 shadow-lg">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
              Answer
            </h3>
            <div className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
              {result.final_answer}
            </div>
          </div>

          {/* Table data */}
          {result.table_data && result.table_data.columns.length > 0 && (
            <div className="bg-bg-card rounded-xl border border-border shadow-lg overflow-hidden">
              <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Table2 className="w-4 h-4 text-accent" />
                  <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
                    Results
                  </h3>
                </div>
                <span className="text-xs text-text-muted">
                  {result.table_data.rows.length} rows
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      {result.table_data.columns.map((col, i) => (
                        <th
                          key={i}
                          className="px-4 py-3 text-left text-xs font-semibold text-text-secondary uppercase tracking-wider bg-bg-secondary"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.table_data.rows.map((row, ri) => (
                      <tr
                        key={ri}
                        className="border-b border-border-subtle hover:bg-bg-card-hover transition-colors"
                      >
                        {row.map((cell, ci) => (
                          <td key={ci} className="px-4 py-2.5 text-text-primary font-mono text-xs">
                            {cell === null ? (
                              <span className="text-text-muted italic">null</span>
                            ) : (
                              String(cell)
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Trace timeline */}
          {result.node_timings && (
            <TraceTimeline timings={result.node_timings} />
          )}
        </div>
      )}
    </div>
  )
}
