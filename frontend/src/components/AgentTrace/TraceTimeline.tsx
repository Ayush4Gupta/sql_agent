import { Clock } from 'lucide-react'
import { cn } from '@/lib/utils'

const NODE_COLORS: Record<string, string> = {
  list_tables: 'bg-node-list',
  call_get_schema: 'bg-node-schema',
  get_schema: 'bg-node-schema',
  schema_analysis: 'bg-node-analysis',
  generate_query: 'bg-node-generate',
  check_query: 'bg-node-check',
  run_query: 'bg-node-run',
}

const NODE_ORDER = [
  'list_tables',
  'call_get_schema',
  'get_schema',
  'schema_analysis',
  'generate_query',
  'check_query',
  'run_query',
]

interface TraceTimelineProps {
  timings: Record<string, number>
}

export default function TraceTimeline({ timings }: TraceTimelineProps) {
  const totalTime = Object.values(timings).reduce((a, b) => a + b, 0)

  // Sort nodes in execution order
  const orderedNodes = Object.entries(timings).sort(([a], [b]) => {
    const ai = NODE_ORDER.indexOf(a)
    const bi = NODE_ORDER.indexOf(b)
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi)
  })

  return (
    <div className="bg-bg-card rounded-xl border border-border p-6 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-accent" />
          <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
            Agent Trace
          </h3>
        </div>
        <span className="text-xs font-mono text-text-secondary">
          Total: {totalTime.toFixed(2)}s
        </span>
      </div>

      {/* Bar chart */}
      <div className="flex rounded-lg overflow-hidden h-8 mb-4">
        {orderedNodes.map(([node, time]) => {
          const pct = (time / totalTime) * 100
          return (
            <div
              key={node}
              className={cn(
                'flex items-center justify-center text-[10px] font-semibold text-white transition-all hover:opacity-80',
                NODE_COLORS[node] || 'bg-text-muted'
              )}
              style={{ width: `${Math.max(pct, 3)}%` }}
              title={`${node}: ${time.toFixed(3)}s`}
            >
              {pct > 8 ? `${time.toFixed(1)}s` : ''}
            </div>
          )
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3">
        {orderedNodes.map(([node, time]) => (
          <div key={node} className="flex items-center gap-1.5">
            <div className={cn('w-2.5 h-2.5 rounded-full', NODE_COLORS[node] || 'bg-text-muted')} />
            <span className="text-xs text-text-secondary">
              {node.replace(/_/g, ' ')} ({time.toFixed(2)}s)
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
