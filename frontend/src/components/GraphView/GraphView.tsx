import { useCallback, useMemo } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  Position,
  MarkerType,
} from '@xyflow/react'
import CustomNode from './CustomNode'

const nodeTypes = { custom: CustomNode }

const NODE_DEFS: { id: string; label: string; color: string; description: string }[] = [
  { id: 'list_tables',     label: 'List Tables',     color: '#6366f1', description: 'Lists all available database tables' },
  { id: 'call_get_schema', label: 'Call Get Schema', color: '#8b5cf6', description: 'Decides which table schemas to fetch' },
  { id: 'get_schema',      label: 'Get Schema',      color: '#8b5cf6', description: 'Retrieves DDL + sample rows' },
  { id: 'schema_analysis', label: 'Schema Analysis', color: '#d97706', description: 'Evaluates schema completeness' },
  { id: 'generate_query',  label: 'Generate Query',  color: '#ea580c', description: 'Generates SQL from NL question' },
  { id: 'check_query',     label: 'Check Query',     color: '#0891b2', description: 'Validates and rewrites SQL' },
  { id: 'run_query',       label: 'Run Query',       color: '#0d9488', description: 'Executes SQL, returns results' },
]

const POSITIONS: Record<string, { x: number; y: number }> = {
  list_tables:     { x: 300, y: 0 },
  call_get_schema: { x: 300, y: 120 },
  get_schema:      { x: 300, y: 240 },
  schema_analysis: { x: 300, y: 360 },
  generate_query:  { x: 300, y: 500 },
  check_query:     { x: 300, y: 640 },
  run_query:       { x: 300, y: 760 },
}

export default function GraphView() {
  const nodes: Node[] = useMemo(() =>
    NODE_DEFS.map((def) => ({
      id: def.id,
      type: 'custom',
      position: POSITIONS[def.id],
      data: { label: def.label, color: def.color, description: def.description },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    })),
    []
  )

  const edges: Edge[] = useMemo(() => [
    { id: 'e1', source: 'list_tables', target: 'call_get_schema', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b' }, style: { stroke: '#64748b' } },
    { id: 'e2', source: 'call_get_schema', target: 'get_schema', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b' }, style: { stroke: '#64748b' } },
    { id: 'e3', source: 'get_schema', target: 'schema_analysis', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b' }, style: { stroke: '#64748b' } },
    { id: 'e4a', source: 'schema_analysis', target: 'generate_query', label: 'COMPLETE', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#22c55e' }, style: { stroke: '#22c55e' } },
    { id: 'e4b', source: 'schema_analysis', target: 'call_get_schema', label: 'NEED MORE', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#f59e0b' }, style: { stroke: '#f59e0b' }, type: 'smoothstep' },
    { id: 'e5', source: 'generate_query', target: 'check_query', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b' }, style: { stroke: '#64748b' } },
    { id: 'e6', source: 'check_query', target: 'run_query', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b' }, style: { stroke: '#64748b' } },
    { id: 'e7', source: 'run_query', target: 'generate_query', label: 'LOOP', animated: true, markerEnd: { type: MarkerType.ArrowClosed, color: '#ea580c' }, style: { stroke: '#ea580c' }, type: 'smoothstep' },
  ], [])

  return (
    <div className="bg-bg-card rounded-xl border border-border shadow-lg overflow-hidden" style={{ height: '80vh' }}>
      <div className="px-6 py-4 border-b border-border">
        <h2 className="text-sm font-bold text-text-primary">Agent Workflow Graph</h2>
        <p className="text-xs text-text-muted mt-0.5">
          LangGraph execution pipeline — list tables → schema → generate → validate → execute
        </p>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        proOptions={{ hideAttribution: true }}
        className="bg-bg-primary"
      >
        <Background color="#1e293b" gap={20} />
        <Controls
          className="!bg-bg-card !border-border !shadow-lg [&>button]:!bg-bg-secondary [&>button]:!border-border [&>button]:!text-text-secondary [&>button:hover]:!bg-bg-card-hover"
        />
        <MiniMap
          nodeColor={(node) => node.data?.color as string || '#64748b'}
          className="!bg-bg-secondary !border-border"
          maskColor="rgba(11, 17, 32, 0.7)"
        />
      </ReactFlow>
    </div>
  )
}
