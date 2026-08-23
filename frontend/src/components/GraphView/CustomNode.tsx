import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'

interface CustomNodeData {
  label: string
  color: string
  description: string
  [key: string]: unknown
}

function CustomNode({ data }: NodeProps) {
  const { label, color, description } = data as CustomNodeData

  return (
    <div
      className="bg-bg-card border-2 rounded-xl px-5 py-3 shadow-lg min-w-[180px] transition-all hover:scale-105 hover:shadow-xl"
      style={{ borderColor: color }}
    >
      <Handle type="target" position={Position.Top} className="!bg-border !w-2 !h-2" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-sm font-bold text-text-primary">{label}</span>
      </div>
      <p className="text-[11px] text-text-muted leading-snug">{description}</p>
      <Handle type="source" position={Position.Bottom} className="!bg-border !w-2 !h-2" />
    </div>
  )
}

export default memo(CustomNode)
