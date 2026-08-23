import { useState, useEffect } from 'react'
import { Database, Table2, ChevronRight, Loader2 } from 'lucide-react'
import { fetchTables, fetchSchema } from '@/hooks/useApi'
import { cn } from '@/lib/utils'

export default function DatabaseExplorer() {
  const [tables, setTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string | null>(null)
  const [schemaText, setSchemaText] = useState<string>('')
  const [loadingTables, setLoadingTables] = useState(true)
  const [loadingSchema, setLoadingSchema] = useState(false)

  useEffect(() => {
    fetchTables()
      .then((res) => setTables(res.tables))
      .catch((err) => console.error('Failed to load tables:', err))
      .finally(() => setLoadingTables(false))
  }, [])

  const handleSelectTable = async (name: string) => {
    setSelectedTable(name)
    setLoadingSchema(true)
    setSchemaText('')
    try {
      const res = await fetchSchema(name)
      setSchemaText(res.schema)
    } catch (err) {
      setSchemaText(`Error loading schema: ${err}`)
    } finally {
      setLoadingSchema(false)
    }
  }

  return (
    <div className="flex gap-6 h-[calc(100vh-120px)]">
      {/* Sidebar */}
      <div className="w-72 bg-bg-card rounded-xl border border-border shadow-lg flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center gap-2">
          <Database className="w-4 h-4 text-accent" />
          <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
            Tables
          </h3>
          <span className="ml-auto text-xs text-text-muted">{tables.length}</span>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {loadingTables ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-5 h-5 text-accent animate-spin" />
            </div>
          ) : (
            tables.map((table) => (
              <button
                key={table}
                onClick={() => handleSelectTable(table)}
                className={cn(
                  'w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-all duration-150',
                  selectedTable === table
                    ? 'bg-accent/15 text-accent font-medium'
                    : 'text-text-secondary hover:bg-bg-card-hover hover:text-text-primary'
                )}
              >
                <Table2 className="w-3.5 h-3.5 flex-shrink-0" />
                <span className="truncate font-mono text-xs">{table}</span>
                {selectedTable === table && (
                  <ChevronRight className="w-3.5 h-3.5 ml-auto text-accent" />
                )}
              </button>
            ))
          )}
        </div>
      </div>

      {/* Schema viewer */}
      <div className="flex-1 bg-bg-card rounded-xl border border-border shadow-lg flex flex-col overflow-hidden">
        {selectedTable ? (
          <>
            <div className="px-6 py-4 border-b border-border">
              <h3 className="text-sm font-bold text-text-primary font-mono">{selectedTable}</h3>
              <p className="text-xs text-text-muted mt-0.5">DDL + sample rows</p>
            </div>
            <div className="flex-1 overflow-auto p-6">
              {loadingSchema ? (
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 text-accent animate-spin" />
                  <span className="text-sm text-text-secondary">Loading schema...</span>
                </div>
              ) : (
                <pre className="text-xs font-mono text-text-secondary leading-relaxed whitespace-pre-wrap break-words">
                  {schemaText}
                </pre>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Database className="w-12 h-12 text-border mx-auto mb-3" />
              <p className="text-sm text-text-muted">Select a table to view its schema</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
