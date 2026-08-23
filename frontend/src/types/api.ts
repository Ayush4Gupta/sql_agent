export interface TableData {
  columns: string[]
  rows: unknown[][]
}

export interface QueryResponse {
  question: string
  messages: Array<{
    role: string
    content: string
    tool_calls?: unknown[]
  }>
  final_answer: string
  table_data: TableData | null
  node_timings: Record<string, number> | null
}

export interface HealthResponse {
  status: string
  agent: string
  database: string
  tables: string[]
}

export interface ConfigResponse {
  profile: string
  db_backend: string
  llm_provider: string
}

export interface TablesResponse {
  tables: string[]
  count: number
}

export interface SchemaResponse {
  table: string
  schema: string
}
