import type {
  QueryResponse,
  HealthResponse,
  ConfigResponse,
  TablesResponse,
  SchemaResponse,
} from '@/types/api'

const BASE = ''  // proxy handles routing in dev

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export function submitQuery(question: string): Promise<QueryResponse> {
  return fetchJson<QueryResponse>('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
}

export function fetchHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>('/health')
}

export function fetchConfig(): Promise<ConfigResponse> {
  return fetchJson<ConfigResponse>('/api/config')
}

export function fetchTables(): Promise<TablesResponse> {
  return fetchJson<TablesResponse>('/database/tables')
}

export function fetchSchema(tableName: string): Promise<SchemaResponse> {
  return fetchJson<SchemaResponse>(`/database/schema/${tableName}`)
}
