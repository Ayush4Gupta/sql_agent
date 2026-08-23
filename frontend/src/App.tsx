import { useState, useEffect } from 'react'
import { fetchConfig } from '@/hooks/useApi'
import type { ConfigResponse } from '@/types/api'
import Header from '@/components/Layout/Header'
import QueryPanel from '@/components/QueryPanel/QueryPanel'
import DatabaseExplorer from '@/components/DatabaseExplorer/DatabaseExplorer'

type Tab = 'query' | 'database'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('query')
  const [config, setConfig] = useState<ConfigResponse | null>(null)

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch((err) => console.warn('Failed to load config:', err))
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      <Header config={config} activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6">
        {activeTab === 'query' && <QueryPanel />}
        {activeTab === 'database' && <DatabaseExplorer />}
      </main>
    </div>
  )
}
