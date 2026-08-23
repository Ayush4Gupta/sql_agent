import { Database, GitGraph, Search, Cpu } from 'lucide-react'
import type { ConfigResponse } from '@/types/api'
import { cn } from '@/lib/utils'

type Tab = 'query' | 'graph' | 'database'

interface HeaderProps {
  config: ConfigResponse | null
  activeTab: Tab
  onTabChange: (tab: Tab) => void
}

const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'query', label: 'Query', icon: <Search className="w-4 h-4" /> },
  { id: 'graph', label: 'Graph', icon: <GitGraph className="w-4 h-4" /> },
  { id: 'database', label: 'Database', icon: <Database className="w-4 h-4" /> },
]

export default function Header({ config, activeTab, onTabChange }: HeaderProps) {
  return (
    <header className="bg-bg-secondary border-b border-border sticky top-0 z-50 backdrop-blur-sm bg-opacity-90">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center shadow-lg shadow-accent/20">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight">SQL Agent</h1>
              <p className="text-xs text-text-muted">Governed NL-to-SQL</p>
            </div>
          </div>

          {/* Tabs */}
          <nav className="flex items-center gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                  activeTab === tab.id
                    ? 'bg-accent text-white shadow-lg shadow-accent/20'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-card'
                )}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Profile badge */}
          {config && (
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-accent/15 text-accent border border-accent/20">
                {config.profile}
              </span>
              <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-bg-card text-text-secondary border border-border">
                {config.db_backend}
              </span>
              <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-bg-card text-text-secondary border border-border">
                {config.llm_provider}
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
