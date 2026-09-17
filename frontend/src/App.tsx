import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { Activity, BookOpen, FlaskConical, GitCompare, Layers, ListTree, Network, ScrollText } from 'lucide-react'
import Overview from './pages/Overview'
import Playground from './pages/Playground'
import Compare from './pages/Compare'
import Leaderboard from './pages/Leaderboard'
import ChunkExplorer from './pages/ChunkExplorer'
import KnowledgeGraph from './pages/KnowledgeGraph'
import QueryInspector from './pages/QueryInspector'
import { useHealth } from './lib/hooks'

const TABS = [
  { to: '/overview', label: 'Overview', icon: BookOpen },
  { to: '/playground', label: 'Playground', icon: FlaskConical },
  { to: '/compare', label: 'Compare', icon: GitCompare },
  { to: '/leaderboard', label: 'Leaderboard', icon: Layers },
  { to: '/chunks', label: 'Chunk Explorer', icon: ListTree },
  { to: '/graph', label: 'Knowledge Graph', icon: Network },
  { to: '/inspector', label: 'Query Inspector', icon: ScrollText },
]

export default function App() {
  const health = useHealth()
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-6 py-3">
          <div className="flex items-center gap-2">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-white"><Activity size={16} /></div>
            <div>
              <div className="text-sm font-semibold leading-tight">BioASQ RAG Lab</div>
              <div className="text-[11px] text-slate-500">chunking × retrieval experiments · rag-mini-bioasq</div>
            </div>
          </div>
          <nav className="ml-4 flex flex-1 gap-1 overflow-x-auto">
            {TABS.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} className={({ isActive }) =>
                `flex items-center gap-1.5 whitespace-nowrap rounded-lg px-3 py-1.5 text-sm ${isActive ? 'bg-accent-soft text-accent font-medium' : 'text-slate-600 hover:bg-slate-100'}`}>
                <Icon size={15} /> {label}
              </NavLink>
            ))}
          </nav>
          <div className="hidden items-center gap-3 text-xs text-slate-500 md:flex">
            {health && <><span>{health.passages.toLocaleString()} passages</span><span>·</span><span>{health.db_size}</span><span>·</span>
              <span className={`chip ${health.tracing === 'off' ? 'border-slate-200' : 'border-emerald-200 bg-emerald-50 text-emerald-700'}`}>tracing: {health.tracing}</span></>}
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/chunks" element={<ChunkExplorer />} />
          <Route path="/graph" element={<KnowledgeGraph />} />
          <Route path="/inspector" element={<QueryInspector />} />
        </Routes>
      </main>
      <footer className="border-t border-slate-200 py-4 text-center text-xs text-slate-400">
        Supabase pgvector · LangGraph · Gemini (free tier) · MedEmbed · MedCPT · Langfuse · RAGAS
      </footer>
    </div>
  )
}
