import { useEffect, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import { api, fmt, type AskResult, type LogRow } from '../lib/api'
import Answer from '../components/Answer'
import Stages from '../components/Stages'
import Metrics from '../components/Metrics'

/** Everything that happened to any question ever asked (API, evals, smoke tests) — from the query_logs table. */
export default function QueryInspector() {
  const [rows, setRows] = useState<LogRow[]>([])
  const [q, setQ] = useState('')
  const [source, setSource] = useState('')
  const [sel, setSel] = useState<(AskResult & Record<string, unknown>) | null>(null)
  useEffect(() => { api.logs(q, source).then(setRows).catch(() => setRows([])) }, [q, source])
  return (
    <div className="grid gap-4 lg:grid-cols-5">
      <div className="card p-4 lg:col-span-2">
        <div className="mb-2 flex gap-2">
          <input className="input" placeholder="filter by question…" value={q} onChange={(e) => setQ(e.target.value)} />
          <select className="input max-w-[110px]" value={source} onChange={(e) => setSource(e.target.value)}><option value="">all</option><option>api</option><option>eval</option><option>smoke</option></select>
        </div>
        <ul className="max-h-[70vh] space-y-1 overflow-auto">
          {rows.map((r) => (
            <li key={r.id}><button onClick={() => api.log(r.id).then(setSel)} className={`w-full rounded p-2 text-left text-xs hover:bg-slate-50 ${sel?.id === r.id ? 'bg-accent-soft' : ''}`}>
              <div className="flex items-center gap-2 text-slate-500"><span className="chip border-slate-200">{r.source}</span><span>{new Date(r.created_at).toLocaleString()}</span><span>{r.latency_ms} ms</span>{r.error && <span className="text-red-600">error</span>}{r.metrics?.['recall@5'] != null && <span>r@5 {fmt(r.metrics['recall@5'], 2)}</span>}</div>
              <div className="line-clamp-2 text-slate-800">{r.question}</div>
              <div className="font-mono text-[10px] text-slate-400">{r.config?.strategy}+{r.config?.retriever}{r.config?.rerank ? '+rerank' : ''}{r.config?.expansion !== 'none' ? '+' + r.config?.expansion : ''}</div>
            </button></li>
          ))}
        </ul>
      </div>
      <div className="space-y-4 lg:col-span-3">
        {!sel && <div className="card p-6 text-sm text-slate-500">Select a run. Each row is one pipeline execution: config, every stage with timing, retrieved chunks with per-retriever ranks, answer, citations, tokens, and a link to the trace.</div>}
        {sel && (
          <>
            <div className="card p-4 text-xs text-slate-600">
              <div className="flex flex-wrap items-center gap-3">
                <span className="chip border-slate-200">{String(sel.source)}</span><span>{new Date(String(sel.created_at)).toLocaleString()}</span>
                <span className="font-mono">{sel.id}</span>
                {sel.trace_url && <a className="text-accent hover:underline" href={sel.trace_url} target="_blank" rel="noreferrer">trace ({sel.trace_provider}) <ExternalLink className="inline" size={11} /></a>}
              </div>
              <div className="mt-1 font-mono">{JSON.stringify(sel.config)}</div>
              {sel.tokens && Object.keys(sel.tokens).length > 0 && <div className="mt-1">tokens {JSON.stringify(sel.tokens)}</div>}
              {sel.error && <div className="mt-1 text-red-600">{sel.error}</div>}
            </div>
            <div className="card p-4 text-sm"><div className="label mb-1">Question</div>{sel.question}</div>
            {sel.metrics && <Metrics m={sel.metrics as Record<string, number>} />}
            <Stages stages={sel.stages ?? []} />
            {(sel.answer || sel.retrieved?.length) ? <Answer r={{ ...sel, citations: sel.citations ?? [], retrieved: sel.retrieved ?? [], answer: sel.answer ?? '' }} /> : null}
          </>
        )}
      </div>
    </div>
  )
}
