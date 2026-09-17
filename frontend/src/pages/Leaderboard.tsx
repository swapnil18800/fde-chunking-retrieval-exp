import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, fmt, type EvalRun } from '../lib/api'
import { useAsync } from '../lib/hooks'

type Row = Record<string, number | string | boolean>
const METRICS = ['recall@5', 'recall@10', 'precision@5', 'mrr', 'ndcg@10', 'hit@5', 'latency_ms']
const RAGAS = ['faithfulness', 'answer_relevancy', 'id_based_context_precision', 'id_based_context_recall', 'factual_correctness(mode=f1)', 'recall@5', 'latency_ms']

export default function Leaderboard() {
  const runs = useAsync(() => api.evalRuns(), [])
  const [sel, setSel] = useState<string | null>(null)
  const [metric, setMetric] = useState('recall@10')
  const done = (runs.data ?? []).filter((r) => r.status === 'done' && r.summary?.configs?.length)
  const run: EvalRun | undefined = done.find((r) => r.id === sel) ?? done[0]
  const rows: Row[] = useMemo(() => (run?.summary?.configs ?? []) as Row[], [run])
  const cols = run?.kind === 'ragas' ? RAGAS : METRICS
  const sorted = [...rows].sort((a, b) => Number(b[metric] ?? 0) - Number(a[metric] ?? 0))
  const heat = useMemo(() => {
    if (run?.kind !== 'retrieval') return null
    const base = rows.filter((r) => r.rerank === false && r.transform === 'none' && r.expansion === 'none')
    const strategies = [...new Set(base.map((r) => String(r.strategy)))]
    const retrievers = [...new Set(base.map((r) => String(r.retriever)))]
    if (strategies.length < 2 || retrievers.length < 2) return null
    return { strategies, retrievers, at: (s: string, r: string) => base.find((x) => x.strategy === s && x.retriever === r)?.[metric] as number | undefined }
  }, [rows, metric, run])

  if (runs.loading) return <div className="text-sm text-slate-500">loading eval runs…</div>
  if (!done.length) return <div className="card p-6 text-sm text-slate-600">No finished eval runs yet. Run <code>uv run python evals/run_retrieval_eval.py --set eval150 --matrix</code>.</div>
  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-center gap-3 p-4">
        <div className="label">Run</div>
        <select className="input max-w-xl" value={run?.id} onChange={(e) => setSel(e.target.value)}>
          {done.map((r) => <option key={r.id} value={r.id}>{r.kind} · {r.name} · {r.eval_set} ({r.n_questions} q) · {new Date(r.created_at).toLocaleString()}</option>)}
        </select>
        <div className="label ml-4">Metric</div>
        <select className="input max-w-[220px]" value={metric} onChange={(e) => setMetric(e.target.value)}>{cols.map((c) => <option key={c}>{c}</option>)}</select>
      </div>

      {heat && (
        <div className="card overflow-x-auto p-5">
          <div className="label mb-3">{metric} — chunking × retriever (no transform / rerank / expansion)</div>
          <table className="text-sm">
            <thead><tr><th className="p-2 text-left text-xs text-slate-500">chunking \ retriever</th>{heat.retrievers.map((r) => <th key={r} className="p-2 font-mono text-xs">{r}</th>)}</tr></thead>
            <tbody>
              {heat.strategies.map((s) => (
                <tr key={s}><td className="p-2 font-mono text-xs">{s}</td>
                  {heat.retrievers.map((r) => { const v = heat.at(s, r); return <td key={r} className="p-1"><Cell v={v} max={metric === 'latency_ms' ? Math.max(...rows.map((x) => Number(x.latency_ms) || 0)) : 1} invert={metric === 'latency_ms'} /></td> })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card p-5">
        <div className="label mb-3">{metric} by config</div>
        <ResponsiveContainer width="100%" height={Math.max(220, sorted.length * 22)}>
          <BarChart data={sorted.map((r) => ({ name: String(r.config), v: Number(r[metric]) }))} layout="vertical" margin={{ left: 10, right: 30 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" domain={metric === 'latency_ms' ? [0, 'auto'] : [0, 1]} tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={280} tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
            <Tooltip formatter={(v) => fmt(Number(v), 3)} />
            <Bar dataKey="v" fill="#0e7490" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card overflow-x-auto p-5">
        <div className="label mb-3">All configs · {run?.n_questions} questions · sorted by {metric}</div>
        <table className="w-full text-sm">
          <thead><tr className="text-left text-xs text-slate-500"><th className="py-1">config</th>{cols.map((c) => <th key={c} className="px-2">{c}</th>)}<th>errors</th></tr></thead>
          <tbody>{sorted.map((r, i) => (
            <tr key={i} className="border-t"><td className="py-1.5 font-mono text-xs">{String(r.config)}</td>
              {cols.map((c) => <td key={c} className={`px-2 tabular-nums ${c === metric ? 'font-semibold' : ''}`}>{c === 'latency_ms' ? Math.round(Number(r[c] ?? 0)) : fmt(Number(r[c]), 3)}</td>)}
              <td>{String(r.errors ?? 0)}</td></tr>))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Cell({ v, max, invert }: { v?: number; max: number; invert: boolean }) {
  if (v == null) return <div className="h-9 w-20 rounded bg-slate-50" />
  const t = Math.min(1, Math.max(0, v / max))
  const a = (invert ? 1 - t : t) * 0.9 + 0.05
  return <div className="grid h-9 w-20 place-items-center rounded text-xs font-semibold tabular-nums" style={{ background: `rgba(14,116,144,${a})`, color: a > 0.5 ? 'white' : '#0f172a' }}>{invert ? Math.round(v) : v.toFixed(3)}</div>
}
