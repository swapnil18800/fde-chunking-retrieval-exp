import type { Stage } from '../lib/api'

/** Timeline of pipeline stages with per-stage latency — what actually happened to this question. */
export default function Stages({ stages, queries, hypothetical }: { stages: Stage[]; queries?: string[]; hypothetical?: string | null }) {
  const total = stages.reduce((a, s) => a + (s.ms || 0), 0) || 1
  return (
    <div className="card p-5">
      <div className="label mb-3">Pipeline stages · {total} ms</div>
      <div className="mb-3 flex h-2 overflow-hidden rounded bg-slate-100">
        {stages.map((s, i) => <div key={i} title={`${s.name} ${s.ms}ms`} style={{ width: `${Math.max(1, (s.ms / total) * 100)}%` }} className={COLORS[i % COLORS.length]} />)}
      </div>
      <ul className="space-y-1 text-sm">
        {stages.map((s, i) => (
          <li key={i} className="flex flex-wrap items-baseline gap-2">
            <span className={`h-2 w-2 rounded-full ${COLORS[i % COLORS.length]}`} />
            <span className="font-mono text-xs">{s.name}</span>
            <span className="text-xs text-slate-500">{s.ms} ms</span>
            <span className="text-xs text-slate-400">{Object.entries(s).filter(([k]) => !['name', 'ms'].includes(k)).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : String(v)}`).join(' · ')}</span>
          </li>
        ))}
      </ul>
      {queries && queries.length > 1 && (
        <div className="mt-3 border-t pt-3 text-sm">
          <div className="label mb-1">Queries sent to the retriever</div>
          <ul className="list-disc pl-5 text-slate-700">{queries.map((q, i) => <li key={i}>{q}</li>)}</ul>
          {hypothetical && <p className="mt-2 rounded bg-slate-50 p-2 text-xs text-slate-600"><b>HyDE draft:</b> {hypothetical}</p>}
        </div>
      )}
    </div>
  )
}
const COLORS = ['bg-cyan-600', 'bg-amber-500', 'bg-violet-500', 'bg-emerald-500', 'bg-rose-400', 'bg-sky-400']
