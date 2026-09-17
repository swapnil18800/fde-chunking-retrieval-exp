import { useEffect, useState } from 'react'
import { api } from '../lib/api'

type Node = { id: number; name: string; doc_freq: number; shared?: number }

/** Radial neighbourhood view of one entity in the scispaCy entity–passage graph. */
export default function KnowledgeGraph() {
  const [q, setQ] = useState('hirschsprung disease')
  const [hits, setHits] = useState<Node[]>([])
  const [data, setData] = useState<Awaited<ReturnType<typeof api.kgEntity>> | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { if (q.length > 1) api.kgSearch(q).then(setHits).catch(() => setHits([])) }, [q])
  const load = (name: string) => api.kgEntity(name).then((d) => { setData(d); setErr(null) }).catch((e) => setErr(String(e)))
  useEffect(() => { load(q) }, []) // eslint-disable-line react-hooks/exhaustive-deps
  const nb = data?.neighbours ?? []
  const maxShared = Math.max(1, ...nb.map((n) => n.shared))
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card space-y-3 p-4">
        <div className="label">Entity</div>
        <input className="input" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && load(q)} placeholder="gene, disease, drug…" />
        <ul className="max-h-64 space-y-1 overflow-auto">{hits.map((h) => <li key={h.id}><button className="w-full rounded px-2 py-1 text-left text-sm hover:bg-slate-50" onClick={() => { setQ(h.name); load(h.name) }}>{h.name} <span className="text-xs text-slate-400">· {h.doc_freq} passages</span></button></li>)}</ul>
        <p className="text-xs text-slate-500">Nodes = lower-cased scispaCy entity surface forms (df 2…2.5% of corpus). An edge's weight is the number of passages where both entities co-occur; this is what the <span className="font-mono">kg</span> retriever walks for 1-hop expansion.</p>
        {err && <div className="text-xs text-red-600">{err}</div>}
      </div>
      <div className="card p-4 lg:col-span-2">
        {data && (
          <>
            <div className="mb-2 text-sm"><b>{data.entity.name}</b> <span className="text-slate-500">· in {data.entity.doc_freq} passages · {nb.length} strongest neighbours</span></div>
            <svg viewBox="0 0 800 520" className="w-full">
              {nb.map((n, i) => {
                const a = (i / nb.length) * Math.PI * 2 - Math.PI / 2
                const r = 150 + (1 - n.shared / maxShared) * 80
                const x = 400 + Math.cos(a) * r, y = 260 + Math.sin(a) * r
                return (
                  <g key={n.id} className="cursor-pointer" onClick={() => { setQ(n.name); load(n.name) }}>
                    <line x1={400} y1={260} x2={x} y2={y} stroke="#0e7490" strokeOpacity={0.15 + 0.7 * (n.shared / maxShared)} strokeWidth={1 + 4 * (n.shared / maxShared)} />
                    <circle cx={x} cy={y} r={6 + 10 * (n.shared / maxShared)} fill="#cffafe" stroke="#0e7490" />
                    <text x={x} y={y + (y > 260 ? 26 : -16)} textAnchor="middle" fontSize={11} fill="#0f172a">{n.name.length > 22 ? n.name.slice(0, 21) + '…' : n.name}</text>
                    <text x={x} y={y + 4} textAnchor="middle" fontSize={9} fill="#0e7490">{n.shared}</text>
                  </g>
                )
              })}
              <circle cx={400} cy={260} r={34} fill="#0e7490" />
              <text x={400} y={264} textAnchor="middle" fontSize={12} fill="white" fontWeight={600}>{data.entity.name.length > 14 ? data.entity.name.slice(0, 13) + '…' : data.entity.name}</text>
            </svg>
            <div className="mt-2 border-t pt-2">
              <div className="label mb-1">Example passages</div>
              <ul className="space-y-1 text-xs text-slate-600">{data.passages.map((p) => <li key={p.id}><a className="font-mono text-accent hover:underline" href={`https://pubmed.ncbi.nlm.nih.gov/${p.id}/`} target="_blank" rel="noreferrer">{p.id}</a> {p.preview}…</li>)}</ul>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
