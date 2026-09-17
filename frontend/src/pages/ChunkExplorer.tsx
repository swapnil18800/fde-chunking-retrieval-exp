import { useEffect, useState } from 'react'
import { api, type PassageDetail } from '../lib/api'
import { useOptions } from '../lib/hooks'

const PALETTE = ['bg-cyan-100', 'bg-amber-100', 'bg-violet-100', 'bg-emerald-100', 'bg-rose-100', 'bg-sky-100', 'bg-lime-100', 'bg-orange-100']

/** See exactly how each strategy cuts a real passage — the most instructive view in the lab. */
export default function ChunkExplorer() {
  const opts = useOptions()
  const [samples, setSamples] = useState<{ id: number; preview: string; n_tokens: number }[]>([])
  const [pmid, setPmid] = useState<number | null>(null)
  const [p, setP] = useState<PassageDetail | null>(null)
  const [q, setQ] = useState('')
  const [custom, setCustom] = useState('')
  useEffect(() => { api.samplePassages(q).then(setSamples).catch(() => setSamples([])) }, [q])
  useEffect(() => { if (pmid) api.passage(pmid).then(setP) }, [pmid])
  useEffect(() => { if (!pmid && samples.length) setPmid(samples[0].id) }, [samples, pmid])
  const runCustom = async () => { const r = await api.chunkPreview(custom); setP({ id: 0, text: r.text, n_words: r.text.split(' ').length, n_tokens: 0, sentence_offsets: null, url: '', chunks: r.chunks, entities: [] }) }
  const strategies = p ? Object.keys(p.chunks).sort((a, b) => (opts?.strategies.findIndex((s) => s.name === a) ?? 0) - (opts?.strategies.findIndex((s) => s.name === b) ?? 0)) : []
  return (
    <div className="grid gap-4 lg:grid-cols-4">
      <div className="space-y-3 lg:col-span-1">
        <div className="card p-4">
          <div className="label mb-2">Pick a passage</div>
          <input className="input mb-2" placeholder="search text… (e.g. metformin)" value={q} onChange={(e) => setQ(e.target.value)} />
          <ul className="max-h-[420px] space-y-1 overflow-auto">
            {samples.map((s) => <li key={s.id}><button onClick={() => setPmid(s.id)} className={`w-full rounded p-2 text-left text-xs hover:bg-slate-50 ${pmid === s.id ? 'bg-accent-soft' : ''}`}><b>PMID {s.id}</b> · {s.n_tokens} tok<div className="line-clamp-2 text-slate-500">{s.preview}</div></button></li>)}
          </ul>
        </div>
        <div className="card p-4">
          <div className="label mb-2">…or paste your own text</div>
          <textarea className="input min-h-[90px]" value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="Any abstract. Semantic chunking needs stored embeddings so it's skipped here." />
          <button className="btn-ghost mt-2 w-full justify-center" onClick={runCustom} disabled={custom.length < 40}>chunk it</button>
        </div>
      </div>
      <div className="space-y-4 lg:col-span-3">
        {p && (
          <>
            <div className="card p-5">
              <div className="mb-2 flex flex-wrap items-center gap-3 text-xs text-slate-500">
                {p.id ? <a className="font-semibold text-accent hover:underline" href={p.url} target="_blank" rel="noreferrer">PMID {p.id}</a> : <b>custom text</b>}
                <span>{p.n_words} words</span>{p.n_tokens ? <span>{p.n_tokens} tokens</span> : null}
                {p.sentence_offsets && <span>{p.sentence_offsets.length} sentences (scispaCy)</span>}
              </div>
              <p className="text-sm leading-relaxed text-slate-700">{p.text}</p>
              {p.entities.length > 0 && <div className="mt-3 flex flex-wrap gap-1">{p.entities.slice(0, 30).map((e) => <span key={e.id} className="chip border-violet-200 bg-violet-50 text-violet-700" title={`in ${e.doc_freq} passages`}>{e.name}</span>)}</div>}
            </div>
            {strategies.map((s) => {
              const chunks = p.chunks[s]
              const meta = opts?.strategies.find((x) => x.name === s)
              return (
                <div key={s} className="card p-5">
                  <div className="mb-2 flex flex-wrap items-baseline gap-3">
                    <span className="font-mono text-sm font-semibold text-accent">{s}</span>
                    <span className="text-xs text-slate-500">{chunks.length} chunk{chunks.length === 1 ? '' : 's'} · {chunks.map((c) => c.n_tokens).join(' / ')} tokens</span>
                    {meta && <span className="text-xs text-slate-400">{meta.description}</span>}
                  </div>
                  <Highlighted text={p.text} spans={chunks.map((c) => [c.char_start, c.char_end] as [number, number])} />
                </div>
              )
            })}
          </>
        )}
      </div>
    </div>
  )
}

/** Overlapping chunks are drawn as stacked rows so overlaps stay visible. */
function Highlighted({ text, spans }: { text: string; spans: [number, number][] }) {
  const rows: [number, number][][] = []
  spans.forEach((sp) => {
    const row = rows.find((r) => r.every(([s, e]) => sp[0] >= e || sp[1] <= s))
    if (row) row.push(sp); else rows.push([sp])
  })
  return (
    <div className="space-y-1">
      {rows.map((row, ri) => (
        <div key={ri} className="text-sm leading-relaxed text-slate-400">
          {segments(text, row).map((seg, i) => seg.idx == null ? <span key={i}>{seg.t}</span> : <span key={i} className={`rounded px-0.5 text-slate-800 ${PALETTE[spans.indexOf(row[seg.idx]) % PALETTE.length]}`}>{seg.t}</span>)}
        </div>
      ))}
    </div>
  )
}
function segments(text: string, row: [number, number][]) {
  const cuts = [0, ...row.flat(), text.length].sort((a, b) => a - b)
  const out: { t: string; idx: number | null }[] = []
  for (let i = 0; i < cuts.length - 1; i++) {
    const s = cuts[i], e = cuts[i + 1]
    if (e <= s) continue
    const idx = row.findIndex(([a, b]) => s >= a && e <= b)
    out.push({ t: text.slice(s, e), idx: idx >= 0 ? idx : null })
  }
  return out
}
