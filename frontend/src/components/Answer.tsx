import { useState } from 'react'
import { ExternalLink } from 'lucide-react'
import type { AskResult, Citation, Hit } from '../lib/api'
import { fmt, pubmed } from '../lib/api'
import PassageView from './PassageView'

/** Answer with clickable [n] citations → chunk → parent passage → PubMed. */
export default function Answer({ r }: { r: AskResult }) {
  const [active, setActive] = useState<Citation | null>(null)
  const parts = r.answer.split(/(\[\d{1,2}\])/g)
  return (
    <div className="space-y-4">
      <div className="card p-5">
        <div className="mb-2 flex items-center justify-between">
          <div className="label">Answer <span className="ml-2 font-normal normal-case text-slate-400">{String(r.tokens?.model ?? '')} · {r.latency_ms} ms</span></div>
          {r.trace_url && <a className="text-xs text-accent hover:underline" href={r.trace_url} target="_blank" rel="noreferrer">open trace ({r.trace_provider}) <ExternalLink className="inline" size={11} /></a>}
        </div>
        {r.error && <div className="rounded bg-red-50 p-3 text-sm text-red-700">{r.error}</div>}
        <p className="text-[15px] leading-relaxed">
          {parts.map((p, i) => {
            const m = p.match(/^\[(\d{1,2})\]$/)
            if (!m) return <span key={i}>{p}</span>
            const c = r.citations.find((x) => x.n === Number(m[1]))
            return c ? (
              <button key={i} onClick={() => setActive(c)} title={`PMID ${c.pmid}`}
                className={`mx-0.5 rounded px-1.5 text-xs font-semibold ${active?.n === c.n ? 'bg-gold text-white' : 'bg-gold-soft text-gold hover:bg-amber-200'}`}>{c.n}</button>
            ) : <span key={i} className="text-slate-400">{p}</span>
          })}
        </p>
      </div>
      {active && (
        <div className="card p-5">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm font-semibold">Citation [{active.n}] → chunk #{active.chunk_id} → PMID {active.pmid}</div>
            <div className="flex gap-2 text-xs">
              <span className="chip border-slate-200">score {fmt(active.score)}</span>
              <span className="chip border-slate-200">via {active.source}</span>
              <a className="chip border-accent/30 text-accent hover:bg-accent-soft" href={active.url} target="_blank" rel="noreferrer">PubMed <ExternalLink size={11} /></a>
            </div>
          </div>
          <PassageView pmid={active.pmid} highlight={{ start: active.char_start, end: active.char_end }}
            context={{ start: active.context_start, end: active.context_end }} />
        </div>
      )}
      <Retrieved hits={r.retrieved} gold={r.gold_passage_ids} />
    </div>
  )
}

export function Retrieved({ hits, gold, compact = false }: { hits: Hit[]; gold?: number[]; compact?: boolean }) {
  const [openId, setOpenId] = useState<number | null>(null)
  return (
    <div className="card p-5">
      <div className="label mb-3">Retrieved chunks ({hits.length}){gold?.length ? <span className="ml-2 font-normal normal-case text-slate-400">gold = BioASQ relevant PMIDs</span> : null}</div>
      <ol className="space-y-2">
        {hits.map((h) => (
          <li key={h.chunk_id} className={`rounded-lg border p-3 text-sm ${h.is_gold ? 'border-emerald-300 bg-emerald-50/50' : 'border-slate-200'}`}>
            <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span className="font-semibold text-slate-700">#{h.rank}</span>
              {h.is_gold != null && <span className={`chip ${h.is_gold ? 'border-emerald-300 bg-emerald-100 text-emerald-800' : 'border-slate-200'}`}>{h.is_gold ? 'gold' : 'not gold'}</span>}
              <a className="hover:underline" href={pubmed(h.passage_id)} target="_blank" rel="noreferrer">PMID {h.passage_id}</a>
              <span>chunk {h.chunk_index} · chars {h.char_start}–{h.char_end}</span>
              <span>score {fmt(h.score)}</span>
              <span className="chip border-slate-200">{h.source}</span>
              {Object.entries(h.retriever_ranks ?? {}).map(([k, v]) => <span key={k} className="chip border-slate-200">{k} #{v}</span>)}
              {typeof h.meta?.entities === 'object' && <span className="chip border-violet-200 bg-violet-50 text-violet-700">kg: {(h.meta.entities as string[]).slice(0, 4).join(', ')}</span>}
              {Array.isArray(h.meta?.terms) && <span className="chip border-sky-200 bg-sky-50 text-sky-700">grep: {(h.meta.terms as string[]).join(', ')}</span>}
              <button className="ml-auto text-accent hover:underline" onClick={() => setOpenId(openId === h.chunk_id ? null : h.chunk_id)}>{openId === h.chunk_id ? 'hide parent' : 'show parent passage'}</button>
            </div>
            {!compact && <p className="text-slate-700">{h.context_text ?? h.text}</p>}
            {openId === h.chunk_id && <div className="mt-2 border-t pt-2"><PassageView pmid={h.passage_id} highlight={{ start: h.char_start, end: h.char_end }} /></div>}
          </li>
        ))}
      </ol>
    </div>
  )
}
