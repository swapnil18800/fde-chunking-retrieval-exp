import { useState } from 'react'
import { Loader2, Plus, Trash2 } from 'lucide-react'
import { api, fmt, type AskResult } from '../lib/api'
import { useOptions } from '../lib/hooks'
import ConfigPicker, { DEFAULT_CFG, labelOf } from '../components/ConfigPicker'
import QuestionPicker from '../components/QuestionPicker'
import { Retrieved } from '../components/Answer'

const PRESETS = [
  { ...DEFAULT_CFG, strategy: 'passage', retriever: 'bm25' },
  { ...DEFAULT_CFG, strategy: 'passage', retriever: 'dense' },
  { ...DEFAULT_CFG, strategy: 'sentence', retriever: 'hybrid', rerank: true, expansion: 'window' },
]

export default function Compare() {
  const opts = useOptions()
  const [cfgs, setCfgs] = useState(PRESETS)
  const [q, setQ] = useState('Which genes are involved in Hirschsprung disease?')
  const [qaId, setQaId] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState<{ gold_passage_ids: number[]; results: AskResult[] } | null>(null)
  const [generate, setGenerate] = useState(false)
  const run = async () => {
    setBusy(true)
    try { setOut(await api.compare({ question: q, configs: cfgs, qa_id: qaId, generate })) } finally { setBusy(false) }
  }
  const gold = new Set(out?.gold_passage_ids ?? [])
  return (
    <div className="space-y-4">
      <div className="card space-y-4 p-5">
        <QuestionPicker value={q} qaId={qaId} onPick={(t, id) => { setQ(t); setQaId(id) }} />
        <div className="space-y-3">
          {cfgs.map((c, i) => (
            <div key={i} className="flex items-start gap-3 rounded-lg border border-slate-200 p-3">
              <div className="mt-1 w-6 text-center text-sm font-semibold text-slate-400">{String.fromCharCode(65 + i)}</div>
              <div className="flex-1"><ConfigPicker cfg={c} onChange={(n) => setCfgs(cfgs.map((x, j) => (j === i ? n : x)))} opts={opts} /></div>
              <button className="btn-ghost p-2" onClick={() => setCfgs(cfgs.filter((_, j) => j !== i))} disabled={cfgs.length === 1}><Trash2 size={14} /></button>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-3">
          <button className="btn-ghost" onClick={() => setCfgs([...cfgs, { ...DEFAULT_CFG }])} disabled={cfgs.length >= 6}><Plus size={14} /> add config</button>
          <button className="btn-primary" onClick={run} disabled={busy}>{busy && <Loader2 className="animate-spin" size={16} />} Compare {cfgs.length} configs</button>
          <label className="flex items-center gap-2 text-sm text-slate-600"><input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} /> also generate answers</label>
        </div>
      </div>
      {out && (
        <>
          <div className="card overflow-x-auto p-5">
            <div className="label mb-3">Side by side {gold.size ? `· ${gold.size} gold passages` : '· pick a BioASQ question to see gold overlap'}</div>
            <table className="w-full text-sm">
              <thead><tr className="text-left text-xs text-slate-500"><th className="py-1">config</th><th>recall@5</th><th>recall@10</th><th>precision@5</th><th>mrr</th><th>ndcg@10</th><th>latency</th><th>top passages (green = gold)</th></tr></thead>
              <tbody>
                {out.results.map((r, i) => (
                  <tr key={i} className="border-t">
                    <td className="py-2 font-mono text-xs">{String.fromCharCode(65 + i)} · {r.label ?? labelOf(r.config)}</td>
                    <td>{fmt(r.metrics?.['recall@5'], 2)}</td><td>{fmt(r.metrics?.['recall@10'], 2)}</td><td>{fmt(r.metrics?.['precision@5'], 2)}</td>
                    <td>{fmt(r.metrics?.mrr, 2)}</td><td>{fmt(r.metrics?.['ndcg@10'], 2)}</td><td>{r.latency_ms} ms</td>
                    <td className="flex flex-wrap gap-1 py-2">
                      {[...new Set(r.retrieved.map((h) => h.passage_id))].slice(0, 10).map((p) => (
                        <a key={p} href={`https://pubmed.ncbi.nlm.nih.gov/${p}/`} target="_blank" rel="noreferrer"
                          className={`chip ${gold.has(p) ? 'border-emerald-300 bg-emerald-100 text-emerald-800' : 'border-slate-200 text-slate-500'}`}>{p}</a>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
            {out.results.map((r, i) => (
              <div key={i} className="space-y-2">
                <div className="text-sm font-semibold">{String.fromCharCode(65 + i)} · {r.label ?? labelOf(r.config)}</div>
                {r.answer && <div className="card p-3 text-sm">{r.answer}</div>}
                <Retrieved hits={r.retrieved} gold={out.gold_passage_ids} compact />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
