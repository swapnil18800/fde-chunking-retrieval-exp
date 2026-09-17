import { useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { api, type AskResult } from '../lib/api'
import { useOptions } from '../lib/hooks'
import ConfigPicker, { DEFAULT_CFG } from '../components/ConfigPicker'
import QuestionPicker from '../components/QuestionPicker'
import Answer from '../components/Answer'
import Stages from '../components/Stages'
import Metrics from '../components/Metrics'

export default function Playground() {
  const opts = useOptions()
  const [cfg, setCfg] = useState(DEFAULT_CFG)
  const [q, setQ] = useState('Is RANKL secreted from the cells?')
  const [qaId, setQaId] = useState<number | null>(null)
  const [generate, setGenerate] = useState(true)
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState<AskResult | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const run = async () => {
    setBusy(true); setErr(null)
    try { setRes(await api.ask({ question: q, qa_id: qaId, generate, ...cfg })) } catch (e) { setErr(String(e)) } finally { setBusy(false) }
  }
  return (
    <div className="space-y-4">
      <div className="card space-y-4 p-5">
        <QuestionPicker value={q} qaId={qaId} onPick={(t, id) => { setQ(t); setQaId(id) }} />
        <ConfigPicker cfg={cfg} onChange={setCfg} opts={opts} />
        <div className="flex items-center gap-4">
          <button className="btn-primary" onClick={run} disabled={busy || q.length < 3}>{busy ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />} Run</button>
          <label className="flex items-center gap-2 text-sm text-slate-600"><input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} /> generate answer (LLM)</label>
          {opts && <span className="text-xs text-slate-400">embeddings {opts.embedding_model} · LLM {opts.llm_provider}: {opts.llm_models.join(', ')}</span>}
        </div>
        {err && <div className="rounded bg-red-50 p-3 text-sm text-red-700">{err}</div>}
      </div>
      {res && (
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            {res.metrics && <Metrics m={res.metrics} />}
            <Answer r={res} />
          </div>
          <div className="space-y-4">
            <Stages stages={res.stages} queries={res.queries} hypothetical={res.hypothetical} />
            <div className="card p-4 text-xs text-slate-500">
              <div className="label mb-1">Run</div>
              <div>id <span className="font-mono">{res.id}</span></div>
              <div>config <span className="font-mono">{JSON.stringify(res.config)}</span></div>
              <div className="mt-1">Every run is persisted to <span className="font-mono">query_logs</span> — see Query Inspector.</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
