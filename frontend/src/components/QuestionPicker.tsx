import { useEffect, useState } from 'react'
import { api, type Question } from '../lib/api'

/** Search the eval questions (with gold labels) or type your own. */
export default function QuestionPicker({ value, qaId, onPick }:
  { value: string; qaId: number | null; onPick: (q: string, qaId: number | null) => void }) {
  const [set, setSet] = useState<'smoke5' | 'eval150' | 'all'>('eval150')
  const [rows, setRows] = useState<Question[]>([])
  const [open, setOpen] = useState(false)
  useEffect(() => { api.questions(set, value.length > 2 && qaId == null ? value : '').then(setRows).catch(() => setRows([])) }, [set, value, qaId])
  return (
    <div className="relative">
      <div className="mb-1 flex items-center justify-between">
        <span className="label">Question {qaId != null && <span className="chip ml-2 border-gold/30 bg-gold-soft text-gold">BioASQ #{qaId} · has gold labels</span>}</span>
        <div className="flex gap-1 text-xs">
          {(['smoke5', 'eval150', 'all'] as const).map((s) => (
            <button key={s} onClick={() => setSet(s)} className={`rounded px-2 py-0.5 ${set === s ? 'bg-slate-800 text-white' : 'text-slate-500 hover:bg-slate-100'}`}>{s}</button>
          ))}
        </div>
      </div>
      <textarea className="input min-h-[60px] resize-y" value={value} placeholder="Ask a biomedical question, or pick a BioASQ question below…"
        onFocus={() => setOpen(true)} onChange={(e) => { onPick(e.target.value, null) }} />
      {open && (
        <div className="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-slate-200 bg-white shadow-lg">
          <div className="flex items-center justify-between border-b px-3 py-1.5 text-xs text-slate-500"><span>{rows.length} questions in {set}</span><button onClick={() => setOpen(false)}>close</button></div>
          {rows.map((r) => (
            <button key={r.id} onClick={() => { onPick(r.question, r.id); setOpen(false) }}
              className="block w-full border-b border-slate-100 px-3 py-2 text-left text-sm hover:bg-slate-50">
              <span className="chip mr-2 border-slate-200 text-slate-500">{r.question_type}</span>{r.question}
              <span className="ml-2 text-xs text-slate-400">{r.n_gold} gold</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
