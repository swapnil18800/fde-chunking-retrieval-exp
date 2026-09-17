import { fmt } from '../lib/api'

const ORDER = ['recall@5', 'recall@10', 'precision@5', 'mrr', 'ndcg@10', 'hit@5']
export default function Metrics({ m }: { m: Record<string, number> | null | undefined }) {
  if (!m) return null
  return (
    <div className="card p-4">
      <div className="label mb-2">Retrieval metrics vs BioASQ gold ({m.n_gold} gold passages)</div>
      <div className="grid grid-cols-3 gap-2 md:grid-cols-6">
        {ORDER.filter((k) => k in m).map((k) => (
          <div key={k} className="rounded-lg bg-slate-50 p-2 text-center">
            <div className="text-lg font-semibold tabular-nums">{fmt(m[k], 2)}</div>
            <div className="text-[11px] text-slate-500">{k}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
