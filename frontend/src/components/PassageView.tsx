import { api } from '../lib/api'
import { useAsync } from '../lib/hooks'

/** Full parent passage with the cited chunk highlighted (and the wider expansion context, if any). */
export default function PassageView({ pmid, highlight, context }:
  { pmid: number; highlight: { start: number; end: number }; context?: { start: number; end: number } }) {
  const { data, loading } = useAsync(() => api.passage(pmid), [pmid])
  if (loading || !data) return <div className="text-xs text-slate-400">loading passage…</div>
  const t = data.text
  const ctx = context && (context.start !== highlight.start || context.end !== highlight.end) ? context : null
  const cuts = [...new Set([0, highlight.start, highlight.end, ctx?.start ?? 0, ctx?.end ?? 0, t.length])].sort((a, b) => a - b)
  return (
    <div className="text-sm leading-relaxed text-slate-700">
      <div className="mb-1 text-xs text-slate-400">PMID {pmid} · {data.n_tokens} tokens · <mark className="cite">cited chunk</mark>{ctx && <> · <mark className="chunk">expanded context</mark></>}</div>
      {cuts.slice(0, -1).map((s, i) => {
        const e = cuts[i + 1]
        const seg = t.slice(s, e)
        if (s >= highlight.start && e <= highlight.end) return <mark key={i} className="cite">{seg}</mark>
        if (ctx && s >= ctx.start && e <= ctx.end) return <mark key={i} className="chunk">{seg}</mark>
        return <span key={i}>{seg}</span>
      })}
    </div>
  )
}
