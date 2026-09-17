import type { AskConfig, Options } from '../lib/api'

const HELP: Record<string, string> = {
  bm25: 'Lexical: Lucene BM25 over stemmed tokens (bm25s, in-process index per strategy).',
  dense: 'Semantic: cosine over MedEmbed-small (384-d) embeddings in pgvector (HNSW).',
  hybrid: 'Dense ∪ BM25 fused with reciprocal-rank fusion (k=60).',
  grep: 'Literal term/regex matching over passages (entity + content words), the way an agent with a grep tool searches.',
  kg: 'Entity graph: query entities → passages via the scispaCy entity–passage graph, 1-hop expansion, dense tie-break.',
  none: 'No query transform / no expansion.',
  hyde: 'HyDE: LLM drafts a hypothetical abstract; retrieve with it and the question, fuse by RRF.',
  multi_query: 'LLM writes 3 paraphrases (synonyms, angles); union fused by RRF.',
  decompose: 'Multi-hop style: split into atomic sub-questions, retrieve each, fuse.',
  window: 'Sentence-window: return the chunk plus ±1 neighbouring chunks of the same passage.',
  parent: 'Parent-document: return the whole abstract of every hit (deduplicated).',
}

export default function ConfigPicker({ cfg, onChange, opts, compact = false }:
  { cfg: AskConfig; onChange: (c: AskConfig) => void; opts: Options | null; compact?: boolean }) {
  const set = (k: keyof AskConfig, v: unknown) => onChange({ ...cfg, [k]: v })
  const strat = opts?.strategies.find((s) => s.name === cfg.strategy)
  return (
    <div className={`grid gap-3 ${compact ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-4 lg:grid-cols-7'}`}>
      <Field label="Chunking" help={strat?.description}>
        <select className="input" value={cfg.strategy} onChange={(e) => set('strategy', e.target.value)}>
          {(opts?.strategies ?? [{ name: cfg.strategy } as never]).map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
        </select>
      </Field>
      <Field label="Retriever" help={HELP[cfg.retriever]}>
        <select className="input" value={cfg.retriever} onChange={(e) => set('retriever', e.target.value)}>
          {(opts?.retrievers ?? [cfg.retriever]).map((r) => <option key={r}>{r}</option>)}
        </select>
      </Field>
      <Field label="Query transform" help={HELP[cfg.transform]}>
        <select className="input" value={cfg.transform} onChange={(e) => set('transform', e.target.value)}>
          {(opts?.transforms ?? ['none']).map((r) => <option key={r}>{r}</option>)}
        </select>
      </Field>
      <Field label="Expansion" help={HELP[cfg.expansion]}>
        <select className="input" value={cfg.expansion} onChange={(e) => set('expansion', e.target.value)}>
          {(opts?.expansions ?? ['none']).map((r) => <option key={r}>{r}</option>)}
        </select>
      </Field>
      <Field label="Rerank" help={cfg.rerank ? `Cross-encoder ${opts?.reranker_model ?? ''} over the top-${cfg.candidate_k} candidates.` : 'Off: retriever order is final.'}>
        <label className="input flex cursor-pointer items-center gap-2">
          <input type="checkbox" checked={cfg.rerank} onChange={(e) => set('rerank', e.target.checked)} /> MedCPT
        </label>
      </Field>
      <Field label="top_k">
        <input className="input" type="number" min={1} max={20} value={cfg.top_k} onChange={(e) => set('top_k', Number(e.target.value))} />
      </Field>
      <Field label="candidates">
        <input className="input" type="number" min={1} max={100} value={cfg.candidate_k} onChange={(e) => set('candidate_k', Number(e.target.value))} />
      </Field>
    </div>
  )
}

function Field({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  return (
    <div title={help}>
      <div className="label mb-1">{label}</div>
      {children}
    </div>
  )
}

export const DEFAULT_CFG: AskConfig = { strategy: 'passage', retriever: 'hybrid', transform: 'none', rerank: false, expansion: 'none', top_k: 5, candidate_k: 20 }
export const labelOf = (c: AskConfig) =>
  [c.strategy, c.retriever, c.transform !== 'none' && c.transform, c.rerank && 'rerank', c.expansion !== 'none' && c.expansion].filter(Boolean).join('+')
