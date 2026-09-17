import { Link } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { useOptions } from '../lib/hooks'

const CHUNKERS = [
  ['passage', 'whole abstract (~225 tok)', 'baseline — same granularity as the gold labels'],
  ['fixed_128_32', '128-token windows, 32 overlap', 'the naive default; ignores sentence boundaries'],
  ['recursive_128_32', 'LangChain recursive splitter', 'sentence → clause → word aware version of fixed'],
  ['sentence', 'one scispaCy sentence (~30 tok)', 'finest granularity; pairs with window / parent expansion'],
  ['semantic', 'embedding-similarity breakpoints', 'topic-coherent sentence groups (global threshold)'],
]
const RETRIEVERS = [
  ['bm25', 'lexical', 'Lucene BM25 (bm25s), stemmed, per-strategy index'],
  ['dense', 'semantic', 'MedEmbed-small 384-d cosine in pgvector (HNSW)'],
  ['hybrid', 'lexical + semantic', 'RRF fusion of BM25 and dense'],
  ['grep', 'literal', 'entity/term regex over passages — the agentic "grep" baseline'],
  ['kg', 'graph', 'scispaCy entity–passage graph, 1-hop expansion, dense tie-break'],
]
const EXTRAS = [
  ['transforms', 'hyde · multi_query · decompose', 'LLM query rewriting (Gemini free tier) — HyDE, paraphrase union, multi-hop decomposition'],
  ['rerank', 'ncbi/MedCPT-Cross-Encoder', 'PubMed-trained cross-encoder over the top-N candidates'],
  ['expansion', 'window · parent', 'sentence-window and parent-document (small-to-big) context'],
]

export default function Overview() {
  const opts = useOptions()
  return (
    <div className="space-y-6">
      <section className="card p-6">
        <h1 className="text-2xl font-semibold">How should medical abstracts be chunked and retrieved?</h1>
        <p className="mt-2 max-w-3xl text-slate-600">
          A controlled experiment on <b>rag-mini-bioasq</b> (PubMed abstracts + BioASQ questions with gold passage ids).
          Every combination of <b>5 chunking strategies × 5 retrievers</b> (+ query transforms, reranking, context expansion)
          is scored two ways: exact retrieval metrics against gold PMIDs (no LLM), then RAGAS answer-quality metrics on the best configs.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-sm">
          <Link className="btn-primary" to="/playground">Ask a question <ArrowRight size={14} /></Link>
          <Link className="btn-ghost" to="/compare">Compare configs</Link>
          <Link className="btn-ghost" to="/leaderboard">See the leaderboard</Link>
          <Link className="btn-ghost" to="/chunks">Watch a passage get chunked</Link>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Card title="1 · Chunking" rows={CHUNKERS} note={opts ? `${opts.strategies.reduce((a, s) => a + s.n_chunks, 0).toLocaleString()} chunks stored as char offsets + halfvec` : ''} />
        <Card title="2 · Retrieval" rows={RETRIEVERS} note="all return chunk hits for one strategy; metrics collapse to parent PMID" />
        <Card title="3 · Extras" rows={EXTRAS} note="composable on top of any (chunking, retriever) cell" />
      </section>

      <section className="card p-6">
        <div className="label mb-3">Pipeline</div>
        <pre className="overflow-x-auto rounded-lg bg-slate-900 p-4 text-xs leading-relaxed text-slate-100">{`question
  └─ transform (none | hyde | multi_query | decompose)           LangGraph node: retrieve
       └─ retriever(strategy) ──▶ candidates  ─┬─ rerank (MedCPT) ─┐
                                              └──────────────────┴─▶ expand (none | window | parent)
  └─ generate (Gemini, cited [n] answer)                          LangGraph node: generate
  └─ cite: [n] → chunk span → parent passage → PubMed URL         LangGraph node: cite
  └─ persist: query_logs row + Langfuse trace                     every run, every source (api / eval)`}</pre>
        <div className="mt-3 grid gap-3 text-sm text-slate-600 md:grid-cols-4">
          <Fact k="Corpus" v="20,000 PubMed abstracts (free-tier subset; all eval150 gold kept)" />
          <Fact k="Questions" v="eval150 stratified over factoid / list / yes-no / summary; smoke5 ⊂ eval150" />
          <Fact k="Embeddings" v={opts?.embedding_model ?? 'MedEmbed-small-v0.1'} />
          <Fact k="LLM" v={opts ? `${opts.llm_provider}: ${opts.llm_models.join(', ')}` : 'Gemini free tier'} />
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <div className="card p-6">
          <div className="label mb-2">Tier 1 — retrieval metrics (no LLM)</div>
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
            <li><b>recall@k</b> — fraction of gold passages in the top-k (evidence surfaced?)</li>
            <li><b>precision@k</b> — how much of what the generator sees is gold</li>
            <li><b>MRR / nDCG@k</b> — how high the first / all gold passages rank</li>
            <li>Chunk hits are collapsed to their parent PMID first, so fine-grained strategies aren't rewarded for duplicates.</li>
          </ul>
        </div>
        <div className="card p-6">
          <div className="label mb-2">Tier 2 — RAGAS (LLM-judged, top configs only)</div>
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
            <li><b>faithfulness</b> — claims supported by retrieved context</li>
            <li><b>answer relevancy</b> — does the answer address the question</li>
            <li><b>context precision / recall</b> — id-based against gold PMIDs (exact)</li>
            <li><b>factual correctness</b> — vs the BioASQ reference answer</li>
          </ul>
        </div>
      </section>
    </div>
  )
}

function Card({ title, rows, note }: { title: string; rows: string[][]; note: string }) {
  return (
    <div className="card p-5">
      <div className="mb-3 text-sm font-semibold">{title}</div>
      <ul className="space-y-2">
        {rows.map(([n, u, why]) => (
          <li key={n} className="text-sm"><span className="font-mono text-xs font-semibold text-accent">{n}</span> <span className="text-slate-500">— {u}</span><div className="text-xs text-slate-500">{why}</div></li>
        ))}
      </ul>
      {note && <div className="mt-3 border-t pt-2 text-xs text-slate-400">{note}</div>}
    </div>
  )
}
function Fact({ k, v }: { k: string; v: string }) { return <div><div className="label">{k}</div><div>{v}</div></div> }
