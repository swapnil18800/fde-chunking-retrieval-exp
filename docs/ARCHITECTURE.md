# Architecture

> Companion docs: [CHUNKING.md](CHUNKING.md) · [RETRIEVAL.md](RETRIEVAL.md) · [EVALUATION.md](EVALUATION.md) · [TRACING.md](TRACING.md) · [HOW_TO_RUN.md](HOW_TO_RUN.md) · [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)

## 1. What this system is

A **controlled experiment harness** for RAG over medical literature. The corpus is
`rag-datasets/rag-mini-bioasq` (PubMed abstracts keyed by PMID; BioASQ questions with gold
passage ids). Every combination of *chunking strategy × retriever* (plus optional query
transform, cross-encoder rerank and context expansion) can be run on the same question,
inspected chunk-by-chunk, and scored against gold — first with exact retrieval metrics (no LLM),
then with RAGAS on the best configurations.

Three design constraints shaped it:

1. **Everything is comparable.** All chunk sets live in one `chunks` table keyed by `strategy`,
   embedded with the same model, retrieved through one `run_retrieval()` entry point. A config is
   a plain dataclass (`RetrievalConfig`), and its label (`sentence+hybrid+rerank+window`) is the
   key used in logs, eval tables and the UI.
2. **Everything is traceable.** One question = one LangGraph run = one `query_logs` row = one
   Langfuse trace. The stage timings the UI shows are the same list the eval harness records.
3. **Everything is (nearly) free.** Local open-source embeddings (`MedEmbed-small`) and reranker
   (`MedCPT`), Gemini free tier for generation/judging (round-robined across models), Supabase
   free tier for storage, Langfuse Hobby tier for traces.

## 2. High-level diagram

```mermaid
flowchart LR
  subgraph Offline["Offline — db/ingestion/"]
    HF[HF dataset<br/>40,221 passages · 4,719 QA] --> LC[load_corpus.py<br/>normalise · tiktoken]
    LC --> SUB[subsample_corpus.py<br/>20k subset, eval150 gold protected]
    SUB --> NLP[preprocess_nlp.py<br/>scispaCy sentences + entities]
    NLP --> BC[build_chunks.py<br/>5 chunkers → MedEmbed → halfvec + HNSW]
  end

  subgraph DB["Supabase Postgres + pgvector"]
    P[(passages)] ; C[(chunks<br/>offsets + halfvec)] ; K[(kg_entities<br/>passage_ids[])] ; Q[(qa_pairs · eval_sets)] ; L[(query_logs)] ; E[(eval_runs · eval_results)]
  end

  subgraph Online["Online — pipeline/"]
    API[FastAPI app/main.py] --> G[LangGraph graph.py]
    G --> R[retrieve-context<br/>runner.run_retrieval]
    R --> T[transform-query<br/>none · hyde · multi_query · decompose]
    T --> RET[retrieve-chunks<br/>bm25 · dense · hybrid · grep · kg]
    RET --> RR[rerank-chunks<br/>MedCPT cross-encoder]
    RR --> EX[expand<br/>none · window · parent]
    G --> GEN[generate-answer<br/>Gemini via OpenAI-compatible API]
    G --> CIT[cite-sources<br/>[n] → chunk → passage → PubMed]
  end

  subgraph Eval["evals/"]
    RE[run_retrieval_eval.py<br/>recall/precision/MRR/nDCG] ; RG[run_ragas_eval.py<br/>faithfulness · relevancy · id-precision/recall · factual]
  end

  UI[React SPA<br/>7 tabs] --> API
  RET --> C ; RET --> P ; RET --> K
  G --> L ; RE --> G ; RG --> G ; RE --> E ; RG --> E
  G -.-> LF[Langfuse / LangSmith]
```

## 3. Data model

| Table | Rows (subset) | Purpose |
|---|---|---|
| `passages` | 20,000 | Normalised abstract text, token counts, `sentence_offsets int[]` (scispaCy), `kg_entity_ids int[]` |
| `qa_pairs` | 4,719 | BioASQ question, reference answer, `relevant_passage_ids` (filtered to surviving corpus), `n_gold_total`, heuristic `question_type` |
| `eval_sets` | eval150 · smoke5 | Named, ordered question subsets (also committed as JSON in `evals/datasets/`) |
| `chunk_strategies` | 5 | Registry: description, params, embedding model, chunk count, avg tokens |
| `chunks` | ~280k | `(strategy, passage_id, chunk_index, char_start, char_end, n_tokens, prefix, embedding halfvec(384))` |
| `chunk_text` (view) | — | `substr(passage.text, char_start+1, len)` — chunk text is never duplicated |
| `kg_entities` | ~49k | Entity surface form, doc_freq, `passage_ids bigint[]` (entity→passage adjacency) |
| `query_logs` | grows | One row per pipeline run: config, stages, retrieved hits, answer, citations, metrics, tokens, trace url |
| `eval_runs` / `eval_results` | grows | Eval run header + per-question metrics; `query_log_id` links each number to its full run |
| `corpus_meta` | 1 | Subsample recipe (seed, sizes) for reproducibility |

**Why offsets instead of text?** ~280k chunks × ~600 B of text would double storage on a
500 MB free tier, and offsets give exact highlight spans for citations for free.

**Why `halfvec`?** 2 bytes/dim halves vector storage with negligible recall loss at 384-d.
HNSW indexes are partial (`where strategy = '…'`); on the free tier only `passage` and `recursive_128_32` are indexed, the rest are exact-scanned (50–200 ms).

## 4. Request lifecycle (one question)

```
POST /api/ask {question, strategy, retriever, transform, rerank, expansion, top_k, candidate_k, qa_id?}
 └─ run_pipeline()                       pipeline/graph.py
     ├─ trace_span("rag-answer")          Langfuse root observation (chain) + propagate_attributes(session, tags)
     ├─ LangGraph "rag-graph"
     │   ├─ retrieve-context              runner.run_retrieval(cfg)
     │   │   ├─ transform-query           LLM rewrite (optional) → 1..5 queries
     │   │   ├─ retrieve-chunks           retriever.retrieve(q, strategy, k) per query → RRF if >1 query
     │   │   ├─ rerank-chunks             MedCPT over candidate_k → top_k (optional)
     │   │   └─ expand                    window / parent context (optional)
     │   ├─ generate-answer               Gemini, numbered context, inline [n] citations
     │   └─ cite-sources                  [n] → {chunk_id, char span, pmid, url}; invalid markers stripped
     ├─ write_query_log()                 Postgres query_logs (best-effort)
     └─ return JSON                       + retrieval metrics if qa_id has gold labels
```

## 5. Component map

| Concern | Entry point | Notes |
|---|---|---|
| Config | `config.py` | pydantic-settings; every env var declared here |
| Logging | `pipeline/logging_setup.py` | `[tag]` console + JSONL per process in `logs/` |
| Chunkers | `pipeline/chunking/strategies.py` | `build_chunker(name)`; `DEFAULT_STRATEGIES` |
| Embeddings | `pipeline/embedder.py` | sentence-transformers, MPS; model recorded per chunk set |
| Retrievers | `pipeline/retrieval/{bm25,dense,hybrid,grep,kg}.py` | all return `Hit`s for one strategy |
| Transforms / expansion / rerank | `pipeline/retrieval/transforms.py`, `expand.py`, `pipeline/reranker.py` | composable |
| Runner | `pipeline/retrieval/runner.py` | `RetrievalConfig`, `run_retrieval()`, stage timings |
| Graph | `pipeline/graph.py` | LangGraph nodes, `run_pipeline()` |
| Prompts | `pipeline/prompts.py` (+ transform prompts in `transforms.py`) | never inline elsewhere |
| LLM | `pipeline/llm.py` | one OpenAI-compatible client, Gemini round-robin, 429 retryDelay |
| Tracing | `pipeline/tracing.py` | `TRACING_PROVIDER=langfuse|langsmith|off` |
| Query log | `pipeline/query_log.py` | write/fetch `query_logs` |
| NLP | `pipeline/nlp.py` | scispaCy loader (config fix for spaCy 3.8), entity cleaning |
| API | `app/main.py`, `app/schemas.py` | FastAPI; serves `frontend/dist` in prod |
| UI | `frontend/src/pages/*` | Overview, Playground, Compare, Leaderboard, Chunk Explorer, Knowledge Graph, Query Inspector |
| Evals | `evals/metrics.py`, `evals/run_retrieval_eval.py`, `evals/run_ragas_eval.py` | Tier 1 / Tier 2 |

## 6. Free-tier storage budget

| Item | Size |
|---|---|
| passages (20k) + qa + KG arrays | ~55 MB |
| chunks: 5 strategies = 317k × (768 B halfvec + ~60 B row) | ~260 MB (heap) |
| HNSW on `passage` (20k) + `recursive_128_32` (57k) | ~80 MB |
| `fixed_128_32`, `sentence`, `semantic` — exact scan, no index | included above |
| logs / eval results | ~10–20 MB per full matrix run |
| **measured after build + vacuum** | **~430 MB** (488 MB before dropping the `fixed` index) |

The 40k → 20k subsample (`corpus_meta.subsample`) is what makes this fit; the recipe keeps every
gold passage of `eval150`, adds gold passages of other BioASQ questions as topically-close
distractors, then random passages.
