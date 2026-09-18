# BioASQ RAG Lab — chunking × retrieval experiments on medical literature

**Which way of chunking PubMed abstracts, and which way of retrieving them, actually surfaces the
right evidence?** This repo answers that with a controlled matrix — **5 chunking strategies × 5
retrievers**, plus query transforms, a PubMed-trained cross-encoder reranker and context expansion
— evaluated on `rag-datasets/rag-mini-bioasq` (PubMed abstracts + BioASQ questions with gold
passage ids), first with exact retrieval metrics and then with RAGAS. Everything runs on free
tiers: Supabase pgvector, Gemini, Langfuse, and local open-source models on a laptop.

> Results: see [§ Results](#results) and [docs/EVALUATION.md](docs/EVALUATION.md).

---

## What's inside

| Axis | Options | Notes |
|---|---|---|
| **Chunking** | `passage` · `fixed_128_32` · `recursive_128_32` · `sentence` · `semantic` | char-offset chunks into the abstract; one embedding model for all ([docs/CHUNKING.md](docs/CHUNKING.md)) |
| **Retrieval** | `bm25` · `dense` · `hybrid` (RRF) · `grep` (literal terms) · `kg` (entity graph) | all return chunk hits for one strategy ([docs/RETRIEVAL.md](docs/RETRIEVAL.md)) |
| **Query transform** | `hyde` · `multi_query` · `decompose` (multi-hop) | Gemini free tier, RRF-fused with the original question |
| **Rerank** | `ncbi/MedCPT-Cross-Encoder` | PubMed click-log trained cross-encoder, local |
| **Expansion** | `window` (sentence-window) · `parent` (small-to-big) | changes what the generator sees, not what was retrieved |
| **Generation** | Gemini via OpenAI-compatible API, round-robined across models | cited `[n]` answers; citations resolve chunk → parent passage → PubMed |
| **Evaluation** | Tier 1: recall/precision/MRR/nDCG vs gold PMIDs · Tier 2: RAGAS | every eval number links to a full logged run |
| **Observability** | Langfuse (default) / LangSmith / off · `query_logs` table · JSONL logs | one question = one trace = one log row ([docs/TRACING.md](docs/TRACING.md)) |
| **UI** | Overview · Playground · Compare · Leaderboard · Chunk Explorer · Knowledge Graph · Query Inspector | React + Vite + Tailwind |

## Architecture

```
question ─▶ transform (none|hyde|multi_query|decompose) ─▶ retriever(strategy) ─▶ rerank (MedCPT) ─▶ expand (window|parent)
        ─▶ generate (Gemini, numbered context) ─▶ cite ([n] → chunk span → parent PMID → PubMed) ─▶ query_logs + Langfuse trace
```

Full diagram, data model and component map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI · LangGraph · psycopg3 · Python 3.12 (`uv`) |
| Database | Supabase Postgres 17 + pgvector 0.8 (`halfvec`, partial HNSW per strategy) |
| Embeddings | `abhinand/MedEmbed-small-v0.1` (bge-small fine-tuned on medical retrieval, 384-d) — local, MPS |
| Reranker | `ncbi/MedCPT-Cross-Encoder` — local |
| Lexical | `bm25s` (Lucene BM25) with Snowball stemming |
| NLP | spaCy 3.8 + scispaCy `en_core_sci_sm` (sentences, entities for the knowledge graph) |
| LLM | Gemini free tier (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.5-flash`), DeepSeek as paid fallback |
| Evaluation | custom passage-level metrics · RAGAS 0.4 |
| Observability | Langfuse Cloud (SDK v4) · LangSmith (validation only) |
| Frontend | React 19 · Vite · TypeScript · Tailwind v4 · recharts |

## Corpus

| | |
|---|---|
| Source | [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq): 40,221 PubMed abstracts (PMID-keyed), 4,719 BioASQ QA pairs with `relevant_passage_ids` |
| Used | **20,000 passages** — Supabase free-tier subset: every gold passage of `eval150` + gold passages of other questions (hard distractors) + random (seed 42, recorded in `corpus_meta`) |
| Passage shape | avg 147 words / 225 tokens / 6.7 sentences; no section headers |
| Questions | `eval150` (stratified factoid / list / yes-no / summary, 1–20 gold each) · `smoke5 ⊂ eval150` |
| Chunks | ~280k across 5 strategies, stored as char offsets + `halfvec(384)` |

## Results

**Tier 1 — retrieval on `eval150` (150 BioASQ questions, recall@10 vs gold PMIDs, no rerank)**

| strategy         |   bm25 |   dense |   grep |   hybrid |    kg |
|:-----------------|-------:|--------:|-------:|---------:|------:|
| fixed_128_32     |  0.485 |   0.453 |  0.447 |    0.497 | 0.346 |
| passage          |  0.533 |   0.46  |  0.447 |    0.533 | 0.346 |
| recursive_128_32 |  0.478 |   0.449 |  0.447 |    0.497 | 0.346 |
| semantic         |  0.479 |   0.468 |  0.447 |    0.495 | 0.346 |
| sentence         |  0.446 |   0.471 |  0.447 |    0.484 | 0.346 |

- **Cutting below the abstract costs ~4 recall points, and *how* you cut barely matters** (fixed 0.497 · recursive 0.497 · semantic 0.495 · sentence 0.484 with hybrid). Gold labels are whole abstracts, so `passage` has a structural edge — precision@5 is flat (0.49–0.53) across strategies.
- **Chunk length flips the winner between lexical and dense**: BM25 wins on whole abstracts (0.533 vs 0.460), dense wins on single sentences (0.471 vs 0.446). Entity-heavy medical questions favour exact tokens against long text; a small 384-d embedder represents one sentence better than one abstract.
- **Hybrid RRF is never worse than its parts**; `grep` (literal terms) trails BM25 by 9 points — the measured value of stemming + IDF; `kg` (entity co-occurrence) is weakest (0.346) but the only retriever that explains its hits.
- Yes/no questions are easiest (0.59), list questions hardest (0.48): their 8.5 gold abstracts don't fit in top-10.
- **MedCPT cross-encoder reranking lifts every cell** (recall@10 +0.023 avg, MRR → 0.79–0.81 across the board) and shrinks the chunking gap; best overall: `passage+bm25+rerank` **0.568 recall@10 / 0.552 precision@5 / 0.803 MRR**; best sub-abstract: `recursive_128_32+bm25+rerank` 0.515.

Rerank / expansion / query-transform sweeps and RAGAS answer quality: [docs/EVALUATION.md](docs/EVALUATION.md).

## Quick start

```bash
git clone https://github.com/swapnil18800/fde-chunking-retrieval-exp.git && cd fde-chunking-retrieval-exp
uv sync && (cd frontend && npm install)
cp .env.example .env                       # DATABASE_URL, GEMINI_API_KEY, LANGFUSE_* …

# data (once) — see docs/HOW_TO_RUN.md for the full order and timings
uv run python db/setup_db.py && uv run python db/ingestion/load_corpus.py && uv run python evals/build_eval_sets.py
uv run python db/ingestion/preprocess_nlp.py && uv run python db/ingestion/subsample_corpus.py --target 20000 --yes
uv run python db/ingestion/build_chunks.py --all

# run
uv run uvicorn app.main:app --reload --port 8000      # API
cd frontend && npm run dev                            # UI → http://localhost:5173

# evaluate
uv run python evals/run_retrieval_eval.py --set eval150 --matrix
uv run python evals/run_ragas_eval.py --set eval150 --limit 50 --configs sentence+hybrid+rerank+window passage+hybrid+rerank
```

## Documentation

| Doc | What |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, request lifecycle, data model, storage budget |
| [CHUNKING.md](docs/CHUNKING.md) | The five strategies, why these, implementation notes, how to add one |
| [RETRIEVAL.md](docs/RETRIEVAL.md) | The five retrievers + transforms / rerank / expansion, multi-hop |
| [EVALUATION.md](docs/EVALUATION.md) | Metrics, methodology, results history |
| [TRACING.md](docs/TRACING.md) | Langfuse structure, LangSmith validation, logs, how to read a question's story |
| [HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | Setup, data build, run, evaluate, troubleshooting |
| [DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md) | File-level map |
| [CLAUDE.md](CLAUDE.md) + [.claude/skills/](.claude/skills) | Working rules and runbooks for Claude Code (navigate, debug a question, run experiments, add strategy, tracing) |

## References

- MultiHop-RAG — [yixuantt/MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) (multi-hop retrieval evaluation; inspiration for `decompose`)
- MedCPT — Jin et al., *Contrastive Pre-trained Transformers with large-scale PubMed search logs* (2023)
- MedEmbed — [abhinand/MedEmbed-small-v0.1](https://huggingface.co/abhinand/MedEmbed-small-v0.1)
- Sibling projects by the same author: [AlphaLens](https://github.com/swapnil18800/alphalens) (SEC filings RAG), [WebLens](https://github.com/swapnil18800/weblens) (web search RAG)

## License

MIT — see [LICENSE](LICENSE).
