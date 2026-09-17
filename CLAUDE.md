# fde-chunking-retrieval-exp — Claude Code instructions

## Project
Chunking × retrieval experiment lab for medical RAG on `rag-datasets/rag-mini-bioasq`
(PubMed abstracts, BioASQ questions with gold PMIDs). 5 chunkers × 5 retrievers (+ query transforms,
MedCPT rerank, window/parent expansion) → LangGraph answer with `[n]` citations → two-tier evals
(exact retrieval metrics, then RAGAS). Supabase pgvector (free tier, 20k-passage subset), Gemini free
tier, Langfuse tracing, React UI.

Read `docs/ARCHITECTURE.md` first. Task-specific runbooks live in `.claude/skills/`:
`navigate-architecture`, `debug-a-question`, `run-experiments`, `add-strategy-or-retriever`, `tracing-langfuse`.

## Stack
- Python 3.12 via **uv** (`uv run …`, never bare `python`). FastAPI · LangGraph · psycopg3 + pgvector ·
  sentence-transformers (MPS) · bm25s · spaCy 3.8 + `en_core_sci_sm` · langfuse · ragas.
- Frontend: `frontend/` React 19 + Vite + TypeScript + Tailwind v4 + recharts (`npm run dev` → :5173, proxies `/api` → :8000).
- DB: Supabase Postgres 17 + pgvector 0.8 (`halfvec`, partial HNSW per strategy). Direct connection string in `.env`.

## Run
```bash
uv run uvicorn app.main:app --reload --port 8000      # API
cd frontend && npm run dev                            # UI
uv run python evals/run_retrieval_eval.py --set smoke5 --strategies passage --retrievers hybrid   # quick eval
```
Full data build order: `docs/HOW_TO_RUN.md` §1.

## Key files
| Concern | File |
|---|---|
| Config (all env vars) | `config.py` |
| Chunkers + registry | `pipeline/chunking/strategies.py` (`build_chunker`, `DEFAULT_STRATEGIES`) |
| Retrievers + registry | `pipeline/retrieval/runner.py` (`RETRIEVERS`, `RetrievalConfig`, `run_retrieval`) |
| Individual retrievers | `pipeline/retrieval/{bm25,dense,hybrid,grep,kg}.py` |
| Transforms / expansion / rerank | `pipeline/retrieval/transforms.py`, `expand.py`, `pipeline/reranker.py` |
| Graph + run entry | `pipeline/graph.py` (`run_pipeline`) |
| Prompts | `pipeline/prompts.py`, transform prompts in `transforms.py` |
| LLM client | `pipeline/llm.py` (`get_llm(role)`; Gemini round-robin) |
| Tracing switch | `pipeline/tracing.py` |
| Query logs | `pipeline/query_log.py` → table `query_logs` |
| Metrics | `evals/metrics.py` |
| Eval runners | `evals/run_retrieval_eval.py`, `evals/run_ragas_eval.py` |
| API | `app/main.py`, `app/schemas.py` |
| UI pages | `frontend/src/pages/*.tsx`; API client `frontend/src/lib/api.ts` |
| Schema | `db/schema.sql` (apply with `db/setup_db.py`) |
| Ingestion | `db/ingestion/{load_corpus,preprocess_nlp,subsample_corpus,build_chunks}.py` |

## Working rules
- **Read before editing; minimal diffs; one concern per change.** Don't refactor around a fix.
- **Config only via `config.py`** (pydantic-settings). No `os.getenv` elsewhere.
- **Prompts only in `pipeline/prompts.py` / `transforms.py`.** LLM calls only through `get_llm(role).chat()`.
- **DB access** through `db/conn.py` pools (`get_pool()` sync in retrievers/scripts). SET/DDL statements
  cannot take bind parameters — use `psycopg.sql` or f-strings with validated values.
- **Chunks are offsets** into `passages.text`; never store chunk text. Use the `chunk_text` view.
- **Storage budget** is the Supabase free tier (500 MB). Check `GET /api/health` → `db_size` before
  adding chunk sets or indexes; the `sentence` set is intentionally not HNSW-indexed (`NO_INDEX`).
- **Gemini free tier = 15 RPM per model.** Round-robin is in `pipeline/llm.py`; don't add tight loops
  of LLM calls without checking `GEMINI_MODELS`. DeepSeek only when quality is imperative.
- **Observation names in traces are stable verbs** (`retrieve-chunks`, `generate-answer`); put variants
  in metadata, not names.
- **Every pipeline run must stay loggable**: new state that matters for debugging goes into `stages`
  or `hit.meta`, which flow to `query_logs` and the UI automatically.
- Frontend: types in `frontend/src/lib/api.ts` must match FastAPI responses. Run `npx tsc --noEmit -p tsconfig.app.json` after UI changes.

## Debugging
- A question's full story: UI **Query Inspector**, or `select * from query_logs where id = '<run_id>'`.
- Logs: `logs/api.jsonl`, `logs/eval.jsonl`, `logs/ingest.jsonl` — grep the `[tag]`.
- Known gotchas: scispaCy 0.5.4 model on spaCy 3.8 needs the config fix in `pipeline/nlp.py`;
  `ragas` needs `evals/ragas_compat.py` imported first; HNSW builds must use small
  `maintenance_work_mem` and `max_parallel_maintenance_workers = 0` on Supabase Nano.

## Tables
`passages`, `qa_pairs`, `eval_sets`, `chunk_strategies`, `chunks` (+ view `chunk_text`), `kg_entities`,
`query_logs`, `eval_runs`, `eval_results`, `corpus_meta`. See `db/schema.sql`.
