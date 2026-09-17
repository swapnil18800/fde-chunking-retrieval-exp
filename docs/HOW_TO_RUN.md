# How to run

## Prerequisites

- Python 3.12 via [`uv`](https://docs.astral.sh/uv/) · Node 20+ · a Supabase Postgres (free tier is enough)
- Keys: Gemini (AI Studio, free), Langfuse (Hobby, free). Optional: DeepSeek, LangSmith.
- Apple Silicon or CUDA is nice for embeddings but CPU works (slower).

```bash
git clone https://github.com/swapnil18800/fde-chunking-retrieval-exp.git && cd fde-chunking-retrieval-exp
uv sync                                # python deps (torch, sentence-transformers, spaCy model …)
cd frontend && npm install && cd ..
cp .env.example .env                   # fill DATABASE_URL, GEMINI_API_KEY, LANGFUSE_* (see below)
```

`.env` essentials:

```
DATABASE_URL=postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres?sslmode=require
GEMINI_API_KEY=...            GEMINI_MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash
EMBEDDING_MODEL=abhinand/MedEmbed-small-v0.1   EMBEDDING_DIM=384
RERANKER_MODEL=ncbi/MedCPT-Cross-Encoder
TRACING_PROVIDER=langfuse     # langfuse | langsmith | off
LANGFUSE_PUBLIC_KEY=pk-lf-...  LANGFUSE_SECRET_KEY=sk-lf-...  LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## 1. Build the database (once, ~45 min end-to-end on an M4)

```bash
uv run python db/setup_db.py                              # schema (idempotent)
uv run python db/ingestion/load_corpus.py                 # 40,221 passages + 4,719 QA from HF (~2 min)
uv run python evals/build_eval_sets.py                    # eval150 + smoke5 (deterministic, seed 42)
uv run python db/ingestion/preprocess_nlp.py              # scispaCy sentences + entity graph (~18 min)
uv run python db/ingestion/subsample_corpus.py --target 20000 --yes   # free tier: 40k → 20k (~1 min)
uv run python db/ingestion/build_chunks.py --all          # 5 strategies, embed, HNSW (~25 min)
```

Each step is idempotent and logs to `logs/ingest.jsonl`. `--replace` rebuilds. Skip the subsample
step on a Pro instance (and drop `NO_INDEX` in `build_chunks.py` to index the `sentence` set too).

## 2. Run the app

```bash
uv run uvicorn app.main:app --reload --port 8000          # API  (warms embedder, reranker, BM25 caches)
cd frontend && npm run dev                                # UI   http://localhost:5173  (proxies /api → :8000)
```

Production: `cd frontend && npm run build` → FastAPI serves `frontend/dist` at `/`.

## 3. Evaluate

```bash
# Tier 1: retrieval metrics, no LLM — the full matrix on 150 questions (~30-40 min)
uv run python evals/run_retrieval_eval.py --set eval150 --matrix
uv run python evals/run_retrieval_eval.py --set eval150 --matrix --rerank-only          # + MedCPT on every cell
uv run python evals/run_retrieval_eval.py --set eval150 --strategies sentence --retrievers hybrid \
    --transforms hyde multi_query decompose --expansions none window parent           # LLM transforms

# Tier 2: RAGAS on the winners (Gemini judge, free tier → slow; ~15 RPM)
uv run python evals/run_ragas_eval.py --set smoke5  --configs passage+hybrid+rerank      # harness check
uv run python evals/run_ragas_eval.py --set eval150 --configs sentence+hybrid+rerank+window passage+hybrid+rerank --limit 50
```

Results land in `evals/results/<timestamp>_<name>/` (`summary.json`, `per_config.csv`,
`per_question.csv`, `report.md`) and in the `eval_runs` / `eval_results` tables (Leaderboard tab).

## 4. Tracing switch

| `TRACING_PROVIDER` | Effect |
|---|---|
| `langfuse` (default) | Root chain `rag-answer` + LangGraph nodes + generations with model/tokens, session/tags |
| `langsmith` | Native LangChain tracing to `LANGSMITH_PROJECT` (used once to validate; credits are limited) |
| `off` | No exporter. `query_logs` + `logs/*.jsonl` still record everything |

Override per command: `TRACING_PROVIDER=off uv run python evals/run_retrieval_eval.py …`

## 5. Where to look when something is off

| Symptom | Look at |
|---|---|
| A question gave a bad answer | Query Inspector tab → stages, retrieved chunks, citations, trace link; or `select * from query_logs order by created_at desc limit 1` |
| Retrieval numbers dropped | `evals/results/*/per_question.csv`; compare `retriever_ranks` in `query_logs.retrieved` |
| 429 / empty LLM output | `logs/api.jsonl` lines tagged `[llm]`; Gemini free tier is 15 RPM per model — add models to `GEMINI_MODELS` |
| Slow first request | Model downloads (MedEmbed ~130 MB, MedCPT ~440 MB) and BM25 index build — one-time, cached in `~/.cache/huggingface` and `.cache/bm25/` |
| DB size | `GET /api/health` → `db_size`; free tier is 500 MB |
