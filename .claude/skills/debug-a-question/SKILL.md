---
name: debug-a-question
description: Diagnose why a specific question produced a bad/empty answer, wrong citations, missing gold passages, a slow run or an error. Walks query_logs, stages, retriever ranks, the Langfuse trace and process logs.
---

# Debug a question

Every run has a `run_id` (= `query_logs.id`, returned by `/api/ask`, shown in the UI and in the trace metadata).

## 1. Get the record
```sql
select created_at, source, config, latency_ms, error, trace_url from query_logs where id = '<run_id>';
select jsonb_pretty(stages), jsonb_pretty(retrieved), jsonb_pretty(citations), metrics from query_logs where id = '<run_id>';
```
or UI → Query Inspector → row. Or `grep '<run_id>' logs/api.jsonl`.

## 2. Localise the failure by stage
| Symptom | Where to look | Typical cause |
|---|---|---|
| `error` set | `stages` stops early; `logs/api.jsonl` `[graph]` line with traceback | LLM 429/400 (`[llm]` lines), DB timeout, unknown strategy |
| gold passages missing from `retrieved` (low recall) | `retrieved[*].retriever_ranks` — did dense or bm25 have it deep in the list? Re-run with `candidate_k=50`, `retriever=hybrid` | vocabulary mismatch (try `hyde`/`multi_query`), fine chunks diluting (try `parent`), KG entity noise |
| gold retrieved but answer says "passages do not answer" | `stages[generate].context_chunks`, chunk `text` length | chunk too small (`sentence` without `window`), prompt too strict |
| `[n]` cites the wrong passage | `citations[*].chunk_id` vs `retrieved` order — numbering is the *final* hit order after rerank/expand | expansion `parent` collapsed duplicates, invalid markers stripped (`stages[cite].invalid_markers`) |
| slow | `stages[*].ms`; first request pays model download + BM25 build | `grep` with many terms (0.5 s/term), exact-scan `sentence` dense (~200 ms), reranker cold start |
| retrieval metrics missing | `qa_id` was null → no gold known | pick the question from the BioASQ picker (sets qa_id) |

## 3. Reproduce in isolation
```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
from pipeline.retrieval.runner import run_retrieval, RetrievalConfig
r = run_retrieval('<question>', RetrievalConfig(strategy='sentence', retriever='hybrid', rerank=True, candidate_k=40, top_k=10))
for h in r.hits: print(h.rank, h.passage_id, round(h.score,3), h.retriever_ranks, (h.text or '')[:90])
print([(s['name'], s['ms']) for s in r.stages])"
```
Compare against gold: `select relevant_passage_ids from qa_pairs where id = <qa_id>`.

## 4. Check the trace
Open `trace_url` (Langfuse). Confirm `retrieve-chunks` output ids match `retrieved`, `generate-answer`
generation shows model + tokens + the numbered context. If the trace is missing: `TRACING_PROVIDER`,
`init_tracing()` auth log line at API start (`[tracing] langfuse auth_check=True`).

## 5. Known failure modes
- **Gemini 429**: 15 RPM per model. `[llm] 429 … rotating model` lines; add models to `GEMINI_MODELS` or lower concurrency.
- **Empty LLM text**: thinking models consume `max_tokens`; `LLM_MAX_TOKENS` ≥ 2048.
- **`kg` returns nothing**: question had no entities in `kg_entities` (check `GET /api/kg/search?q=`). Seeds are in `hit.meta.seed_entities`.
- **`grep` slow/noisy**: terms are in `hit.meta.query_terms`; generic words inflate matches — tune `STOP` in `grep.py`.
- **statement timeout** on heavy SQL: Supabase default is 2 min; set `statement_timeout` in the session for maintenance scripts.
