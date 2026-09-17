# Tracing & logging

Three layers, all on by default, all keyed by the same `run_id`:

| layer | where | what | switch |
|---|---|---|---|
| **Process log** | `logs/<process>.jsonl` (+ rich console) | `[tag]`-prefixed records: `[ingest] [chunk] [embed] [retrieve] [llm] [graph] [eval] [api]`, with `ctx_*` fields (run_id, config, latency) | `LOG_LEVEL` |
| **Query log** | Postgres `query_logs` | One row per pipeline run: question, config, every stage with ms, retrieved hits (chunk/passage ids, scores, per-retriever ranks), answer, citations, retrieval metrics (when gold known), tokens, trace url, error | always (best-effort insert) |
| **Trace** | Langfuse (default) / LangSmith / off | Root chain `rag-answer` → LangGraph nodes → typed observations; generations carry model + token usage | `TRACING_PROVIDER` |

## Langfuse structure (SDK v4)

```
rag-answer                      chain   input={question, config}  output={answer, n_citations, retrieved_pmids}
└─ rag-graph                    chain   (LangGraph via langfuse.langchain.CallbackHandler)
   ├─ retrieve-context          chain
   │  ├─ transform-query        span|chain   metadata.mode = none|hyde|multi_query|decompose  (+ rewrite-query generation when LLM used)
   │  ├─ retrieve-chunks        retriever    input={queries,k} output=[{chunk_id,passage_id,score}] metadata={retriever,strategy}
   │  └─ rerank-chunks          span         (only when rerank=true)
   ├─ generate-answer           chain
   │  └─ generate-answer        generation   model=gemini-… usage=input/output tokens, prompt+completion text
   └─ cite-sources              chain
```

- Trace attributes (`session_id`, `tags = [source, strategy, retriever, rerank?]`, `metadata =
  {qa_id, run_id}`) are set once with `propagate_attributes()` and inherited by every child,
  including the callback-created ones.
- Observation names are **stable verbs**; the variant (which retriever, which transform) lives in
  metadata so dashboards/evaluators keyed on names keep working when configs change.
- The trace URL is returned by `/api/ask` and stored in `query_logs.trace_url`; the UI links it.
- `pipeline/graph.py` calls `init_tracing()` at import, *before* any LangChain client exists.
  Scripts call `pipeline.graph.shutdown()` → `flush()` before exit.

## LangSmith

`TRACING_PROVIDER=langsmith` sets `LANGSMITH_TRACING=true` and the project; LangGraph/LangChain
trace natively (the RAGAS judge calls are traced too). It was used once — `smoke5` with
`passage+hybrid+rerank` — to validate the harness against a second backend; credits are limited so
it is off by default.

## Reading a question's story

1. UI → **Query Inspector** → pick the run: config, stage bar, retrieved chunks with
   `retriever_ranks` (e.g. `dense #3, bm25 #12`), answer with `[n]` → chunk → parent → PubMed,
   trace link.
2. SQL: `select stages, retrieved, citations from query_logs where id = '<run_id>'`.
3. Files: `grep '<run_id>' logs/api.jsonl`.
