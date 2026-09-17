---
name: tracing-langfuse
description: Work with tracing in this repo — Langfuse (default), LangSmith (validation only) or off; verify traces exist and are well-structured; audit via the Langfuse REST API; extend spans without breaking naming rules.
---

# Tracing (Langfuse v4 SDK)

Switch: `TRACING_PROVIDER=langfuse|langsmith|off` in `.env` (or per command). Code: `pipeline/tracing.py`.
Structure and rules: `docs/TRACING.md`.

## Verify traces are flowing
1. API start log: `[tracing] langfuse auth_check=True`.
2. Ask a question → response has `trace_provider`, `trace_id`, `trace_url`.
3. Audit via REST (new orgs must use the v2 observations API, not `/traces/{id}`):
   ```bash
   source .env; A="$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY"
   curl -s -u "$A" "$LANGFUSE_BASE_URL/api/public/v2/observations?traceId=<trace_id>&limit=50" | python3 -m json.tool | head -60
   curl -s -u "$A" "$LANGFUSE_BASE_URL/api/public/observations/<observation_id>"     # full payload incl. input/output/usage
   ```
   Expect: root `rag-answer` (CHAIN) → `rag-graph` → `retrieve-context` {`transform-query`, `retrieve-chunks` (RETRIEVER), `rerank-chunks`} · `generate-answer` → GENERATION with `model` + `usageDetails` · `cite-sources`. `sessionId` populated when the request had one.
4. Docs first for anything SDK-related (it changes often): `curl -s https://langfuse.com/docs/observability/sdk/python/instrumentation.md`.

## Adding a span
Use `pipeline.tracing.child_span(name, as_type, input, metadata)` inside pipeline code. Rules:
stable verb names (`rerank-chunks`, not `rerank:medcpt`), variants in `metadata`, set `output` via
`sp.update(output=...)`. Trace-level attributes only through `trace_span(...)` (propagate_attributes).

## LangSmith
`TRACING_PROVIDER=langsmith` — env-driven native LangChain tracing (`LANGSMITH_PROJECT`). Used once for
harness validation (`smoke5`, `passage+hybrid+rerank`); credits are limited — do not leave on.

## Gotchas
- `init_tracing()` must run before LangChain clients exist — `pipeline/graph.py` does this at import.
- Scripts must `pipeline.graph.shutdown()` (flush) before exit or the last traces are lost.
- `LangfuseSpan` has no `update_trace` in v4; use `set_trace_io` / `propagate_attributes`.
