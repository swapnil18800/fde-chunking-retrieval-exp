# Challenges, trade-offs and what to do next

An honest engineering log: what got in the way, what we decided and why, and what the results say
should be done next. Companion to [EVALUATION.md](EVALUATION.md) (numbers) and
[ARCHITECTURE.md](ARCHITECTURE.md) (design).

## 1. Challenges we hit (and how each was solved)

| # | Challenge | What happened | Resolution |
|---|---|---|---|
| 1 | **Free-tier storage** (Supabase 500 MB) | Full plan = 40k passages × 5 chunk sets ≈ 570k chunks → ~480 MB of 384-d `halfvec` *before* any HNSW index; the knowledge graph as row tables alone was 180 MB. | Chunks stored as **char offsets** (no text duplication); `halfvec` (2 B/dim); KG rewritten as **array columns** (~180 → 21 MB); corpus **subsampled 40k → 20k** keeping every gold passage of `eval150`; HNSW only on `passage` + `recursive_128_32` — the third index put the DB at 488 MB, so it was dropped. Final: ~430 MB. |
| 2 | **Free-tier LLM quota** (Gemini 15 RPM *per model*) | Query-transform sweeps need 150 LLM calls per cell; RAGAS judging needs ~8 calls per question. Bursts produced 429s; one thinking model returned empty text when `max_tokens` was small; Gemini's OpenAI endpoint rejects `n>1`. | **Round-robin across models** (`pipeline/llm.py`, quota is per model so 3 models ≈ 45 RPM), honour `retryDelay`, a LangChain `RoundRobinChat` for RAGAS, `ResponseRelevancy(strictness=1)`, generous `max_tokens`. Transform cells still take ~16 min each. |
| 3 | **Supabase round-trip latency (~0.35 s/query, fresh TLS connect 2.8 s)** | The eval harness opened a new connection per question and wrote two rows per question → ~4 s/question, 4 h for the matrix. | Pooled connections, **one batched insert per config**, `query_logs` text stripped for eval rows, tracing off for large sweeps → ~1 s/question. |
| 4 | **Laptop sleep killed a 5-hour eval** | Pooled connections died silently; the process hung on the 4th cell. | Pools now **validate connections on checkout** + TCP keepalives; sweeps run under `caffeinate -i`; per-config writes mean a crash loses at most one cell. |
| 5 | **Dependency rot** | scispaCy 0.5.4 model pins spaCy < 3.8 whose `thinc` is binary-incompatible with numpy 2; its `config.cfg` has a quoted boolean confection ≥ 1 rejects. `ragas` 0.4 imports a module removed from `langchain-community` 0.4. | Load only the *model* (drop the `scispacy` library), override the spaCy pin, fix the quoted bool at load time (`pipeline/nlp.py`); `evals/ragas_compat.py` shim. |
| 6 | **Postgres gotchas on a Nano instance** | HNSW builds with `maintenance_work_mem=256MB` crashed (“could not resize shared memory segment”); `NOT IN (…)` over 790k rows ran > 10 min under the 2-min `statement_timeout`; SET/DDL statements reject bind parameters. | `maintenance_work_mem=64MB`, `max_parallel_maintenance_workers=0`; set-based `LEFT JOIN … FILTER` rewrites; `psycopg.sql` literals for DDL. |
| 7 | **Langfuse SDK v4 / new-org API** | `span.update_trace` no longer exists; legacy `/api/public/traces/{id}` is blocked for organisations created after 2026-09-16. | `propagate_attributes()` for session/tags/metadata, `set_trace_io` for trace I/O; audits via `/api/public/v2/observations?fields=core,basic,io,usage,model,metadata`. |
| 8 | **Entity noise in the knowledge graph** | scispaCy `en_core_sci_sm` tags generic nouns ("mechanism", "reduced", "clinical trials") as entities; surface forms fragment ("hirschsprung disease" / "hirschsprung's disease" / "hscr"). | Document-frequency caps (multi-word ≤ 2.5 %, single-word ≤ 300 passages) and a stop-list; trigram fuzzy matching at query time. Still the weakest retriever (recall@10 0.346) — see §3. |
| 9 | **Metric design for chunk-level systems** | A strategy with 7 chunks per abstract could fill top-10 with one passage and look great or terrible depending on how you count. | All metrics are **passage-level** (chunk hits collapsed to parent PMID in rank order). Side effect: `grep`/`kg` (passage rankers) and expansion modes are provably invariant across chunking — we stopped evaluating expansion in Tier 1 after confirming it. |

## 2. Trade-offs we chose

| Decision | We chose | Instead of | Because |
|---|---|---|---|
| Corpus size | 20k-passage subset (all eval gold kept + other questions' gold as hard distractors + random) | full 40k | Only way to keep all five chunk sets *and* HNSW on the free tier. Retrieval is somewhat easier than on 40k; documented in `corpus_meta` and every table caption. |
| Vector index | HNSW on 2 sets, exact scan on 3 | HNSW everywhere | Exact scan over 50–130k `halfvec` rows costs 50–200 ms — invisible next to a 1–4 s LLM call; each index costs ~1 KB/row of the 500 MB. |
| Embedding model | `MedEmbed-small-v0.1` (384-d, local) | bge-base / Gemini embeddings | Medical fine-tune, zero cost, 4× smaller vectors than 768-d. The price shows up in the results: dense loses to BM25 on whole abstracts (0.460 vs 0.533). |
| Reranker | `MedCPT-Cross-Encoder` (local) | LLM reranking | PubMed-trained, free, +0.023 recall@10 and MRR → 0.80 on every cell for ~0.5–1 s. |
| Generator / judge | Gemini free tier, DeepSeek only as fallback | paid frontier model | Cost. Answer quality is good enough (see RAGAS); judge noise is mitigated with id-based context metrics that need no LLM. |
| Evaluation | Two tiers: exact retrieval metrics on everything, RAGAS on winners only | RAGAS everywhere | 25+ configs × 150 questions × ~8 judge calls would take days on free quota and add noise; retrieval metrics are exact because gold PMIDs exist. |
| Knowledge graph | scispaCy surface forms + co-occurrence, arrays in Postgres | LLM-extracted triples / UMLS linking / a graph DB | Free, one CPU pass, fits the DB. It is a baseline that explains its hits, not a competitive retriever. |
| Chunk storage | offsets into the passage | duplicated chunk text | Halves storage and gives exact highlight spans for citations for free. |
| Tracing | Langfuse default, LangSmith validated once | one vendor | Langfuse is free at this volume; LangSmith credits were limited. Switchable per run via `TRACING_PROVIDER`. |
| Eval traceability | every eval number links to a `query_logs` row | aggregate scores only | Lets any heatmap cell be explained question by question; costs ~2 KB/row. |

## 3. What the results say to do next

Ordered by expected value on this benchmark.

1. **Reranking is free recall — make it the default** (`+0.023` recall@10, MRR → 0.80 everywhere).
   Try `candidate_k` 50–100: BM25 recall@20 is well above recall@10, so the reranker is currently
   starved by the first stage, not by its own accuracy.
2. **HyDE + rerank.** HyDE alone gave the best MRR (0.827 on `passage+hybrid`); it has not yet been
   combined with MedCPT. Same for `multi_query`/`decompose` once their cells land.
3. **List questions are the weak spot** (recall@10 0.48 vs 0.59 for yes/no; 8.5 gold abstracts each).
   Evaluate at k = 20 and give list questions a larger top-k in generation; this is where
   `decompose` should pay off.
4. **A stronger embedder for long chunks.** Dense trails BM25 by 7 points on whole abstracts. Test
   `MedEmbed-base`/`bge-base` (768-d — halve the chunk sets or upgrade the DB) or MedCPT's own
   query/article encoders; keep `sentence`+dense, where the small model already wins.
5. **Fix the knowledge graph before judging the idea.** Normalise entities (lower-case lemma +
   abbreviation expansion, or UMLS CUIs via scispaCy's linker), drop single-token generic entities
   entirely, weight edges by PMI rather than raw co-occurrence. Then re-run the `kg` column.
6. **Cheap contextual retrieval.** Prefix each chunk with its passage's top entities (no LLM) and
   embed — a zero-cost approximation of Anthropic-style contextual chunks. Storage: +1 chunk set.
7. **Tier 2 on expansion modes.** Expansion is invisible to retrieval metrics by construction; only
   RAGAS faithfulness/relevancy can show whether `sentence+window` or `+parent` beats `passage` as
   generator context. Run it on the top 3 configs with `--limit 50`.
8. **Ops.** Move the eval sweep to a cron/CI job (Langfuse dataset + experiment or GitHub Action) so
   a chunker/retriever change re-scores `eval150` automatically; cache the Gemini transform outputs
   per question so transform sweeps stop paying 150 LLM calls per cell.

## 4. Things we would do differently from the start

- Measure the storage budget on 500 passages *before* building anything — two chunk sets were built
  twice.
- Decide passage-level metrics first; they make half the planned sweeps (expansion, grep/kg × chunking)
  unnecessary.
- Batch every DB write and never open ad-hoc connections when the database is 350 ms away.
- Run long jobs under `caffeinate`/`nohup` with per-unit checkpoints from day one.
