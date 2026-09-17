# Evaluation

> Methodology here; the results tables are appended in §4 as runs complete (newest first).

## 1. Philosophy

- **Retrieval first, generation second.** The question of this project is *which chunking and
  retrieval choices surface the right evidence*. That is measurable exactly, for free, at scale
  (gold PMIDs exist for every question). LLM-judged answer quality is measured only on the
  configurations that survive Tier 1 — it is slow (15 RPM free tier) and noisy.
- **Passage-level truth.** Gold labels are PMIDs. Chunk hits are collapsed to their parent passage
  *in rank order* before scoring, so a strategy with 7 chunks per abstract is not rewarded for
  filling the top-10 with one passage.
- **One question = one logged run.** Every eval number links to a `query_logs` row
  (`eval_results.query_log_id`) with stages, hits and per-retriever ranks, so any surprising cell
  in the heatmap can be explained question by question.
- **Same questions everywhere.** `eval150` is fixed (seed 42, stratified by question type, 1–20
  gold passages). `smoke5 ⊂ eval150` is for wiring checks only.

## 2. Tier 1 — retrieval metrics (`evals/run_retrieval_eval.py`, `evals/metrics.py`)

| metric | definition | catches |
|---|---|---|
| `recall@k` | \|gold ∩ top-k passages\| / \|gold\| | evidence not surfaced |
| `precision@k` | \|gold ∩ top-k\| / k | noise the generator must wade through |
| `hit@k` | 1 if any gold in top-k | "at least one right passage" |
| `mrr` | 1 / rank of first gold passage | how high the first good hit is |
| `ndcg@k` | binary-gain DCG / ideal DCG | rank-sensitive recall |
| `latency_ms` | wall time of the retrieval-only pipeline (transform → retrieve → rerank → expand) | cost of the method |

`k ∈ {1, 3, 5, 10, 20}`; runs use `top_k=10`, `candidate_k=30` unless stated. **Headline metric:
`recall@10`** (BioASQ questions average 10 gold passages). `precision@5` is the best single proxy
for "what the generator sees".

Sweeps: the 5 × 5 base matrix; the matrix `+rerank` (MedCPT over 30 candidates); expansion
(`window`/`parent`) on fine strategies; LLM query transforms (`hyde`/`multi_query`/`decompose`).

## 3. Tier 2 — RAGAS (`evals/run_ragas_eval.py`)

| metric | type | what it measures |
|---|---|---|
| `faithfulness` | LLM judge | fraction of answer claims supported by the retrieved context (hallucination) |
| `answer_relevancy` | LLM + local embeddings | answer addresses the question (strictness=1) |
| `id_based_context_precision` | exact | retrieved PMIDs vs gold, rank-aware |
| `id_based_context_recall` | exact | gold PMIDs present in retrieved |
| `factual_correctness` (F1) | LLM judge | answer claims vs the BioASQ reference answer |

Judge: Gemini free tier, round-robined over `GEMINI_MODELS` (`pipeline/llm.py::RoundRobinChat`)
because quotas are per model; `--judge deepseek` is the paid fallback. Generation model for the
answers under test: `gemini-3.5-flash-lite` (round-robin). Runs are limited (`--limit 50`) to fit
daily free quotas.

## 4. Results

_(appended per run — see `evals/results/<stamp>_<name>/report.md` for the raw tables)_
