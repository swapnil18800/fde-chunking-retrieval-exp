---
name: run-experiments
description: Run, extend or re-run the evaluation matrix (Tier-1 retrieval metrics, Tier-2 RAGAS), read results, and update the results docs/leaderboard. Use for "evaluate X", "compare configs", "why did recall drop", "add results to README".
---

# Run experiments

## Question sets
`smoke5` (5 q, ⊂ eval150) for wiring checks; `eval150` (150 q, stratified factoid/list/yesno/summary,
1–20 gold each) for numbers. JSON copies in `evals/datasets/`; rebuild with `evals/build_eval_sets.py`
(deterministic, seed 42). After the 20k subsample every eval150 gold passage is still in the corpus.

## Tier 1 — retrieval (no LLM, free, ~10 s per config on 150 q)
```bash
uv run python evals/run_retrieval_eval.py --set eval150 --matrix                       # 5 × 5 base cells
uv run python evals/run_retrieval_eval.py --set eval150 --matrix --rerank-only         # every cell + MedCPT
uv run python evals/run_retrieval_eval.py --set eval150 --strategies sentence recursive_128_32 \
     --retrievers hybrid --expansions none window parent --name expansion_sweep
uv run python evals/run_retrieval_eval.py --set eval150 --strategies passage --retrievers hybrid \
     --transforms hyde multi_query decompose --name transforms   # LLM calls: ~150 × 3 at 15 RPM/model
```
Metrics (`evals/metrics.py`) are **passage-level**: chunk hits collapse to parent PMID. Primary:
`recall@10`; also `recall@5`, `precision@5`, `mrr`, `ndcg@10`, `hit@5`, `latency_ms`. Every question
run also writes a `query_logs` row (source=eval) so any number can be traced to a full run.

## Tier 2 — RAGAS (Gemini judge, ~15 RPM → budget time)
```bash
uv run python evals/run_ragas_eval.py --set smoke5 --configs passage+hybrid+rerank
uv run python evals/run_ragas_eval.py --set eval150 --limit 50 --configs <label> <label>
```
Metrics: `faithfulness`, `answer_relevancy` (local embeddings), `id_based_context_precision/recall`
(exact, from PMIDs), `factual_correctness` (vs BioASQ reference). `evals/ragas_compat.py` must be
imported before `ragas` (langchain-community 0.4 shim).

## Outputs
`evals/results/<UTC stamp>_<name>/{summary.json, per_config.csv, per_question.csv, report.md}` plus
`eval_runs`/`eval_results` rows → Leaderboard tab (heatmap chunking × retriever, bar chart, table).

## Reading results honestly
- Compare cells only within one run (same question set, same `top_k`/`candidate_k`).
- `passage` has a structural recall advantage (gold granularity); judge fine strategies with `+parent`
  or by `precision@5` / RAGAS faithfulness, not recall alone.
- `grep`/`kg` are baselines; expect lower recall — the point is to *measure* the gap.
- Per-type breakdown: `per_question.csv` has `question_type`; list questions have the most gold and the lowest recall@k by construction.

## Updating docs after a run
1. Copy the `report.md` table into `docs/EVALUATION.md` (results section) with the run stamp.
2. Refresh the headline numbers in `README.md`.
3. Note what changed and why in `docs/EVALUATION.md` history — one bullet per run, newest first.
