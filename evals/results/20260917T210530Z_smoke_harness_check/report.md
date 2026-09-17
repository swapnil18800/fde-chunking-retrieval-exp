# Retrieval eval — smoke_harness_check

- set: `smoke5` (5 questions) · top_k=10 · candidate_k=30 · run_id `7cbed34d-d856-40df-93dd-87be882028a2` · 20260917T210530Z

Metrics are passage-level (chunk hits collapsed to parent PMID). Sorted by recall@10.

| config         |   recall@5 |   recall@10 |   precision@5 |   mrr |   ndcg@10 |   hit@5 |   latency_ms |   errors |
|:---------------|-----------:|------------:|--------------:|------:|----------:|--------:|-------------:|---------:|
| passage+hybrid |      0.499 |       0.575 |          0.52 | 0.84  |     0.694 |       1 |       2851   |        0 |
| passage+bm25   |      0.474 |       0.535 |          0.48 | 0.867 |     0.648 |       1 |       1214.4 |        0 |
