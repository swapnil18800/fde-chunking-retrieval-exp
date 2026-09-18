# Retrieval eval — eval150_transforms

- set: `eval150` (150 questions) · top_k=10 · candidate_k=30 · run_id `8a57d122-3b64-480b-aefd-4ede9bf03e6f` · 20260918T051228Z

Metrics are passage-level (chunk hits collapsed to parent PMID). Sorted by recall@10.

| config                      |   recall@5 |   recall@10 |   precision@5 |   mrr |   ndcg@10 |   hit@5 |   latency_ms |   errors |
|:----------------------------|-----------:|------------:|--------------:|------:|----------:|--------:|-------------:|---------:|
| passage+hybrid+hyde         |      0.454 |       0.558 |         0.544 | 0.827 |     0.65  |   0.867 |      6404.15 |        0 |
| passage+hybrid+multi_query  |      0.434 |       0.547 |         0.523 | 0.775 |     0.62  |   0.853 |      7871.67 |        0 |
| passage+hybrid+decompose    |      0.422 |       0.533 |         0.517 | 0.781 |     0.612 |   0.84  |      7078    |        0 |
| sentence+hybrid+hyde        |      0.425 |       0.486 |         0.512 | 0.783 |     0.58  |   0.853 |      7149.9  |        0 |
| sentence+hybrid+multi_query |      0.41  |       0.478 |         0.489 | 0.786 |     0.573 |   0.833 |      8631.96 |        0 |
| sentence+hybrid+decompose   |      0.421 |       0.476 |         0.501 | 0.787 |     0.572 |   0.853 |      7225.19 |        0 |
