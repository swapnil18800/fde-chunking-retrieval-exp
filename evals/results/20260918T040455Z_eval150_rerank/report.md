# Retrieval eval — eval150_rerank

- set: `eval150` (150 questions) · top_k=10 · candidate_k=30 · run_id `5aaa89fc-1282-49b1-b4b3-e34a4900cf85` · 20260918T040455Z

Metrics are passage-level (chunk hits collapsed to parent PMID). Sorted by recall@10.

| config                         |   recall@5 |   recall@10 |   precision@5 |   mrr |   ndcg@10 |   hit@5 |   latency_ms |   errors |
|:-------------------------------|-----------:|------------:|--------------:|------:|----------:|--------:|-------------:|---------:|
| passage+bm25+rerank            |      0.465 |       0.568 |         0.552 | 0.803 |     0.661 |   0.867 |     1860.96  |        0 |
| passage+hybrid+rerank          |      0.462 |       0.562 |         0.555 | 0.798 |     0.656 |   0.86  |     2564.63  |        0 |
| semantic+bm25+rerank           |      0.446 |       0.516 |         0.525 | 0.802 |     0.613 |   0.867 |     1162.89  |        0 |
| recursive_128_32+bm25+rerank   |      0.45  |       0.515 |         0.535 | 0.81  |     0.614 |   0.86  |     1011.69  |        0 |
| fixed_128_32+hybrid+rerank     |      0.455 |       0.515 |         0.543 | 0.803 |     0.611 |   0.86  |     1913.69  |        0 |
| semantic+hybrid+rerank         |      0.439 |       0.511 |         0.525 | 0.791 |     0.608 |   0.853 |     1950.07  |        0 |
| recursive_128_32+hybrid+rerank |      0.455 |       0.509 |         0.543 | 0.807 |     0.608 |   0.853 |     1830.89  |        0 |
| fixed_128_32+bm25+rerank       |      0.452 |       0.507 |         0.541 | 0.806 |     0.61  |   0.86  |      965.393 |        0 |
| passage+dense+rerank           |      0.412 |       0.5   |         0.517 | 0.75  |     0.593 |   0.813 |     2574.48  |        0 |
| sentence+hybrid+rerank         |      0.427 |       0.489 |         0.511 | 0.808 |     0.592 |   0.847 |     1916.29  |        0 |
| sentence+dense+rerank          |      0.424 |       0.489 |         0.508 | 0.798 |     0.591 |   0.84  |     2202.83  |        0 |
| semantic+dense+rerank          |      0.421 |       0.485 |         0.508 | 0.786 |     0.587 |   0.84  |     2178.01  |        0 |
| sentence+bm25+rerank           |      0.417 |       0.476 |         0.491 | 0.803 |     0.58  |   0.84  |      793.313 |        0 |
| fixed_128_32+dense+rerank      |      0.424 |       0.469 |         0.524 | 0.779 |     0.572 |   0.833 |     1859.51  |        0 |
| recursive_128_32+dense+rerank  |      0.409 |       0.457 |         0.508 | 0.771 |     0.559 |   0.82  |     1826.8   |        0 |
