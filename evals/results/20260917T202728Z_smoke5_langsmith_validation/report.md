# RAGAS eval — smoke5_langsmith_validation

- set `smoke5` (5 q) · judge `gemini-3.5-flash` · top_k=5 · run `4786b5f3-a779-47e5-985e-ee6166f1c65e` · 20260917T202728Z

| config                |   faithfulness |   id_based_context_precision |   id_based_context_recall |   factual_correctness(mode=f1) |   recall@5 |   mrr |   latency_ms |   errors |
|:----------------------|---------------:|-----------------------------:|--------------------------:|-------------------------------:|-----------:|------:|-------------:|---------:|
| passage+hybrid+rerank |              1 |                          0.6 |                     0.535 |                           0.09 |      0.535 |     1 |       9612.2 |        0 |
