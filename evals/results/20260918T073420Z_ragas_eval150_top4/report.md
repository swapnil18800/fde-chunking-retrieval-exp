# RAGAS eval — ragas_eval150_top4

- set `eval150` (40 q) · judge `deepseek` · top_k=5 · run `9be47280-acc4-45c0-bffb-f2d459b24540` · 20260918T073420Z

| config                        |   faithfulness |   answer_relevancy |   id_based_context_precision |   id_based_context_recall |   factual_correctness(mode=f1) |   recall@5 |   mrr |   latency_ms |   errors |
|:------------------------------|---------------:|-------------------:|-----------------------------:|--------------------------:|-------------------------------:|-----------:|------:|-------------:|---------:|
| passage+hybrid                |          0.934 |              0.867 |                        0.505 |                     0.529 |                          0.42  |      0.529 | 0.776 |      5791.2  |        0 |
| passage+bm25+rerank           |          0.896 |              0.881 |                        0.525 |                     0.566 |                          0.447 |      0.566 | 0.852 |      6423.6  |        0 |
| passage+hybrid+hyde           |          0.905 |              0.862 |                        0.5   |                     0.515 |                          0.37  |      0.515 | 0.808 |     11034.5  |        0 |
| sentence+hybrid+rerank+window |          0.917 |              0.873 |                        0.644 |                     0.501 |                          0.382 |      0.501 | 0.821 |      7458.05 |        0 |
