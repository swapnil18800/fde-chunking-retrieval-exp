# Directory structure

```
fde-chunking-retrieval-exp/
├── README.md · CLAUDE.md · LICENSE · pyproject.toml · uv.lock · .env.example · .python-version
├── config.py                        # pydantic-settings; the only place env vars are read
│
├── app/                             # FastAPI
│   ├── main.py                      # routes: health/options · ask/compare · questions/passages/chunk-preview · kg · logs · eval-runs; serves frontend/dist
│   └── schemas.py                   # AskRequest / CompareRequest
│
├── pipeline/                        # the RAG pipeline
│   ├── chunking/
│   │   ├── base.py                  # Passage, Chunk, Chunker, ChunkerSpec (char-offset chunks)
│   │   └── strategies.py            # passage · fixed_128_32 · recursive_128_32 · sentence · semantic · build_chunker()
│   ├── retrieval/
│   │   ├── base.py                  # Hit, Retriever, rrf(), dedupe_by_passage()
│   │   ├── store.py                 # DB helpers: attach_text, neighbours, passages_text, chunks_for_passages
│   │   ├── bm25.py · dense.py · hybrid.py · grep.py · kg.py
│   │   ├── transforms.py            # hyde · multi_query · decompose (LLM) + retrieve_multi (RRF)
│   │   ├── expand.py                # none · window · parent
│   │   └── runner.py                # RetrievalConfig, RETRIEVERS, run_retrieval() with stage timings
│   ├── embedder.py                  # sentence-transformers wrapper (MedEmbed-small, MPS)
│   ├── reranker.py                  # MedCPT cross-encoder
│   ├── llm.py                       # LLMClient (Gemini round-robin / DeepSeek), RoundRobinChat, get_llm(role)
│   ├── prompts.py                   # answer prompts
│   ├── citations.py                 # [n] → chunk → passage → PubMed
│   ├── graph.py                     # LangGraph retrieve-context → generate-answer → cite-sources; run_pipeline()
│   ├── tracing.py                   # TRACING_PROVIDER switch, root span, child spans
│   ├── query_log.py                 # query_logs write/read
│   ├── nlp.py                       # scispaCy loader (spaCy 3.8 config fix), entity cleaning
│   ├── tokens.py                    # tiktoken helpers, token→char offsets
│   └── logging_setup.py             # [tag] console + JSONL
│
├── db/
│   ├── schema.sql · setup_db.py · conn.py (psycopg pools, pgvector types)
│   └── ingestion/
│       ├── load_corpus.py           # HF → passages, qa_pairs
│       ├── preprocess_nlp.py        # sentence offsets + entity graph arrays
│       ├── subsample_corpus.py      # 40k → 20k free-tier subset (eval150 gold protected)
│       └── build_chunks.py          # chunk → embed → COPY halfvec → partial HNSW
│
├── evals/
│   ├── metrics.py                   # passage-level recall/precision/hit/MRR/nDCG
│   ├── build_eval_sets.py           # eval150 + smoke5
│   ├── run_retrieval_eval.py        # Tier 1 matrix runner → eval_runs/eval_results + results/
│   ├── run_ragas_eval.py            # Tier 2 RAGAS runner
│   ├── ragas_compat.py              # langchain-community 0.4 shim
│   ├── datasets/                    # eval150.json · smoke5.json (committed)
│   └── results/<stamp>_<name>/      # summary.json · per_config.csv · per_question.csv · report.md
│
├── frontend/                        # React 19 + Vite + TS + Tailwind v4
│   └── src/
│       ├── App.tsx                  # tab shell
│       ├── lib/api.ts · lib/hooks.ts
│       ├── components/              # ConfigPicker · QuestionPicker · Answer (citations) · PassageView · Stages · Metrics
│       └── pages/                   # Overview · Playground · Compare · Leaderboard · ChunkExplorer · KnowledgeGraph · QueryInspector
│
├── docs/                            # ARCHITECTURE · CHUNKING · RETRIEVAL · EVALUATION · TRACING · HOW_TO_RUN · DIRECTORY_STRUCTURE
├── .claude/skills/                  # navigate-architecture · debug-a-question · run-experiments · add-strategy-or-retriever · tracing-langfuse
├── logs/                            # *.jsonl per process (git-ignored)
└── .cache/                          # bm25 indexes (git-ignored)
```
