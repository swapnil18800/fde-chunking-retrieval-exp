# Retrieval strategies

Every retriever implements `pipeline/retrieval/base.py::Retriever.retrieve(query, strategy, k)`
and returns ranked `Hit`s at **chunk** granularity for **one** chunking strategy. Metrics collapse
hits to the parent PMID, so a retriever cannot score by returning five chunks of one passage.
`pipeline/retrieval/runner.py::run_retrieval` composes them with the optional layers below.

## Base retrievers

| name | family | how | index | cost / latency |
|---|---|---|---|---|
| `bm25` | lexical | Lucene BM25 (`bm25s`, k1=1.2, b=0.75) over lower-cased, stop-worded, Snowball-stemmed tokens; one in-process index per strategy cached in `.cache/bm25/<strategy>/` | built from `chunk_text` on first use (~10 s / 50k chunks) | ~20 ms |
| `dense` | semantic | cosine over `MedEmbed-small-v0.1` (384-d) embeddings in pgvector; HNSW where indexed (`ef_search = max(64, 2k)`: `passage`, `recursive_128_32`), exact scan otherwise | partial HNSW | 15–60 ms (HNSW) / 50–200 ms (exact) |
| `hybrid` | lexical + semantic | `dense` top-3k ∪ `bm25` top-3k → reciprocal-rank fusion (k=60) → top-k. Per-source ranks kept on each hit (`retriever_ranks`) | both | sum of both |
| `grep` | literal | terms = scispaCy entities in the question + remaining content words; per term a word-boundary regex scan over `passages.text`; passage score = Σ log(N/df_term) over matched terms; best chunk per passage = most matched terms. No embeddings, no LLM | none (sequential scan of 20k abstracts) | 0.5–6 s (scales with #terms) |
| `kg` | graph | question entities → `kg_entities` (exact, else trigram similarity ≥ 0.55) → passages via `passage_ids`, scored Σ log(N/df); 1-hop: co-occurring entities of the top-40 passages add damped weight (0.3); final chunk per passage chosen by cosine to the query | `kg_entities` arrays + trigram GIN | 0.3–3 s |

**What `grep` is for.** It is the retrieval an agent gets from a literal search tool. It is
strong on exact gene/drug names (`RANKL`, `metformin`) and weak on morphology and synonyms
(`secreted` vs `secretion`) — which is exactly the gap stemmed BM25 and dense retrieval close.
Keeping it in the matrix makes that gap measurable rather than assumed.

**What `kg` is for.** A deliberately simple GraphRAG: no LLM-built triples, no UMLS linking —
scispaCy entity mentions and co-occurrence only. It is *explainable* (every hit carries the entity
path that produced it, shown in the UI) and it is the only retriever whose recall does not depend
on lexical or embedding similarity to the question. Its weakness is entity noise
("clinical trials", "consistent with" are entities too) which the df caps only partly remove.

## Composable layers

| layer | options | what changes |
|---|---|---|
| **query transform** (`transforms.py`) | `none` · `hyde` · `multi_query` · `decompose` | LLM produces extra queries (a hypothetical abstract / 3 paraphrases / 2-4 sub-questions); the base retriever runs once per query and results are RRF-fused. Original question is always included. |
| **rerank** (`reranker.py`) | off · on | `ncbi/MedCPT-Cross-Encoder` (PubMed click-log trained) rescores `candidate_k` chunks, keeps `top_k`. Pre-rerank score/rank preserved in `hit.meta`. |
| **expansion** (`expand.py`) | `none` · `window` · `parent` | What the generator sees: the chunk, chunk ± 1 neighbours (sentence-window), or the whole abstract (parent-document; duplicates collapse). Does not change *which* passages were retrieved — only context. |

A configuration is `RetrievalConfig(strategy, retriever, transform, rerank, expansion, top_k,
candidate_k)`; its label `sentence+hybrid+decompose+rerank+window` is the identifier used in
`query_logs.config`, eval results and the UI.

## Multi-hop

BioASQ list questions ("List signaling molecules that interact with EGFR") have gold evidence
spread over 10–20 passages — the same shape as MultiHop-RAG's cross-document queries. `decompose`
is the MultiHop-RAG style approach (sub-questions → retrieve each → fuse); `multi_query` and
`hyde` attack the same problem through vocabulary instead of structure. All three are free-tier
Gemini calls (`get_llm("transform")`, ≤ 512 output tokens).

## Adding a retriever

Subclass `Retriever`, set `name`, implement `retrieve()`; add to `RETRIEVERS` in `runner.py`.
It immediately appears in the API `/api/options`, the UI pickers and `--matrix` evals.
