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

Sweeps: the 5 × 5 base matrix; `+rerank` (MedCPT over 30 candidates) for bm25/dense/hybrid; LLM query
transforms (`hyde`/`multi_query`/`decompose`). **Expansion (`window`/`parent`) is not a Tier-1 axis**: it
changes the context handed to the generator, not which passages rank, so passage-level metrics are
identical by construction (verified: `sentence+hybrid+window` ≡ `sentence+hybrid`). It is evaluated in Tier 2.

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

### 4.1 Tier 1 — base matrix (`eval150_matrix`, 2026-09-18)

`eval150` · 150 questions · top_k=10 · candidate_k=30 · no transform / rerank / expansion ·
raw tables: `evals/results/20260918T024231Z_eval150_matrix/`

**recall@10 — chunking × retriever**

| strategy         |   bm25 |   dense |   grep |   hybrid |    kg |
|:-----------------|-------:|--------:|-------:|---------:|------:|
| fixed_128_32     |  0.485 |   0.453 |  0.447 |    0.497 | 0.346 |
| passage          |  0.533 |   0.46  |  0.447 |    0.533 | 0.346 |
| recursive_128_32 |  0.478 |   0.449 |  0.447 |    0.497 | 0.346 |
| semantic         |  0.479 |   0.468 |  0.447 |    0.495 | 0.346 |
| sentence         |  0.446 |   0.471 |  0.447 |    0.484 | 0.346 |

**All 25 cells, sorted by recall@10**

| config                  |   recall@5 |   recall@10 |   precision@5 |   mrr |   ndcg@10 |   hit@5 |   latency_ms |   errors |
|:------------------------|-----------:|------------:|--------------:|------:|----------:|--------:|-------------:|---------:|
| passage+hybrid          |      0.435 |       0.533 |         0.527 | 0.779 |     0.61  |   0.853 |     1329.12  |        0 |
| passage+bm25            |      0.435 |       0.533 |         0.513 | 0.794 |     0.622 |   0.867 |      635.933 |        0 |
| fixed_128_32+hybrid     |      0.439 |       0.497 |         0.528 | 0.791 |     0.59  |   0.867 |     1302.29  |        0 |
| recursive_128_32+hybrid |      0.428 |       0.497 |         0.517 | 0.774 |     0.583 |   0.847 |     1357.87  |        0 |
| semantic+hybrid         |      0.433 |       0.495 |         0.519 | 0.781 |     0.587 |   0.853 |     1298.43  |        0 |
| fixed_128_32+bm25       |      0.424 |       0.485 |         0.505 | 0.76  |     0.57  |   0.847 |      550.647 |        0 |
| sentence+hybrid         |      0.413 |       0.484 |         0.489 | 0.795 |     0.577 |   0.833 |     1274.73  |        0 |
| semantic+bm25           |      0.413 |       0.479 |         0.479 | 0.772 |     0.563 |   0.847 |      540.487 |        0 |
| recursive_128_32+bm25   |      0.416 |       0.478 |         0.487 | 0.761 |     0.56  |   0.853 |      549.533 |        0 |
| sentence+dense          |      0.416 |       0.471 |         0.497 | 0.781 |     0.563 |   0.847 |     1675.59  |        0 |
| semantic+dense          |      0.417 |       0.468 |         0.5   | 0.766 |     0.556 |   0.853 |     1330.91  |        0 |
| passage+dense           |      0.372 |       0.46  |         0.472 | 0.728 |     0.543 |   0.8   |     1394.37  |        0 |
| fixed_128_32+dense      |      0.401 |       0.453 |         0.488 | 0.764 |     0.549 |   0.847 |     1334.95  |        0 |
| recursive_128_32+dense  |      0.395 |       0.449 |         0.492 | 0.753 |     0.545 |   0.82  |     1310.69  |        0 |
| fixed_128_32+grep       |      0.384 |       0.447 |         0.451 | 0.747 |     0.541 |   0.793 |      556.7   |        0 |
| sentence+grep           |      0.384 |       0.447 |         0.451 | 0.747 |     0.541 |   0.793 |      690.047 |        0 |
| semantic+grep           |      0.384 |       0.447 |         0.451 | 0.747 |     0.541 |   0.793 |      522.007 |        0 |
| recursive_128_32+grep   |      0.384 |       0.447 |         0.451 | 0.747 |     0.541 |   0.793 |      602.3   |        0 |
| passage+grep            |      0.384 |       0.447 |         0.451 | 0.747 |     0.541 |   0.793 |     1804.09  |        0 |
| sentence+bm25           |      0.374 |       0.446 |         0.425 | 0.742 |     0.519 |   0.807 |      552.567 |        0 |
| fixed_128_32+kg         |      0.266 |       0.346 |         0.36  | 0.534 |     0.396 |   0.64  |     1147.49  |        0 |
| recursive_128_32+kg     |      0.266 |       0.346 |         0.36  | 0.534 |     0.396 |   0.64  |     1269.47  |        0 |
| passage+kg              |      0.266 |       0.346 |         0.36  | 0.534 |     0.396 |   0.64  |     2621.9   |        0 |
| semantic+kg             |      0.266 |       0.346 |         0.36  | 0.534 |     0.396 |   0.64  |     1242.17  |        0 |
| sentence+kg             |      0.266 |       0.346 |         0.36  | 0.534 |     0.396 |   0.64  |     1315.64  |        0 |

**recall@10 by question type** (selected configs; gold passages per question: factoid 5.4 · list 8.5 · summary 8.1 · yes/no 7.6)

| config          | factoid | list  | summary | yesno |
|:----------------|--------:|------:|--------:|------:|
| passage+bm25    |   0.522 | 0.476 |   0.581 | 0.589 |
| passage+hybrid  |   0.524 | 0.476 |   0.558 | 0.597 |
| passage+dense   |   0.424 | 0.394 |   0.512 | 0.556 |
| semantic+hybrid |   0.507 | 0.430 |   0.508 | 0.548 |
| sentence+hybrid |   0.503 | 0.424 |   0.491 | 0.525 |
| sentence+dense  |   0.468 | 0.418 |   0.471 | 0.535 |
| passage+grep    |   0.438 | 0.387 |   0.380 | 0.550 |
| passage+kg      |   0.296 | 0.334 |   0.343 | 0.419 |

**What the matrix says**

1. **Whole-abstract chunks win on recall (0.533) — and it is partly structural.** Gold labels are
   whole abstracts, so any sub-passage strategy can only tie `passage` if all its chunks of a gold
   abstract are found. Every finer strategy lands at 0.48–0.50 with `hybrid`; the *gap between fine
   strategies* (fixed 0.497 · recursive 0.497 · semantic 0.495 · sentence 0.484) is small — how you
   cut below the abstract matters far less than whether you cut at all. Precision@5 is flat
   (0.49–0.53) across strategies, so the generator sees similar noise either way.
2. **BM25 beats dense on long chunks; dense beats BM25 on short ones.** `passage`: bm25 0.533 vs
   dense 0.460. `sentence`: dense 0.471 vs bm25 0.446. BioASQ questions are entity-heavy (gene /
   drug / disease names) which favours exact lexical matching against a 225-token abstract; a
   384-d small embedder compresses a whole abstract into one vector and loses those specifics,
   but represents a single sentence well. `semantic` chunks (topic-coherent ~110 tokens) are the
   crossover point where the two families are closest (0.479 vs 0.468).
3. **Hybrid (RRF) is the safe default**: best or tied-best in every row (+0.01–0.04 over the better
   of its two parts on fine strategies; equal to BM25 on `passage`). Cost: ~2× latency.
4. **`grep` (literal terms, no stemming, no embeddings) reaches 0.447** — 9 points under BM25 on the
   same abstracts. That gap is the value of stemming + IDF weighting over "search for the words in
   the question". It also ranks first-hit well (MRR 0.747), so as an *agent tool* it is a reasonable
   first probe, not a retrieval system.
5. **`kg` (entity co-occurrence graph) is the weakest at 0.346.** Surface-form entities without
   normalisation ("hirschsprung disease" ≠ "hirschsprung's disease" ≠ "hscr") fragment the graph,
   and generic entities ("clinical trials") add noise even after df caps. It is the only retriever
   whose hits carry an explanation (entity path) — a complement, not a replacement.
6. **`grep` and `kg` are identical across chunking strategies by construction** — they rank
   *passages* and then pick a chunk to display; passage-level metrics cannot change.
7. **Question type**: yes/no questions are easiest (0.59), list questions hardest (0.48 with the
   best config) — list answers spread over 8.5 gold abstracts and top-10 cannot hold them all. This
   is where `decompose` (multi-hop) and larger k should help (see 4.4).

