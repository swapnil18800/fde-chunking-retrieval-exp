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

### 4.2 Tier 1 — cross-encoder reranking (`eval150_rerank`, 2026-09-18)

Same 150 questions; `ncbi/MedCPT-Cross-Encoder` rescored the top-30 candidates of bm25 / dense / hybrid
for every strategy, keeping 10. Raw tables: `evals/results/20260918T040455Z_eval150_rerank/`.

| config | recall@10 base | recall@10 +rerank | Δ | precision@5 base | +rerank | mrr base | +rerank | ndcg@10 | latency ms |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| passage+bm25+rerank | 0.533 | 0.568 | +0.035 | 0.513 | 0.552 | 0.794 | 0.803 | 0.661 | 1861 |
| passage+hybrid+rerank | 0.533 | 0.562 | +0.029 | 0.527 | 0.555 | 0.779 | 0.798 | 0.656 | 2565 |
| semantic+bm25+rerank | 0.479 | 0.516 | +0.037 | 0.479 | 0.525 | 0.772 | 0.802 | 0.613 | 1163 |
| recursive_128_32+bm25+rerank | 0.478 | 0.515 | +0.037 | 0.487 | 0.535 | 0.761 | 0.810 | 0.614 | 1012 |
| fixed_128_32+hybrid+rerank | 0.497 | 0.515 | +0.017 | 0.528 | 0.543 | 0.791 | 0.803 | 0.611 | 1914 |
| semantic+hybrid+rerank | 0.495 | 0.511 | +0.016 | 0.519 | 0.525 | 0.781 | 0.791 | 0.608 | 1950 |
| recursive_128_32+hybrid+rerank | 0.497 | 0.509 | +0.012 | 0.517 | 0.543 | 0.774 | 0.807 | 0.608 | 1831 |
| fixed_128_32+bm25+rerank | 0.485 | 0.507 | +0.022 | 0.505 | 0.541 | 0.760 | 0.806 | 0.610 | 965 |
| passage+dense+rerank | 0.460 | 0.500 | +0.040 | 0.472 | 0.517 | 0.728 | 0.750 | 0.593 | 2574 |
| sentence+hybrid+rerank | 0.484 | 0.489 | +0.005 | 0.489 | 0.511 | 0.795 | 0.808 | 0.592 | 1916 |
| sentence+dense+rerank | 0.471 | 0.489 | +0.017 | 0.497 | 0.508 | 0.781 | 0.798 | 0.591 | 2203 |
| semantic+dense+rerank | 0.468 | 0.485 | +0.017 | 0.500 | 0.508 | 0.766 | 0.786 | 0.587 | 2178 |
| sentence+bm25+rerank | 0.446 | 0.476 | +0.030 | 0.425 | 0.491 | 0.742 | 0.803 | 0.580 | 793 |
| fixed_128_32+dense+rerank | 0.453 | 0.469 | +0.016 | 0.488 | 0.524 | 0.764 | 0.779 | 0.572 | 1860 |
| recursive_128_32+dense+rerank | 0.449 | 0.457 | +0.008 | 0.492 | 0.508 | 0.753 | 0.771 | 0.559 | 1827 |

**What reranking does**

1. **Consistent lift everywhere**: recall@10 +0.023 on average (min +0.005, max +0.040),
   precision@5 +0.030, MRR +0.025. No configuration got worse.
2. **The gap between chunking strategies shrinks.** Without rerank the fine strategies trailed `passage`
   by 3.6–4.9 points (hybrid); with rerank `semantic`/`recursive`/`fixed` sit at 0.51–0.52 vs `passage`
   0.56–0.57. MRR converges to 0.79–0.81 for almost every cell — the cross-encoder, which reads the
   query and the chunk together, largely undoes first-stage ranking differences.
3. **Dense gains the most in relative terms** (`passage+dense` 0.460 → 0.500, `sentence+dense` 0.471 →
   0.489) but bm25-fed candidates still win: the reranker can only reorder what the first stage
   surfaces, and BM25 surfaces more gold abstracts in its top-30 on this entity-heavy corpus.
4. **Best overall cell: `passage+bm25+rerank` — recall@10 0.568, precision@5 0.552, MRR 0.803**, at
   ~1.9 s per question (rerank ≈ +0.5–1.2 s on Apple M4). `recursive_128_32+bm25+rerank` is the best
   sub-abstract config (0.515, MRR 0.810, ~1.0 s) and the natural pick when the generator needs
   tighter context.

### 4.3 Tier 1 — LLM query transforms (`eval150_transforms`, 2026-09-18)

`hybrid` retriever on `passage` and `sentence`; each transform's extra queries are RRF-fused with the
original question (Gemini free tier, ~150 LLM calls per cell, 15–20 min each). Baselines: `passage+hybrid`
0.533 / MRR 0.779 · `sentence+hybrid` 0.484 / MRR 0.795. Raw tables: `evals/results/20260918T051228Z_eval150_transforms/`.

| config                      |   recall@5 |   recall@10 |   precision@5 |   mrr |   ndcg@10 |   hit@5 |   latency_ms |   errors |
|:----------------------------|-----------:|------------:|--------------:|------:|----------:|--------:|-------------:|---------:|
| passage+hybrid+hyde         |      0.454 |       0.558 |         0.544 | 0.827 |     0.65  |   0.867 |      6404.15 |        0 |
| passage+hybrid+multi_query  |      0.434 |       0.547 |         0.523 | 0.775 |     0.62  |   0.853 |      7871.67 |        0 |
| passage+hybrid+decompose    |      0.422 |       0.533 |         0.517 | 0.781 |     0.612 |   0.84  |      7078    |        0 |
| sentence+hybrid+hyde        |      0.425 |       0.486 |         0.512 | 0.783 |     0.58  |   0.853 |      7149.9  |        0 |
| sentence+hybrid+multi_query |      0.41  |       0.478 |         0.489 | 0.786 |     0.573 |   0.833 |      8631.96 |        0 |
| sentence+hybrid+decompose   |      0.421 |       0.476 |         0.501 | 0.787 |     0.572 |   0.853 |      7225.19 |        0 |

**What the transforms say**

1. **HyDE is the one transform that pays — and only on whole abstracts**: `passage+hybrid+hyde` reaches
   recall@10 0.558 (+2.5) and **MRR 0.827, the best first-hit ranking of any configuration**, including
   the reranked ones. A hypothetical abstract is a query at the same granularity as the documents; on
   `sentence` chunks the same draft is neutral (0.486 vs 0.484).
2. **multi_query widens recall slightly without improving the top of the list** (+1.4 recall@10 on
   `passage`, MRR unchanged; slightly negative on `sentence`). Paraphrases mostly re-find the same
   abstracts.
3. **decompose (multi-hop style) does not help average recall here** (0.533 = baseline on `passage`,
   −0.8 on `sentence`). BioASQ questions are single-hop by construction — the gold passages each answer
   the question independently — so splitting the question dilutes the query. The list-question subset
   is where it could still matter; that needs a per-type read of `per_question.csv`.
4. **Cost**: every transform adds one LLM round-trip (2–5 s on the free tier) and 2–4 extra retrievals;
   latency goes from ~1.3 s to 6–9 s per question. Reranking (+0.5–1 s) buys a similar recall gain for a
   fraction of the latency, but HyDE's MRR gain is unique — `hyde + rerank` is the obvious untested combo.

### 4.4 Tier 1 — summary across sweeps

| lever | best cell | recall@10 | MRR | Δ vs `passage+hybrid` (0.533 / 0.779) |
|---|---|--:|--:|---|
| chunking alone | `passage` (whole abstract) | 0.533 | 0.779 | fine chunks cost 3.6–4.9 pts |
| retriever alone | `bm25` = `hybrid` on `passage` | 0.533 | 0.794 / 0.779 | dense −7.3 · grep −8.6 · kg −18.7 |
| + MedCPT rerank | `passage+bm25+rerank` | **0.568** | 0.803 | +3.5 recall, +0.02 MRR, +0.5–1 s |
| + HyDE | `passage+hybrid+hyde` | 0.558 | **0.827** | +2.5 recall, +0.05 MRR, +5 s |
| best sub-abstract | `recursive_128_32+bm25+rerank` | 0.515 | 0.810 | tightest context for the generator |
