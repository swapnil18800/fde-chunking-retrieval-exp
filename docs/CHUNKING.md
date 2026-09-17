# Chunking strategies

All chunkers implement `pipeline/chunking/base.py::Chunker` and return `(char_start, char_end)`
spans into the **normalised passage text** (whitespace-collapsed at ingest). Register/lookup by
name in `pipeline/chunking/strategies.py::build_chunker`. Every chunk set is embedded with the
same model (`chunk_strategies.embedding_model`) so retrieval differences come from chunking alone.

## Why these five (and not, say, section-aware)

The corpus is PubMed *abstracts*: average 147 words / 225 tokens, ~6.7 sentences, **no section
headers** (0 of 100 sampled had `BACKGROUND:`/`METHODS:` markers), hard-wrapped at 80 chars in the
raw dump. So the interesting axis is *granularity below the abstract*, and the baseline is the
abstract itself — which is also exactly the granularity of the gold labels (PMIDs).

| name | unit | chunks / passage | avg tokens | hypothesis |
|---|---|---|---|---|
| `passage` | whole abstract | 1.0 | ~225 | Gold-label granularity; best recall ceiling, noisiest generator context |
| `fixed_128_32` | 128-token sliding window, 32 overlap, word-snapped | ~2.8 | ~95 | The naive default; splits sentences mid-way — does it hurt dense retrieval? |
| `recursive_128_32` | LangChain `RecursiveCharacterTextSplitter` (tiktoken length), separators `. ? ! ; , space` | ~2.9 | ~90 | Same budget as fixed but sentence/clause aware — isolates the boundary effect |
| `sentence` | one scispaCy sentence; fragments < 6 tokens merged into the previous sentence | ~6.7 | ~33 | Finest granularity: precise dense matches, but a single sentence rarely answers a BioASQ list question — needs window/parent expansion |
| `semantic` | consecutive sentences grouped until adjacent-sentence cosine distance > global threshold (p75 of adjacent distances on a 1,500-passage sample) or 256 tokens | ~2.0 | ~110 | Topic-coherent groups without a fixed size; the "smart" split |

Retrieval-time expansions (`pipeline/retrieval/expand.py`) turn the fine strategies into the classic
patterns: `sentence + window` = *sentence-window retrieval*; `any + parent` = *parent-document /
small-to-big*.

## Implementation notes

- **Token counting** uses `cl100k_base` (tiktoken) everywhere, including LangChain's splitter
  (`from_tiktoken_encoder`). `pipeline/tokens.py::token_char_offsets` maps tokens → char offsets
  through the UTF-8 byte stream so fixed windows can be snapped to word boundaries.
- **LangChain start offsets** — `add_start_index=True` mixes token overlap with char offsets and
  returns `-1` on overlapping pieces; we locate pieces ourselves with `str.find` from a moving cursor.
- **Sentences** come from scispaCy `en_core_sci_sm` (parser-based segmentation handles "Fig. 2",
  "e.g.", "5.4%"), computed once in `preprocess_nlp.py` and stored in `passages.sentence_offsets`.
  A regex fallback exists for text without offsets (the Chunk Explorer's paste-your-own mode).
- **Semantic threshold** is *global*, not per-document: LangChain's percentile-per-document
  approach is unstable on 6-sentence documents. The calibrated threshold is stored in
  `chunk_strategies.params.threshold`.
- **Storage**: offsets not text; `halfvec(384)`; partial HNSW indexes (`chunks_hnsw_<strategy>`, m=16,
  ef_construction=96) on `passage` and `recursive_128_32` only — `fixed_128_32`, `sentence` and `semantic`
  are exact-scanned to stay inside the free tier (see ARCHITECTURE.md §6).

## Adding a strategy

1. Subclass `Chunker`, set `spec = ChunkerSpec(name, description, params)`, implement `chunk()`
   (or `chunk_batch()` if you need batching — see `SemanticChunker`).
2. Add a branch in `build_chunker()` and, if it should be part of the default matrix, to
   `DEFAULT_STRATEGIES`.
3. `uv run python db/ingestion/build_chunks.py --strategies <name>` → chunks + embeddings + HNSW.
   The BM25 cache for the new strategy is built lazily on first use.
4. Nothing else changes: the UI reads `chunk_strategies`, the eval matrix uses `--matrix`.

Ideas not implemented (and why): *contextual retrieval* (LLM-written chunk context) — ~280k LLM
calls on the free tier is not realistic; *late chunking* — needs a long-context embedding model
(jina-v2/v3) rather than MedEmbed; *proposition chunking* — same LLM-cost problem. A cheap
NER-based variant (`[entities] + sentence`) is a natural next experiment.
