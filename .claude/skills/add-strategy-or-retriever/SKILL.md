---
name: add-strategy-or-retriever
description: Add a new chunking strategy, retriever, query transform, expansion mode or reranker so it appears in the matrix, API, UI and evals with no other changes. Includes storage-budget checks.
---

# Add a strategy / retriever

## New chunking strategy
1. `pipeline/chunking/strategies.py`: subclass `Chunker`; `spec = ChunkerSpec(name, description, params)`;
   implement `chunk(p: Passage) -> list[Chunk]` returning char spans into `p.text` (use `self._finalise`).
   Need sentences? set `requires_sentences = True` and use `p.sentences()`. Need embeddings at chunk time?
   `requires_embedder = True` and implement `chunk_batch`. Add a branch to `build_chunker()`.
2. **Budget first**: estimate chunks/passage × 20k × ~830 B (+ ~900 B if HNSW). Check
   `curl localhost:8000/api/health` → `db_size` (free tier 500 MB). Add the name to `NO_INDEX` in
   `db/ingestion/build_chunks.py` to skip HNSW.
3. `uv run python db/ingestion/build_chunks.py --strategies <name>` (use `--limit 500 --no-index` to dev).
4. Add to `DEFAULT_STRATEGIES` if it belongs in `--matrix`. Document in `docs/CHUNKING.md` table.
   The UI picker, Chunk Explorer and `/api/options` read `chunk_strategies` — nothing to change.

## New retriever
1. `pipeline/retrieval/<name>.py`: subclass `Retriever`, set `name`, implement
   `retrieve(query, strategy, k) -> list[Hit]` (chunk-level hits; fill `chunk_index/char_start/char_end`
   if you have them, `attach_text` fills text later). Put explainability into `hit.meta` (shown in UI).
2. Register in `RETRIEVERS` in `pipeline/retrieval/runner.py`. Add a one-liner to `HELP` in
   `frontend/src/components/ConfigPicker.tsx` and a row in `docs/RETRIEVAL.md`.
3. Test: `run_retrieval(q, RetrievalConfig(strategy='passage', retriever='<name>'))` on `smoke5`.

## New transform / expansion
- Transform: add a prompt + branch in `pipeline/retrieval/transforms.py::generate_queries`, name in `TRANSFORMS`.
- Expansion: branch in `pipeline/retrieval/expand.py::expand`, name in `EXPANSIONS`; set
  `hit.meta.context_text/context_start/context_end` (citations use them for highlighting).

## Swapping models
- Embeddings: change `EMBEDDING_MODEL`/`EMBEDDING_DIM` → **rebuild all chunk sets** (`--all --replace`); the
  `halfvec(384)` column type must match the dim. Retrieval mixes are prevented by `chunk_strategies.embedding_model`.
- Reranker: `RERANKER_MODEL` (any sentence-transformers CrossEncoder).
- LLM: `GEMINI_MODELS` list (round-robin) or `LLM_PROVIDER=deepseek`.
