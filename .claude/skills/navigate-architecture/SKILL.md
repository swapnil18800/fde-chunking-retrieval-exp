---
name: navigate-architecture
description: Orient in the fde-chunking-retrieval-exp codebase — where a concern lives, how a question flows from API to answer, which table holds what. Use when asked "where is X", "how does Y work", or before any non-trivial change.
---

# Navigate the architecture

1. Read `docs/ARCHITECTURE.md` §4 (request lifecycle) and §5 (component map). Do not guess paths — every
   subsystem has exactly one entry point:
   - retrieval config & composition → `pipeline/retrieval/runner.py`
   - a specific retriever → `pipeline/retrieval/<name>.py`
   - chunking → `pipeline/chunking/strategies.py`
   - graph / generation / citations → `pipeline/graph.py`, `pipeline/prompts.py`, `pipeline/citations.py`
   - API surface → `app/main.py` (routes are grouped by comment banners: meta / ask-compare / explorer / kg / logs-evals)
   - UI tab → `frontend/src/pages/<Tab>.tsx`; shared widgets in `frontend/src/components/`
2. To see what data a stage produces, run one question and inspect the log row instead of reading code:
   ```bash
   curl -s -X POST localhost:8000/api/ask -H 'Content-Type: application/json' \
     -d '{"question":"Is RANKL secreted from the cells?","strategy":"passage","retriever":"hybrid","generate":false}' | python3 -m json.tool | head -80
   ```
3. Table → purpose: `docs/ARCHITECTURE.md` §3. Chunk text is a view (`chunk_text`), never a column.
4. Naming conventions: config label = `strategy+retriever[+transform][+rerank][+expansion]`; trace
   observation names are stable verbs; log tags are `[ingest] [chunk] [embed] [retrieve] [llm] [graph] [eval] [api]`.
5. Before proposing a change, check `docs/DIRECTORY_STRUCTURE.md` for the file's role and `CLAUDE.md`
   working rules (config/prompt/LLM/DB access go through single choke points).
