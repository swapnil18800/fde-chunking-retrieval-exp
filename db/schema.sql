-- fde-chunking-retrieval-exp — Supabase Postgres schema
-- Apply with: uv run python db/setup_db.py   (add --reset to drop everything first)
--
-- Storage philosophy: chunks are stored as CHAR OFFSETS into their parent passage
-- (no duplicated text) and embeddings as halfvec (2 bytes/dim). This keeps ~500k
-- chunks x 384 dims inside the Supabase free-tier 500 MB budget.

create extension if not exists vector;

-- ── Corpus ────────────────────────────────────────────────────────────────────
-- 40,221 PubMed abstracts from rag-datasets/rag-mini-bioasq. id = PMID.
create table if not exists passages (
    id               bigint primary key,
    text             text    not null,          -- whitespace-normalised
    n_chars          int     not null,
    n_words          int     not null,
    n_tokens         int,                       -- cl100k tokens
    sentence_offsets int[],                     -- char start of each sentence (filled by preprocess_nlp.py)
    kg_entity_ids    int[]                      -- entities mentioned (filled by preprocess_nlp.py)
);

-- 4,719 BioASQ questions with gold passage ids
create table if not exists qa_pairs (
    id                   int primary key,
    question             text not null,
    answer               text not null,
    relevant_passage_ids bigint[] not null,     -- gold ids still present in the corpus
    n_gold_total         int,                   -- gold count in the original dataset (before subsampling)
    question_type        text                   -- yesno | list | factoid | summary (heuristic)
);

-- provenance of corpus-level operations (e.g. the free-tier subsample recipe)
create table if not exists corpus_meta (
    key        text primary key,
    value      jsonb,
    updated_at timestamptz default now()
);

-- Named question subsets: smoke (5) / eval150 (150) / ...
create table if not exists eval_sets (
    name  text not null,
    qa_id int  not null references qa_pairs(id) on delete cascade,
    position int not null,
    primary key (name, qa_id)
);

-- ── Chunking ──────────────────────────────────────────────────────────────────
create table if not exists chunk_strategies (
    name            text primary key,
    description     text,
    params          jsonb,
    embedding_model text,
    embedding_dim   int,
    n_chunks        int,
    avg_tokens      real,
    built_at        timestamptz default now()
);

create table if not exists chunks (
    id          bigserial primary key,
    strategy    text   not null references chunk_strategies(name) on delete cascade,
    passage_id  bigint not null references passages(id) on delete cascade,
    chunk_index int    not null,                -- order within passage
    char_start  int    not null,
    char_end    int    not null,
    n_tokens    int,
    prefix      text,                           -- optional context prepended at embed time (contextual strategies)
    embedding   halfvec(384)
);
create index if not exists chunks_strategy_passage_idx on chunks (strategy, passage_id, chunk_index);

-- Chunk text is derived, never duplicated:
create or replace view chunk_text as
select c.id, c.strategy, c.passage_id, c.chunk_index, c.char_start, c.char_end, c.n_tokens, c.prefix,
       substr(p.text, c.char_start + 1, c.char_end - c.char_start) as text
from chunks c join passages p on p.id = c.passage_id;

-- ── Knowledge graph (scispaCy entities → passages, stored as arrays to stay small) ─────
-- entity→passage adjacency lives in kg_entities.passage_ids; passage→entity in passages.kg_entity_ids.
-- Co-occurrence (1-hop expansion) is computed at query time from those two arrays.
create table if not exists kg_entities (
    id          serial primary key,
    name        text unique not null,           -- lower-cased surface form
    doc_freq    int  not null default 0,
    passage_ids bigint[]
);
-- fuzzy entity lookup for query terms
create extension if not exists pg_trgm;
create index if not exists kg_entities_name_trgm on kg_entities using gin (name gin_trgm_ops);

-- ── Observability: every question that flows through the pipeline ────────────
create table if not exists query_logs (
    id             uuid primary key default gen_random_uuid(),
    created_at     timestamptz default now(),
    source         text,                        -- api | eval | smoke
    question       text not null,
    qa_id          int references qa_pairs(id),
    config         jsonb,                       -- chunk_strategy, retriever, rerank, top_k, llm ...
    stages         jsonb,                       -- [{name, ms, detail}]  per pipeline node
    retrieved      jsonb,                       -- final ranked [{chunk_id, passage_id, score, rank, retriever_ranks}]
    answer         text,
    citations      jsonb,                       -- [{n, chunk_id, passage_id, char_start, char_end, url}]
    metrics        jsonb,                       -- retrieval metrics when qa_id is known
    latency_ms     int,
    tokens         jsonb,                       -- {prompt, completion, model}
    trace_provider text,
    trace_id       text,
    trace_url      text,
    error          text
);
create index if not exists query_logs_created_idx on query_logs (created_at desc);

-- ── Evaluation runs ───────────────────────────────────────────────────────────
create table if not exists eval_runs (
    id           uuid primary key default gen_random_uuid(),
    created_at   timestamptz default now(),
    finished_at  timestamptz,
    name         text,
    kind         text,                          -- retrieval | ragas
    eval_set     text,
    config       jsonb,
    summary      jsonb,
    n_questions  int,
    status       text default 'running'
);
create table if not exists eval_results (
    run_id       uuid not null references eval_runs(id) on delete cascade,
    config       text not null default '',          -- RetrievalConfig.label()
    qa_id        int  not null references qa_pairs(id),
    metrics      jsonb,
    retrieved    jsonb,
    answer       text,
    latency_ms   int,
    query_log_id uuid,
    primary key (run_id, config, qa_id)
);
