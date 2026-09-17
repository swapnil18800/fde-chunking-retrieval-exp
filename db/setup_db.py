"""Apply db/schema.sql to the configured database.

    uv run python db/setup_db.py            # idempotent create-if-not-exists
    uv run python db/setup_db.py --reset    # DROP all project tables first (asks for confirmation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402

TABLES = ["eval_results", "eval_runs", "query_logs", "kg_edges", "kg_mentions", "kg_entities",
          "chunks", "chunk_strategies", "eval_sets", "qa_pairs", "passages"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop all project tables before applying schema")
    ap.add_argument("--yes", action="store_true", help="skip confirmation for --reset")
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "db")
    log = get_logger("db.setup")
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with connect() as conn:
        if args.reset:
            if not args.yes and input("Drop ALL project tables? type 'yes': ") != "yes":
                sys.exit(1)
            with conn.cursor() as cur:
                cur.execute("drop view if exists chunk_text")
                for t in TABLES:
                    cur.execute(f"drop table if exists {t} cascade")
            log.warning("[db] dropped %d tables", len(TABLES))
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute("select table_name from information_schema.tables where table_schema='public' order by 1")
            tables = [r["table_name"] for r in cur.fetchall()]
            cur.execute("select pg_size_pretty(pg_database_size(current_database())) as size")
            size = cur.fetchone()["size"]
        conn.commit()
    log.info("[db] schema applied. tables=%s db_size=%s", tables, size)


if __name__ == "__main__":
    main()
