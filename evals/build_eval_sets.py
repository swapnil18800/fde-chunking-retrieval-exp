"""Create the named question subsets used by every experiment and persist them (DB + JSON).

    uv run python evals/build_eval_sets.py            # eval150 + smoke5, deterministic (seed 42)

Sampling is stratified by question_type so the mix mirrors BioASQ (factoid/list/yesno/summary)
and we only keep questions whose gold passages number between 1 and 20 (extreme outliers with
30+ gold passages make recall@k meaningless). The JSON copy in evals/datasets/ is the artefact
that gets committed so results are reproducible without DB access.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402

OUT = Path(__file__).parent / "datasets"
SETS = {"eval150": 150, "smoke5": 5}
SEED = 42


def main() -> None:
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "eval")
    log = get_logger("eval.sets")
    rng = random.Random(SEED)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""select id, question, answer, relevant_passage_ids, question_type from qa_pairs
                       where cardinality(relevant_passage_ids) between 1 and 20 order by id""")
        rows = cur.fetchall()
        by_type: dict[str, list] = defaultdict(list)
        for r in rows:
            by_type[r["question_type"]].append(r)
        total = len(rows)
        picked: dict[str, list] = {}
        pool = []
        for qt, items in sorted(by_type.items()):
            rng.shuffle(items)
            k = round(SETS["eval150"] * len(items) / total)
            pool.extend(items[:k])
        rng.shuffle(pool)
        picked["eval150"] = pool[: SETS["eval150"]]
        # smoke: one of each type + one extra, taken from eval150 so smoke ⊂ eval150
        smoke, seen = [], set()
        for r in picked["eval150"]:
            if r["question_type"] not in seen:
                smoke.append(r)
                seen.add(r["question_type"])
        for r in picked["eval150"]:
            if len(smoke) >= SETS["smoke5"]:
                break
            if r not in smoke:
                smoke.append(r)
        picked["smoke5"] = smoke
        OUT.mkdir(exist_ok=True)
        for name, items in picked.items():
            cur.execute("delete from eval_sets where name = %s", (name,))
            cur.executemany("insert into eval_sets (name, qa_id, position) values (%s, %s, %s)",
                            [(name, r["id"], i) for i, r in enumerate(items)])
            (OUT / f"{name}.json").write_text(json.dumps(
                [{"qa_id": r["id"], "question": r["question"], "answer": r["answer"],
                  "question_type": r["question_type"], "relevant_passage_ids": r["relevant_passage_ids"]}
                 for r in items], indent=1))
            mix = defaultdict(int)
            for r in items:
                mix[r["question_type"]] += 1
            log.info("[eval] set %s: %d questions, mix=%s, avg gold=%.1f", name, len(items), dict(mix),
                     sum(len(r["relevant_passage_ids"]) for r in items) / len(items))
        conn.commit()


if __name__ == "__main__":
    main()
