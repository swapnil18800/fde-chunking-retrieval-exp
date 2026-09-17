"""Structured logging: human-readable console + JSONL file per process.

Every log record carries the `[tag]` convention used across the repo
(`[ingest]`, `[chunk]`, `[embed]`, `[retrieve]`, `[graph]`, `[eval]`, `[api]`)
so `grep '\\[retrieve\\]' logs/app.jsonl` isolates a subsystem. Per-question
pipeline records are *also* persisted to the `query_logs` table (see
pipeline/query_log.py) — the file log is for process-level debugging.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.logging import RichHandler

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k.startswith("ctx_"):
                payload[k[4:]] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", log_dir: Path | None = None, name: str = "app") -> None:
    """Idempotent. Console via rich, JSONL at <log_dir>/<name>.jsonl."""
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(level.upper())
    console = RichHandler(rich_tracebacks=False, show_path=False, markup=False)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.jsonl", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
