"""Token counting + token→char offset mapping (cl100k_base). Shared by chunkers and the eval harness."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache
def enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(enc().encode(text))


def token_char_offsets(text: str) -> list[int]:
    """Char offset at which each token starts, plus a final entry == len(text).

    tiktoken works on bytes; we walk the UTF-8 byte stream and map cumulative byte
    positions back to char indices (positions inside a multi-byte char snap to that char).
    """
    e = enc()
    toks = e.encode(text)
    byte_lens = [len(b) for b in e.decode_tokens_bytes(toks)]
    byte_to_char: list[int] = []
    for i, ch in enumerate(text):
        byte_to_char.extend([i] * len(ch.encode("utf-8")))
    byte_to_char.append(len(text))
    out, pos = [], 0
    for bl in byte_lens:
        out.append(byte_to_char[min(pos, len(byte_to_char) - 1)])
        pos += bl
    out.append(len(text))
    return out
