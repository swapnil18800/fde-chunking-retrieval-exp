"""Biomedical NLP helpers built on the scispaCy `en_core_sci_sm` pipeline (sentences + entities).

The 0.5.4 model ships a config.cfg written for spaCy 3.7 where a boolean var is quoted
("False"); confection >=1.x rejects that. We load the config ourselves, fix the var, and
hydrate the pipeline from disk — no library patching, no copied model dirs.
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.language import Language

MODEL = "en_core_sci_sm"
# noise filters for KG entity names
_BAD_ENT = re.compile(r"^[\W\d_]+$|^(fig|table|e\.g|i\.e|et al|study|studies|patients?|results?|data|"
                      r"analysis|method|methods|group|groups|cases?|levels?|effects?|role|use|number|"
                      r"rate|rates|years?|days?|weeks?|months?|treatment|control|controls)$", re.I)


def _fix_quoted_bools(node) -> None:
    """In-place: turn "False"/"True" strings into bools anywhere in the (nested) config."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and v in ("False", "True"):
                node[k] = v == "True"
            else:
                _fix_quoted_bools(v)
    elif isinstance(node, list):
        for v in node:
            _fix_quoted_bools(v)


@lru_cache
def load_sci_nlp(disable: tuple[str, ...] = ()) -> Language:
    import importlib

    pkg = importlib.import_module(MODEL)
    model_dir = next(Path(pkg.__file__).parent.glob(f"{MODEL}-*"))
    config = spacy.util.load_config(model_dir / "config.cfg", interpolate=False)
    _fix_quoted_bools(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # spacy_version mismatch warning (3.7 model on 3.8)
        nlp = spacy.util.load_model_from_config(config, disable=list(disable))
        nlp.from_disk(model_dir, exclude=list(disable))
    return nlp


def sentence_offsets(doc) -> list[int]:
    """Char start offset of every sentence in a spaCy Doc."""
    return [s.start_char for s in doc.sents]


def clean_entity(text: str) -> str | None:
    t = re.sub(r"\s+", " ", text).strip(" ,.;:()[]\"'").lower()
    if len(t) < 3 or len(t) > 60 or _BAD_ENT.match(t):
        return None
    return t


def entities(doc) -> dict[str, int]:
    """Cleaned entity surface forms → mention count."""
    out: dict[str, int] = {}
    for e in doc.ents:
        t = clean_entity(e.text)
        if t:
            out[t] = out.get(t, 0) + 1
    return out
