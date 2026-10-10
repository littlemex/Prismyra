"""Deterministic, input-only features for the stage 2 student. DISTILL-RL-DESIGN-v2.md section 3:

    特徴は入力だけから作る (ゲームの JSON は平らにする。短い文は hashing の TF-IDF)。

"Input-only" also means the question text is left out on purpose, not by oversight: every record under one tag
was matched to that tag by having the *same* question (`prismyra.learn.tagging`'s whole job), so within one
tag's training set the question text is a constant -- a feature that cannot vary carries no signal, and keeping
it would only make a human reading the feature list wonder whether it does.

One tokenizer handles both kinds of input the design names -- "a game's JSON" and "a short sentence" -- rather
than branching into two feature schemes a caller would need to know which one applies: see `tokenize`. One
hashed, IDF-weighted bag of those tokens turns a variable-length token list into the fixed-width vector LightGBM
needs: see `HashingTfidf`, fitted once over the training corpus and then applied unchanged to everything else,
including a live request's input at serving time (stage 4, not built here, but the artifact this produces
already carries what that would need).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(context: str) -> list[str]:
    """A context that parses as a JSON object or list is canonicalised (`json.dumps(..., sort_keys=True)`, so
    key order in the source text never matters) and flattened into ``"key=value"`` tokens for every leaf, in
    addition to the bare tokens the canonical form itself contains -- a Snake board's ``{"head_x": 3}``
    contributes ``head_x``, ``3`` and ``head_x=3``, the last of which is what lets a GBDT split on one field's
    value without that field needing its own numeric column. Anything that is not JSON (or parses to a bare
    scalar, which flattening would do nothing useful with) is tokenised as plain lowercased text -- the "short
    sentence" case.
    """
    try:
        obj = json.loads(context)
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict | list):
        tokens = _TOKEN_RE.findall(json.dumps(obj, sort_keys=True, ensure_ascii=False))
        tokens.extend(_flatten_pairs(obj))
        return tokens
    return _TOKEN_RE.findall(context.lower())


def _flatten_pairs(obj: object, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        # Sorted, not `obj.items()` in parse order: `json.dumps(obj, sort_keys=True)` above already makes the
        # bare-token half of `tokenize`'s output order-invariant, and this flattened half should keep the same
        # property -- two JSON texts that differ only in key order are the same input and must tokenize to the
        # same bag of tokens.
        for k, v in sorted(obj.items()):
            out.extend(_flatten_pairs(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_flatten_pairs(v, f"{prefix}{i}."))
    else:
        key = prefix.rstrip(".")
        if key:
            out.append(f"{key}={obj}")
    return out


def _bucket(token: str, dims: int) -> int:
    # md5, not Python's own `hash`: `hash("x")` is salted per-process (`PYTHONHASHSEED`) unless a caller has
    # disabled that, and a feature scheme whose bucket assignment can change between two runs of the identical
    # code is not reproducible -- exactly the determinism this module exists to hand the GBDT. lightgbm's own
    # `deterministic=True` promises nothing about an input encoding that is not deterministic to begin with.
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") % dims


@dataclass(frozen=True)
class HashingTfidf:
    """A fixed-size, hashed bag of tokens with IDF weights fitted once over a training corpus, then applied
    unchanged to every later input. `idf` travels inside a student artifact's own metadata (`fit.py`'s
    `StudentArtifact`) so nothing downstream ever has to refit it to reproduce a vector.
    """

    dims: int
    idf: tuple[float, ...]

    def transform(self, tokens: Iterable[str]) -> list[float]:
        tokens = list(tokens)
        counts = [0.0] * self.dims
        for t in tokens:
            counts[_bucket(t, self.dims)] += 1.0
        n = len(tokens) or 1
        return [(c / n) * self.idf[i] for i, c in enumerate(counts)]

    @staticmethod
    def fit(corpus: list[list[str]], dims: int) -> HashingTfidf:
        doc_count = [0] * dims
        for tokens in corpus:
            for b in {_bucket(t, dims) for t in tokens}:
                doc_count[b] += 1
        n_docs = len(corpus) or 1
        idf = tuple(math.log((n_docs + 1) / (doc_count[i] + 1)) + 1.0 for i in range(dims))
        return HashingTfidf(dims=dims, idf=idf)
