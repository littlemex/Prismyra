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

A tag can ask for a different feature scheme instead of hashed TF-IDF, by naming one in its spec entry's
``features`` field (``prismyra.learn.spec.TagSpec.features``). "Different" means: a hand-written function from
input (document, question) to a dict of named, numeric features, registered once under that name with
``register_features`` -- a game's own board state turned into "is this direction safe", "how many cells can it
still reach" and so on, rather than a bag of hashed JSON tokens that never represents a board's connectivity at
all (learn2's own learning-curve finding, RUN-learn2.md section 7: hashed tokens have no way to carry "which
cells are reachable from here"). The registration mechanism itself is kept deliberately small -- a name, a
function, nothing else -- because *what* a tag's structured features should be is a per-tag, per-domain decision
this module has no business making; `Vectorizer` only needs to know that a name resolves to *some* function
shaped like one, and that calling it twice on the same input gives the same dict back.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable
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


#: ``(document, question) -> {feature name: numeric value}``. The whole shape a registered feature builder
#: has to have. ``question`` exists because the design (section 3) describes the input as "文書と質問", not
#: "文書" alone, even though every builder written so far ignores it (a tag's question text is a constant
#: within that tag's own training set -- see this module's own docstring on why `tokenize` never looks at it
#: either); a builder for a tag whose single context is asked more than one differently-shaped question would
#: be the first to actually need it.
FeatureFn = Callable[[str, str], dict[str, float]]


class FeatureError(Exception):
    """A spec names a feature builder (`TagSpec.features`) that nothing has registered yet, or two different
    functions tried to register under the same name. Raised at *use* time (inside `Vectorizer.fit`, inside
    `fit.py`'s `run()`), never at spec-load time -- `spec.load_spec` validates that `features` is a string;
    whether that string is actually registered depends on which modules the caller has imported, which a
    file-format module cannot know in advance.
    """


_FEATURE_BUILDERS: dict[str, FeatureFn] = {}


def register_features(name: str, fn: FeatureFn) -> None:
    """The entire registration mechanism, on purpose: a name maps to one function. Registering the same name
    with the same function object twice (a module imported more than once, which `register_snake_board_v1`-style
    call-at-import-time modules do every time something else imports them) is a no-op, not an error -- only a
    second, *different* function trying to claim a name already taken is refused, the same "a typo ships
    silently" concern `spec.py`'s own duplicate checks exist for.
    """
    existing = _FEATURE_BUILDERS.get(name)
    if existing is not None and existing is not fn:
        raise FeatureError(f"a feature builder named {name!r} is already registered")
    _FEATURE_BUILDERS[name] = fn


def get_feature_builder(name: str) -> FeatureFn:
    try:
        return _FEATURE_BUILDERS[name]
    except KeyError:
        registered = sorted(_FEATURE_BUILDERS)
        raise FeatureError(f"no feature builder named {name!r} is registered (registered: {registered})") from None


@dataclass(frozen=True)
class Vectorizer:
    """One tag's chosen way of turning (document, question) into a fixed-width numeric vector -- either the
    hashed-TF-IDF default (``features_name is None``, ``hashing`` set: today's only behaviour, unchanged by any
    of this) or a registered structured-feature builder (``features_name`` set, ``names`` fixing the column
    order). Exactly one of ``hashing``/``names`` is set, matching which branch ``features_name`` picked.

    Column order for the structured branch is fixed once, at `fit`, from the *first* training record's keys --
    a builder is expected to return the same key set on every call (its docstring's contract, not this class's
    to enforce), and a later call whose dict is missing a key seen at fit time contributes 0.0 for that column
    rather than raising, the same tolerance `HashingTfidf.transform` already has for a token it never saw.
    """

    features_name: str | None
    hashing: HashingTfidf | None
    names: tuple[str, ...] | None

    def transform(self, context: str, question: str = "") -> list[float]:
        if self.features_name is None:
            assert self.hashing is not None
            return self.hashing.transform(tokenize(context))
        assert self.names is not None
        fn = get_feature_builder(self.features_name)
        values = fn(context, question)
        return [float(values.get(name, 0.0)) for name in self.names]

    @staticmethod
    def fit(records: list[tuple[str, str]], *, features_name: str | None, dims: int) -> Vectorizer:
        """``records`` is a list of ``(context, question)`` pairs from the training set -- the hashed branch
        only ever looks at the ``context`` half (fitting IDF over the training corpus, same as before this
        class existed); the structured branch calls the registered builder once, on the first record, purely
        to learn its fixed set of output keys."""
        if features_name is None:
            corpus = [tokenize(context) for context, _ in records]
            return Vectorizer(features_name=None, hashing=HashingTfidf.fit(corpus, dims), names=None)
        fn = get_feature_builder(features_name)
        names = tuple(sorted(fn(*records[0]).keys())) if records else ()
        return Vectorizer(features_name=features_name, hashing=None, names=names)
