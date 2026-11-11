"""Sentence-embedding feature builders for `prismyra.learn.features.register_features` (PREREG-R2 idea 2:
"better student features through Prismyra's own feature registry").

`tools/snake_features.py` turns one tag's structured input (a game board) into named, numeric columns by hand.
A short sentence has no board to parse -- the structured alternative here is a pretrained sentence encoder's
own embedding, read as `dim_0 .. dim_{N-1}`. Which encoder to use is deliberately not picked by this module:
PREREG-R2 asks for "a model chosen by a gold-label probe on the eval set, never on the certification set"
(`/Users/akazawt/tmp/tierbook/distill/probe/feat_probe.py`), which is an operator's own, out-of-band decision
for one tag's own data -- this module only registers one builder per candidate so that probe (or any later
caller) can ask for any of them by name, and `register_sentence_embedding` so a caller can register one this
module does not already name, without editing this file.

**`sentence-transformers` is an optional, lazily-imported dependency, not a dependency of `prismyra` itself.**
Importing this module (e.g. via `--features-module examples.features.sentence_embedding`) registers every
name in `CANDIDATE_MODELS` immediately -- cheap, a closure and a dict entry each -- but imports neither
`sentence_transformers` nor downloads any model weights until a closure is actually *called* (the first
`Vectorizer.fit`/`.transform` that uses one of these names), exactly the "off by default, pay only when used"
discipline the rest of `prismyra.learn` keeps. `pip install sentence-transformers` only when a tag actually
names one of these builders.

**Commercial-use licensing** (this module only registers candidates already cleared, the same
"commercially-clear" discipline `docs/DATA_PROVENANCE.md` applies to datasets, checked 2026-11-09):

| model | license | commercial use |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | MIT | yes |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | yes |
| `google/embeddinggemma-2` | Apache-2.0 (confirmed via the repo's own `license` tag, 2026-11-09) | yes |

`google/embeddinggemma-2` is new as of this round (2026-10-28): unlike the original `google/embeddinggemma-
300m`, which ships under the custom "Gemma Terms of Use" (a restricted-use license, not a candidate here for
exactly that reason), the Gemma-4-generation release switched to a plain Apache-2.0 grant -- confirmed by
reading the license page itself, not just trusting the tag, since Google's own model cards have in the past
pointed a "License" link at a terms-of-use page while the repo tag said something more permissive-sounding.
It is a native multimodal (text/image/video/audio) encoder with a 768-dim output on this project's text-only
use, considerably larger than the other two candidates (740M total parameters vs. tens of millions) -- the
probe this module exists to support is what decides whether that size buys anything on a 3-way sentiment
tag, not an assumption made here.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from prismyra.learn.features import register_features

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: The candidates this module ships a ready-made registration for -- see the module docstring's license table.
#: `register_sentence_embedding` works for any sentence-transformers model id, not only these; this tuple only
#: names the ones a probe has actually been pointed at so far.
CANDIDATE_MODELS = (
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/embeddinggemma-2",
)

#: `google/embeddinggemma-2`'s own model card ("Best Practices", quick-start) asks for a short task-instruction
#: prefix on text inputs -- "Symmetric Tasks (e.g. classification, similarity): Apply the same task prefix to
#: all inputs being compared", `prompt_name="Classification"` -- rather than the raw, unprefixed text
#: `SentenceTransformer.encode` would otherwise embed. A sentiment tag is exactly a symmetric classification
#: task (every record gets the same prefix, not a query/document split), so this is read, not guessed: a
#: model with no such prompt registered (`BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2`) raises if asked for one
#: that does not exist, so this is a per-model, opt-in mapping, not a blanket default every encoder gets.
_PROMPT_NAME_BY_MODEL: dict[str, str] = {"google/embeddinggemma-2": "Classification"}

_MODEL_LOCK = threading.Lock()
_MODELS: dict[str, SentenceTransformer] = {}


def _load_model(model_name: str) -> SentenceTransformer:
    """Load (and cache, per process, per ``model_name``) a `sentence_transformers.SentenceTransformer` --
    the one place this module actually imports that package, so importing this module with none of it
    installed still succeeds (every name in `CANDIDATE_MODELS` registers fine; only a closure that is
    actually *called* needs the import to have worked).
    """
    with _MODEL_LOCK:
        model = _MODELS.get(model_name)
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "examples.features.sentence_embedding needs the 'sentence-transformers' package "
                    "(pip install sentence-transformers) -- it is not a dependency of prismyra itself, and "
                    "is only imported here, lazily, on the first call to a registered embedding builder"
                ) from e
            model = SentenceTransformer(model_name)
            _MODELS[model_name] = model
        return model


def feature_name_for(model_name: str) -> str:
    """The name a tag's own `prismyra.learn.spec.TagSpec.features` field selects this model with, e.g.
    ``"sentence_embedding:BAAI/bge-small-en-v1.5"`` -- one place this naming convention is spelled out, so
    `register_sentence_embedding` and a caller building a spec entry never retype it independently and drift.
    """
    return f"sentence_embedding:{model_name}"


def register_sentence_embedding(model_name: str, *, prompt_name: str | None = None) -> str:
    """Register a feature builder for ``model_name`` under `feature_name_for(model_name)` and return that
    name. ``prompt_name``, when given, is passed to every `SentenceTransformer.encode` call this builder
    makes (`_PROMPT_NAME_BY_MODEL`'s own per-model default, above, is what the import-time registration loop
    below uses; a caller registering a model of their own choosing picks their own, or leaves it `None` for
    the model's own unprefixed default). Calling this twice for the same ``model_name`` and the same
    ``prompt_name`` is a no-op (`register_features`'s own "the same function object twice is fine" rule,
    since the closure built here is content-stable only in the sense that matters: two calls with identical
    arguments behave identically, even though they are technically two different closures) -- two *different*
    ``prompt_name`` values for the same ``model_name`` refuse the same way the registry already refuses a
    name reused for a genuinely different function, via `register_features` itself raising `FeatureError`.
    """
    name = feature_name_for(model_name)

    def builder(context: str, question: str) -> dict[str, float]:
        # `question` is unused -- same reason tools/snake_features.py's own builder gives: within one tag,
        # the question text is constant.
        model = _load_model(model_name)
        if prompt_name is not None:
            vec = model.encode(context, prompt_name=prompt_name, normalize_embeddings=True)
        else:
            vec = model.encode(context, normalize_embeddings=True)
        return {f"dim_{i}": float(v) for i, v in enumerate(vec)}

    register_features(name, builder)
    return name


# Registering every candidate at import time is the whole point of `--features-module
# examples.features.sentence_embedding`: importing this module is then enough for a tag's own
# `"features": "sentence_embedding:<model>"` to resolve immediately, without a caller also having to call
# `register_sentence_embedding` by hand first. Each registration is cheap (see `register_sentence_embedding`'s
# own docstring); the expensive part -- downloading and loading the actual model -- happens only inside a
# closure, on its first real call, which is this module's whole "optional, lazy" promise.
for _model_name in CANDIDATE_MODELS:
    register_sentence_embedding(_model_name, prompt_name=_PROMPT_NAME_BY_MODEL.get(_model_name))
del _model_name
