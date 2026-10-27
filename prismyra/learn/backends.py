"""`StudentBackend`: the one seam between `prismyra.learn.fit`/`.students` and the actual tree library that
turns `_soft_rows`' rows into a trained model and back into probabilities. Before this module existed, both
callers imported `lightgbm` directly and called its API inline; this is that code moved behind a name, not a
new capability -- LightGBM remains the default, and a tag that never sets `"student"` in its spec entry trains
and predicts exactly as it always did.

A tag opts into a different backend with one field (`prismyra.learn.spec.TagSpec.student`, default
`"lightgbm"`); `python -m prismyra.learn.fit`'s own `--student-backend` overrides it for one run, the same
"per-run override on top of the tag's own spec value" shape `--min-match-rate`/`--min-accuracy-ratio` already
have. Neither library is imported at module import time -- `get_backend` returns a cheap, dependency-free
handle, and only `StudentBackend.train`/`.load` import the library they wrap, lazily, exactly where
`fit.fit_student`/`students.StudentHandle` already did the same for LightGBM alone. A server with no router
configured for any tag still never imports either.

**Soft labels, and the one place the two backends are not identical.** `_soft_rows` (`prismyra.learn.fit`) is
backend-agnostic: it expands each experience into one row per non-zero-probability option, weighted by that
option's probability, the same "宿題を配る" scheme regardless of which booster trains on it next. Both
backends are trained as a weighted multiclass classifier over exactly those rows:

- LightGBM: `objective: "multiclass"`, one row per expanded (features, class, weight) triple, trained with
  `lgb.Dataset(..., weight=w)` -- unchanged from before this module existed.
- XGBoost: `objective: "multi:softprob"` over an `xgb.DMatrix(data, label=y, weight=w)` built from the exact
  same expanded rows. `multi:softprob`'s own training objective is a per-row cross-entropy against one hard
  label, weighted by that row's `weight` -- which is exactly what training on LightGBM's "soft label as
  duplicated, weighted rows" scheme already reduces to: expected cross-entropy over the duplicated rows equals
  cross-entropy against the original soft distribution, scaled by how many options had non-zero probability.
  No row-count rescaling is applied past what `_soft_rows` already does; the two backends are handed identical
  `(x, y, w)` triples.

**Determinism.** Both backends are asked to promise the same thing `fit_student`'s own docstring already
promises for LightGBM: same library build, same input, same `num_threads`, same `seed` -> byte-identical
serialized model text, forever (no training-time randomness is left to the OS scheduler). XGBoost's own
mechanism for this is `tree_method="hist"` on CPU with a fixed `seed` and a fixed `nthread` -- the GPU
histogram path (`tree_method="gpu_hist"`, not used here) is the one XGBoost itself documents as not giving
this guarantee; CPU `"hist"` is deterministic given a fixed thread count, the same build-and-thread-count
caveat LightGBM's own `deterministic=True` already carries, not a new one. One measured difference between
the two, not a bug in either: `XGBoostBackend.train` never sets `subsample`/`colsample_bytree` below their
default of `1.0`, so `seed` has no randomness to feed on CPU `"hist"` and two fits that differ only in
`seed` are byte-identical (`tests/test_learn_backends.py`'s own
`test_xgboost_hist_with_no_subsampling_does_not_depend_on_seed`) -- a *stronger* determinism property than
LightGBM's, whose own histogram tie-breaking does vary with `seed` on some datasets even at
`feature_fraction=bagging_fraction=1.0`.

**`min_data_in_leaf`.** LightGBM's own name for the knob `fit.run`'s `--min-data-in-leaf` sets, chosen there
because a newly-registered tag's first fit (a few dozen rows) cannot form a single split under LightGBM's
default of 20. XGBoost has no field of that exact name; the nearest analogue is `min_child_weight` (the
minimum sum of instance weight, not instance count, needed in a child before a split is allowed) -- close
enough for the same "let a young tag's thin data actually split" purpose, not an exact algebraic match, which
is why `XGBoostBackend.train` passes `min_data_in_leaf` straight through as `min_child_weight` rather than
pretending that mapping is anything more precise than "the knob that exists for the same reason".
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

#: The closed set of backend names a tag's `"student"` field (and this CLI's `--student-backend`) may name.
#: `prismyra.learn.spec._check_entry` imports this directly so a typo'd backend name is a refusal at spec-load
#: time, the same discipline every other closed vocabulary in that module already gets.
STUDENT_BACKENDS = frozenset({"lightgbm", "xgboost"})

#: What a tag's `"student"` field defaults to when the field is absent -- unchanged behaviour for every spec
#: file written before this module existed.
DEFAULT_STUDENT_BACKEND = "lightgbm"


class BackendError(Exception):
    """An unknown backend name was asked for (`get_backend`), or a known one's own library is not installed."""


class LoadedStudentModel(Protocol):
    """One trained model, deserialized and ready to score rows -- what `StudentBackend.load` returns, and the
    only thing `fit.py`/`students.py` call past training. Neither caller holds a `lgb.Booster`/`xgb.Booster`
    directly; both only ever hold one of these.
    """

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        """``vectors``: shape ``(n_rows, n_features)``, ``dtype=float64`` (`Vectorizer.transform`'s own output,
        stacked). Returns ``(n_rows, num_class)`` probabilities, in the same class order `train` was given."""
        ...


class StudentBackend(Protocol):
    """One library's own training and loading, named once (`name`) so a `prismyra.learn.fit.StudentArtifact`
    can record which backend produced it and `students.StudentHandle` can know which one to ask to read it
    back -- see this module's own docstring for why the libraries are never imported outside these two
    methods."""

    name: str

    def train(
        self,
        x_rows: list[list[float]],
        y_rows: list[int],
        w_rows: list[float],
        *,
        num_class: int,
        num_threads: int,
        seed: int,
        num_boost_round: int,
        min_data_in_leaf: int | None,
    ) -> str:
        """Train on `_soft_rows`'s own expanded rows; return the serialized model as one `str` -- what a
        `StudentArtifact.model_text` holds, and what `load` has to be able to read back byte-for-byte."""
        ...

    def load(self, model_text: str) -> LoadedStudentModel:
        """The other half of `train`'s own round trip: deserialize exactly what `train` returned."""
        ...

    def library_version(self) -> str:
        """The installed library's own `__version__` -- recorded on `StudentArtifact.backend_version` purely
        as metadata (never compared against anything; two builds of the same version are not guaranteed
        byte-identical either, which is why the determinism contract above is stated in terms of "same build",
        not "same version string"). Imports the library -- only called once training has already succeeded,
        so the import is already paid for."""
        ...


class LightGBMBackend:
    """`fit.py`'s own training and loading, before this module existed, moved behind `StudentBackend`'s shape
    with no change in behaviour: `objective: "multiclass"`, `deterministic: True`, `force_row_wise: True` --
    see that project's own history (`catboost#1825`) for why `deterministic=True` was the deciding factor
    against CatBoost when LightGBM was first chosen here."""

    name = "lightgbm"

    def train(
        self,
        x_rows: list[list[float]],
        y_rows: list[int],
        w_rows: list[float],
        *,
        num_class: int,
        num_threads: int,
        seed: int,
        num_boost_round: int,
        min_data_in_leaf: int | None,
    ) -> str:
        import lightgbm as lgb

        dataset = lgb.Dataset(np.asarray(x_rows, dtype=np.float64), label=y_rows, weight=w_rows, free_raw_data=True)
        params = {
            "objective": "multiclass",
            "num_class": num_class,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": num_threads,
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "verbosity": -1,
        }
        if min_data_in_leaf is not None:
            params["min_data_in_leaf"] = min_data_in_leaf
        booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
        return booster.model_to_string()

    def load(self, model_text: str) -> LoadedStudentModel:
        import lightgbm as lgb

        booster = lgb.Booster(model_str=model_text)
        return _LightGBMLoaded(booster)

    def library_version(self) -> str:
        import lightgbm as lgb

        return str(lgb.__version__)


class _LightGBMLoaded:
    def __init__(self, booster) -> None:
        self._booster = booster

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(self._booster.predict(vectors), dtype=np.float64)


class XGBoostBackend:
    """`multi:softprob` over `tree_method="hist"` -- see the module docstring for the soft-label equivalence,
    the determinism caveat, and the `min_data_in_leaf` -> `min_child_weight` mapping. `n_jobs`/`nthread` is
    pinned to `num_threads` so a fit run under `prismyra-learn-idle`'s own `--num-threads` bound costs this
    backend exactly what it costs LightGBM, not whatever XGBoost's own default (all visible cores) would
    otherwise take on a shared box.
    """

    name = "xgboost"

    def train(
        self,
        x_rows: list[list[float]],
        y_rows: list[int],
        w_rows: list[float],
        *,
        num_class: int,
        num_threads: int,
        seed: int,
        num_boost_round: int,
        min_data_in_leaf: int | None,
    ) -> str:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(
            np.asarray(x_rows, dtype=np.float64),
            label=np.asarray(y_rows, dtype=np.int32),
            weight=np.asarray(w_rows, dtype=np.float64),
        )
        params: dict = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "tree_method": "hist",
            "seed": seed,
            "nthread": num_threads,
            "verbosity": 0,
        }
        if min_data_in_leaf is not None:
            params["min_child_weight"] = min_data_in_leaf
        booster = xgb.train(params, dmatrix, num_boost_round=num_boost_round)
        # "json", not the default binary/UBJ format: a `str`, the same shape `StudentArtifact.model_text`
        # already holds for LightGBM, so the artifact's on-disk JSON file never has to carry base64 or bytes.
        raw = booster.save_raw(raw_format="json")
        return bytes(raw).decode("utf-8")

    def load(self, model_text: str) -> LoadedStudentModel:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(bytearray(model_text.encode("utf-8")))
        return _XGBoostLoaded(booster, xgb)

    def library_version(self) -> str:
        import xgboost as xgb

        return str(xgb.__version__)


class _XGBoostLoaded:
    def __init__(self, booster, xgb_module) -> None:
        self._booster = booster
        self._xgb = xgb_module

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        dmatrix = self._xgb.DMatrix(np.asarray(vectors, dtype=np.float64))
        return np.asarray(self._booster.predict(dmatrix), dtype=np.float64)


_BACKENDS: dict[str, StudentBackend] = {
    "lightgbm": LightGBMBackend(),
    "xgboost": XGBoostBackend(),
}


def get_backend(name: str) -> StudentBackend:
    """``name`` not in `STUDENT_BACKENDS` is a programming error reaching this far (`spec.py` already refuses
    an unregistered name at load time; `fit.py`'s own `--student-backend` is restricted to the same set by its
    `argparse` `choices`) -- this still raises loudly rather than `KeyError`-ing, for a caller that built a
    `TagSpec`/`StudentArtifact` by hand (a test, say) without going through either validated path.
    """
    try:
        return _BACKENDS[name]
    except KeyError:
        raise BackendError(f"no student backend named {name!r} (known: {sorted(STUDENT_BACKENDS)})") from None
