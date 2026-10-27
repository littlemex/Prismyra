"""`prismyra.learn.backends`: the `StudentBackend` seam, independent of `fit.py`/`students.py`'s own use of it.

Covers the registry (`get_backend`, `STUDENT_BACKENDS`), each backend's own train/load/predict round trip, and
-- the one claim neither `test_learn_fit.py` nor `test_learn_students.py` can make from inside the same
process -- that a model trained once keeps predicting the same thing after an actual process restart, not just
after a second in-process object construction.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from prismyra.learn.backends import STUDENT_BACKENDS, BackendError, get_backend

X_ROWS = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]] * 5
Y_ROWS = [0, 1, 0, 1] * 5
W_ROWS = [1.0] * 20


def test_student_backends_is_exactly_lightgbm_and_xgboost():
    assert frozenset({"lightgbm", "xgboost"}) == STUDENT_BACKENDS


def test_get_backend_refuses_an_unknown_name():
    with pytest.raises(BackendError, match="no student backend named"):
        get_backend("catboost")


@pytest.mark.parametrize("name", sorted(STUDENT_BACKENDS))
def test_get_backend_returns_a_backend_whose_name_matches(name):
    pytest.importorskip(name)
    backend = get_backend(name)
    assert backend.name == name


@pytest.mark.parametrize("name", sorted(STUDENT_BACKENDS))
def test_train_then_load_predicts_probabilities_that_sum_to_one(name):
    pytest.importorskip(name)
    backend = get_backend(name)
    model_text = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=0, num_boost_round=10, min_data_in_leaf=1
    )
    model = backend.load(model_text)
    probs = model.predict_proba(np.asarray(X_ROWS, dtype=np.float64))
    assert probs.shape == (len(X_ROWS), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", sorted(STUDENT_BACKENDS))
def test_training_the_same_rows_twice_gives_byte_identical_model_text(name):
    pytest.importorskip(name)
    backend = get_backend(name)
    text_a = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=0, num_boost_round=10, min_data_in_leaf=1
    )
    text_b = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=0, num_boost_round=10, min_data_in_leaf=1
    )
    assert text_a == text_b


def test_a_different_seed_can_change_the_lightgbm_model_text():
    pytest.importorskip("lightgbm")
    backend = get_backend("lightgbm")
    text_a = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=0, num_boost_round=10, min_data_in_leaf=1
    )
    text_b = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=1, num_boost_round=10, min_data_in_leaf=1
    )
    assert text_a != text_b


def test_xgboost_hist_with_no_subsampling_does_not_depend_on_seed():
    """Not a bug: `XGBoostBackend` never sets `subsample`/`colsample_bytree` below their default of `1.0`, so
    CPU `tree_method="hist"` has no random source `seed` could feed on this (or any) dataset -- every split is
    the single best split over the full histogram, every round, regardless of `seed`. `test_learn_fit.py`'s
    own `test_a_different_seed_can_change_the_model` makes the "a different seed *can* change the model"
    claim for LightGBM specifically, on a dataset where it does; this test instead pins down that XGBoost's
    *lack* of seed-dependence here is the expected, measured behaviour, not something a future change should
    "fix" by rediscovering it as a regression.
    """
    pytest.importorskip("xgboost")
    backend = get_backend("xgboost")
    text_a = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=0, num_boost_round=10, min_data_in_leaf=1
    )
    text_b = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=1, num_boost_round=10, min_data_in_leaf=1
    )
    assert text_a == text_b


@pytest.mark.parametrize("name", sorted(STUDENT_BACKENDS))
def test_library_version_is_a_nonempty_string(name):
    pytest.importorskip(name)
    backend = get_backend(name)
    version = backend.library_version()
    assert isinstance(version, str) and version


# --------------------------------------------------------------------------- real process restart, not just a
# second in-process object: a fresh `python -I` subprocess loads the exact `model_text` `train()` returned and
# predicts on a fixed vector; run twice, independently, and the two subprocesses' own stdout must match
# byte-for-byte. `-I` (isolated mode) is used for the same reason the harness's own sandboxing note asks for it
# when running an interpreter over anything written to a temp file: a planted module on `sys.path`/cwd must
# never get a vote in what this subprocess imports.

_PREDICT_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from prismyra.learn.backends import get_backend

    payload = json.loads(sys.stdin.read())
    backend = get_backend(payload["backend"])
    model = backend.load(payload["model_text"])
    probs = model.predict_proba(np.asarray(payload["vectors"], dtype=np.float64))
    print(json.dumps(probs.tolist()))
    """
)


@pytest.mark.parametrize("name", sorted(STUDENT_BACKENDS))
def test_prediction_survives_a_real_process_restart(name, tmp_path):
    pytest.importorskip(name)
    backend = get_backend(name)
    model_text = backend.train(
        X_ROWS, Y_ROWS, W_ROWS, num_class=2, num_threads=1, seed=0, num_boost_round=10, min_data_in_leaf=1
    )
    payload = json.dumps({"backend": name, "model_text": model_text, "vectors": X_ROWS})

    script_path = tmp_path / "predict_once.py"
    script_path.write_text(_PREDICT_SCRIPT)

    def run_once() -> str:
        result = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return result.stdout.strip()

    out_a = run_once()
    out_b = run_once()
    assert out_a == out_b
    probs = np.asarray(json.loads(out_a))
    assert probs.shape == (len(X_ROWS), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
