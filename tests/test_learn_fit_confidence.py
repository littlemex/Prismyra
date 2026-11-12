"""PREREG-R2 idea 1 ("admission for what is served"): `prismyra.learn.fit`'s `"confidence"` admission mode.

Two layers, kept separate the same way `tests/test_learn_fit.py` already separates pure-Python pieces
(`argmax`, `wilson_lower_bound`, `_soft_rows`) from the LightGBM-backed `run()` round trip:

* `score_eval_confidence`/`held_out_confidence_agreement` are pure functions of a list of items and a
  ``predict_fn`` -- tested here with a hand-written stub ``predict_fn``, no booster involved, so these cases
  need no ``importorskip("lightgbm")`` at all.
* `run()`'s own "confidence" branch (admission_mode dispatch, `admitted_for`, the "answer rate 0" refusal) is
  tested the same way `test_run_admits_a_student_that_beats_prismyra_and_agrees_on_held_out` already does --
  a real, tiny LightGBM fit -- because that branch's whole point is which *real* student numbers gate
  admission, not a property provable from the stub alone.
"""

from __future__ import annotations

import json

import pytest

from prismyra.learn.experience import Experience
from prismyra.learn.fit import EvalItem, held_out_confidence_agreement, run, score_eval_confidence

# --------------------------------------------------------------------------- pure-Python: score_eval_confidence


def _eval_item(context, gold, prismyra_probs):
    return EvalItem(context=context, gold=gold, options=("negative", "positive"), prismyra_probabilities=prismyra_probs)


def _stub_predict(answers: dict[str, dict[str, float]]):
    """A ``predict_fn`` that returns a fixed, pre-decided probability dict per ``context`` -- the same
    "answers are hand-picked, not actually modelled" convention `tests/test_learn_router.py`'s own fake
    students use, kept here because these two functions take a ``predict_fn`` as a parameter precisely so a
    caller never has to run a real booster to exercise them.
    """

    def predict_fn(context: str) -> dict[str, float]:
        return answers[context]

    return predict_fn


def test_score_eval_confidence_counts_only_items_with_a_prismyra_prediction():
    items = [
        _eval_item("a", "positive", {"negative": 0.1, "positive": 0.9}),
        _eval_item("b", "positive", None),  # no prismyra_probabilities: excluded from the denominator entirely
    ]
    predict_fn = _stub_predict({"a": {"negative": 0.1, "positive": 0.9}, "b": {"negative": 0.5, "positive": 0.5}})
    out = score_eval_confidence(items, predict_fn, tau=0.8)
    assert out["n"] == 1
    assert out["answered_by_student"] == 1
    assert out["answer_rate"] == 1.0
    assert out["hybrid_accuracy"] == 1.0
    assert out["prismyra_accuracy"] == 1.0


def test_score_eval_confidence_falls_back_to_prismyra_below_tau():
    # The student's own top probability (0.6) is below tau (0.8): the hybrid answer must be Prismyra's own
    # prediction ("negative"), not the student's ("positive") -- and since "negative" is wrong here, the
    # hybrid is wrong too, even though the student itself would have been right.
    items = [_eval_item("a", "positive", {"negative": 0.7, "positive": 0.3})]
    predict_fn = _stub_predict({"a": {"negative": 0.4, "positive": 0.6}})
    out = score_eval_confidence(items, predict_fn, tau=0.8)
    assert out["answered_by_student"] == 0
    assert out["answer_rate"] == 0.0
    assert out["hybrid_accuracy"] == 0.0  # Prismyra's own "negative" answer, scored against gold "positive"
    assert out["prismyra_accuracy"] == 0.0


def test_score_eval_confidence_answers_above_tau_with_the_student():
    items = [_eval_item("a", "positive", {"negative": 0.9, "positive": 0.1})]  # Prismyra itself is wrong here
    predict_fn = _stub_predict({"a": {"negative": 0.05, "positive": 0.95}})  # student is confident and right
    out = score_eval_confidence(items, predict_fn, tau=0.8)
    assert out["answered_by_student"] == 1
    assert out["hybrid_accuracy"] == 1.0  # the student's own answer, not Prismyra's wrong one, was used
    assert out["prismyra_accuracy"] == 0.0


def test_score_eval_confidence_on_an_empty_list_has_n_zero_not_a_division_error():
    out = score_eval_confidence([], _stub_predict({}), tau=0.8)
    assert out == {
        "n": 0,
        "tau": 0.8,
        "answered_by_student": 0,
        "answer_rate": 0.0,
        "hybrid_accuracy": 0.0,
        "prismyra_accuracy": 0.0,
    }


# --------------------------------------------------------------------------- pure-Python: held_out_confidence_agreement


def _exp(context: str, p_positive: float) -> Experience:
    return Experience(
        tag="t",
        spec_version="v",
        ts=0.0,
        context=context,
        question="q",
        options=("negative", "positive"),
        kind="noul",
        probabilities={"negative": 1 - p_positive, "positive": p_positive},
        answered_by="prismyra",
        versions={},
    )


def test_held_out_confidence_agreement_only_scores_the_answered_subset():
    held_out = [_exp("a", 0.9), _exp("b", 0.9)]  # teacher: both "positive"
    # The student answers "a" confidently and agrees; "b" it is unsure about (0.6 < tau) and is never counted.
    predict_fn = _stub_predict(
        {"a": {"negative": 0.02, "positive": 0.98}, "b": {"negative": 0.4, "positive": 0.6}}
    )
    out = held_out_confidence_agreement(held_out, predict_fn, tau=0.8)
    assert out["held_out_n"] == 2
    assert out["answered"] == 1
    assert out["answer_rate"] == 0.5
    assert out["agree"] == 1
    assert out["rate"] == 1.0
    # Wilson lower bound of 1/1 is well below 1.0 (small-n penalty) but still > 0 -- the exact value is
    # `wilson_lower_bound`'s own job to pin; this just checks the subset size feeding it is 1, not 2.
    assert 0.0 < out["lower_bound_95"] < 1.0


def test_held_out_confidence_agreement_answered_zero_when_nothing_clears_tau():
    held_out = [_exp("a", 0.9)]
    predict_fn = _stub_predict({"a": {"negative": 0.4, "positive": 0.6}})  # below tau=0.8
    out = held_out_confidence_agreement(held_out, predict_fn, tau=0.8)
    assert out["answered"] == 0
    assert out["lower_bound_95"] == 0.0  # wilson_lower_bound(0, 0) -- the caller is what must not read this
    # as "a genuine lower bound of zero clearing nothing"; see test_run_rejects_confidence_admission_when_the_
    # answer_rate_is_zero below for that distinction actually being made.


def test_held_out_confidence_agreement_on_an_empty_held_out_set():
    out = held_out_confidence_agreement([], _stub_predict({}), tau=0.8)
    assert out["held_out_n"] == 0
    assert out["answered"] == 0
    assert out["answer_rate"] == 0.0


# --------------------------------------------------------------------------- run(): the "confidence" admission
# branch, LightGBM-backed (same convention as test_learn_fit.py's own run() tests).

lightgbm = pytest.importorskip("lightgbm")

TASK = "sentiment-v1"
BASE_ENTRY = {
    "task": TASK,
    "question": "Is this tweet positive?",
    "options": ["negative", "positive"],
    "kind": "noul",
    "normalize": "strip+lower",
    "retain_days": 30,
    "keep_hidden": False,
    "eval_set": "eval.jsonl",
}

POSITIVE_WORDS = ["love", "great", "amazing", "wonderful", "fantastic", "awesome", "happy", "best"]
NEGATIVE_WORDS = ["hate", "terrible", "awful", "worst", "bad", "sad", "horrible", "angry"]


def synthetic_rows(n_per_class=40):
    rows = []
    idx = 0
    for i in range(n_per_class):
        rows.append((f"this is {POSITIVE_WORDS[i % len(POSITIVE_WORDS)]} news item {i}", 0.97, idx))
        idx += 1
        rows.append((f"this is {NEGATIVE_WORDS[i % len(NEGATIVE_WORDS)]} news item {i}", 0.03, idx))
        idx += 1
    return rows


def make_spec(tmp_path, **overrides):
    from prismyra.learn.spec import load_spec

    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps([{**BASE_ENTRY, **overrides}]))
    return spec_path, load_spec(spec_path)


def write_experience(tmp_path, spec, rows):
    tag_dir = tmp_path / "experience" / TASK
    tag_dir.mkdir(parents=True)
    with (tag_dir / "2026-01-01.jsonl").open("w") as fh:
        for context, p_positive, idx in rows:
            exp = Experience(
                tag=TASK,
                spec_version=spec.version,
                ts=1_700_000_000.0 + idx,
                context=context,
                question=BASE_ENTRY["question"],
                options=("negative", "positive"),
                kind="noul",
                probabilities={"negative": 1 - p_positive, "positive": p_positive},
                answered_by="prismyra",
                versions={"package": "0", "backbone": "stub", "spec": spec.version},
            )
            fh.write(json.dumps(exp.to_json()) + "\n")


def write_eval_set(path, rows):
    with path.open("w") as fh:
        for context, p_positive, _ in rows:
            gold = "positive" if p_positive > 0.5 else "negative"
            fh.write(
                json.dumps(
                    {
                        "context": context,
                        "gold": gold,
                        "options": ["negative", "positive"],
                        "prismyra_probabilities": {"negative": 1 - p_positive, "positive": p_positive},
                    }
                )
                + "\n"
            )


def _run(spec_path, tmp_path, **kwargs):
    defaults = {
        "spec_path": spec_path,
        "task": TASK,
        "log_dir": None,
        "package_version": "0.0.0-test",
        "backbone": "stub/model",
        "dims": 128,
        "held_out_fraction": 0.25,
        "num_threads": 1,
        "seed": 0,
        "num_boost_round": 30,
        "min_data_in_leaf": 2,
        "min_match_rate": 0.5,
        "min_accuracy_ratio": 0.5,
    }
    defaults.update(kwargs)
    return run(**defaults)


def test_run_admits_under_confidence_mode_and_records_admitted_for(tmp_path):
    # An easy, agreeing-with-gold dataset -- the same fixture shape test_learn_fit.py's own full-coverage
    # admission test uses -- is also easy under confidence admission: every item clears a lenient tau, and
    # the hybrid answer is just the student's own (correct) answer everywhere.
    spec_path, spec = make_spec(tmp_path, admission={"mode": "confidence", "tau": 0.3})
    write_experience(tmp_path, spec, synthetic_rows(60))
    write_eval_set(tmp_path / "eval.jsonl", synthetic_rows(15))

    artifact = _run(spec_path, tmp_path)
    assert artifact.admitted, artifact.reasons
    assert artifact.admitted_for == {"mode": "confidence", "tau": 0.3}
    assert artifact.metrics["admission_mode"] == "confidence"
    assert "confidence_admission" in artifact.metrics
    # The full-coverage numbers are still reported, informationally, even though they did not gate this run.
    assert "eval" in artifact.metrics
    assert "held_out_agreement" in artifact.metrics


def test_run_rejects_confidence_admission_when_the_answer_rate_is_zero(tmp_path):
    # tau=0.999 is above any probability this tiny booster will ever actually emit on this toy fixture: the
    # held-out answered subset is empty, which must read as "not admitted", not as a lower bound of 0.0
    # clearing nothing (PREREG-R2: "answer rate が 0 なら admit しない").
    spec_path, spec = make_spec(tmp_path, admission={"mode": "confidence", "tau": 0.999})
    write_experience(tmp_path, spec, synthetic_rows(60))
    write_eval_set(tmp_path / "eval.jsonl", synthetic_rows(15))

    artifact = _run(spec_path, tmp_path)
    assert not artifact.admitted
    assert artifact.admitted_for is None
    held_out = artifact.metrics["confidence_admission"]["held_out"]
    assert held_out["answered"] == 0
    assert any("answer rate is 0" in r for r in artifact.reasons)


def test_run_in_full_mode_never_sets_admitted_for(tmp_path):
    # No "admission" field on the tag at all -- the original, unconditional bar. admitted_for must stay None
    # even when the run clears admission, the same "every artifact written before this field existed reads
    # as full admission" contract StudentArtifact.from_json keeps for a key entirely missing from JSON.
    spec_path, spec = make_spec(tmp_path)
    write_experience(tmp_path, spec, synthetic_rows(60))
    write_eval_set(tmp_path / "eval.jsonl", synthetic_rows(15))

    artifact = _run(spec_path, tmp_path)
    assert artifact.admitted, artifact.reasons
    assert artifact.admitted_for is None
    assert artifact.metrics["admission_mode"] == "full"


def test_the_confidence_admitted_artifact_round_trips_through_json(tmp_path):
    spec_path, spec = make_spec(tmp_path, admission={"mode": "confidence", "tau": 0.3})
    write_experience(tmp_path, spec, synthetic_rows(60))
    write_eval_set(tmp_path / "eval.jsonl", synthetic_rows(15))

    artifact = _run(spec_path, tmp_path)
    assert artifact.admitted

    from prismyra.learn.fit import StudentArtifact

    reloaded = StudentArtifact.from_json(json.loads(json.dumps(artifact.to_json())))
    assert reloaded.admitted_for == {"mode": "confidence", "tau": 0.3}


def test_an_artifact_serialized_before_admitted_for_existed_reads_as_full_admission(tmp_path):
    spec_path, spec = make_spec(tmp_path)
    write_experience(tmp_path, spec, synthetic_rows(60))
    write_eval_set(tmp_path / "eval.jsonl", synthetic_rows(15))
    artifact = _run(spec_path, tmp_path)

    from prismyra.learn.fit import StudentArtifact

    raw = artifact.to_json()
    del raw["admitted_for"]  # simulate a pre-existing artifact on disk, written before this field existed
    reloaded = StudentArtifact.from_json(raw)
    assert reloaded.admitted_for is None


def test_an_external_tag_is_judged_and_reported_on_the_full_bar(tmp_path):
    # Admission mode "external": Prismyra's own verdict no longer decides serving, but it is still computed on the
    # original full-coverage bar and reported, and the artifact says the serving decision was handed out.
    spec_path, spec = make_spec(tmp_path, admission={"mode": "external", "record": "admission.json"})
    write_experience(tmp_path, spec, synthetic_rows(60))
    write_eval_set(tmp_path / "eval.jsonl", synthetic_rows(15))

    artifact = _run(spec_path, tmp_path)
    assert artifact.metrics["admission_mode"] == "full"
    assert artifact.metrics["serving_admission"] == "external"
    assert artifact.admitted_for is None
