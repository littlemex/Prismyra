"""`prismyra.learn.fit`: dedup, the Wilson bound, and -- the one GPU-free claim stage 2 rests its determinism
story on -- that training the same data twice gives byte-identical model text."""

from __future__ import annotations

import json

import pytest

from prismyra.learn.experience import Experience
from prismyra.learn.fit import (
    FitError,
    argmax,
    dedup,
    run,
    split_train_held_out,
    tau_sweep,
    wilson_lower_bound,
)
from prismyra.learn.spec import load_spec

lightgbm = pytest.importorskip("lightgbm")

TASK = "sentiment-v1"
ENTRY = {
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


def _exp(spec, context, p_positive, idx=0):
    return Experience(
        tag=TASK,
        spec_version=spec.version,
        ts=1_700_000_000.0 + idx,
        context=context,
        question=ENTRY["question"],
        options=("negative", "positive"),
        kind="noul",
        probabilities={"negative": 1 - p_positive, "positive": p_positive},
        answered_by="prismyra",
        versions={"package": "0", "backbone": "stub", "spec": spec.version},
    )


def make_spec(tmp_path, **overrides):
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps([{**ENTRY, **overrides}]))
    return spec_path, load_spec(spec_path)


def write_experience(tmp_path, spec, rows):
    tag_dir = tmp_path / "experience" / TASK
    tag_dir.mkdir(parents=True)
    with (tag_dir / "2026-01-01.jsonl").open("w") as fh:
        for context, p_positive, idx in rows:
            fh.write(json.dumps(_exp(spec, context, p_positive, idx).to_json()) + "\n")


def synthetic_rows(n_per_class=40):
    rows = []
    idx = 0
    for i in range(n_per_class):
        rows.append((f"this is {POSITIVE_WORDS[i % len(POSITIVE_WORDS)]} news item {i}", 0.95, idx))
        idx += 1
        rows.append((f"this is {NEGATIVE_WORDS[i % len(NEGATIVE_WORDS)]} news item {i}", 0.05, idx))
        idx += 1
    return rows


# --------------------------------------------------------------------------- small pure-python pieces


def test_argmax():
    assert argmax({"a": 0.1, "b": 0.9}) == "b"


def _simple_exp(context: str, ts: float, probabilities=None) -> Experience:
    return Experience(
        tag="t",
        spec_version="v",
        ts=ts,
        context=context,
        question="q",
        options=("a", "b"),
        kind="boolean",
        probabilities=probabilities or {"a": 0.1, "b": 0.9},
        answered_by="prismyra",
        versions={},
    )


def test_dedup_drops_repeated_contexts_keeping_the_first():
    a = _simple_exp("same text", ts=1.0)
    b = _simple_exp("same text", ts=2.0, probabilities={"a": 0.9, "b": 0.1})
    c = _simple_exp("different text", ts=3.0)
    out = dedup([a, b, c])
    assert len(out) == 2
    assert out[0].ts == 1.0  # first occurrence kept
    assert out[1].context == "different text"


def test_split_train_held_out_is_deterministic_and_covers_everything():
    items = list(range(100))
    train1, held1 = split_train_held_out(items, 0.3, seed=7)
    train2, held2 = split_train_held_out(items, 0.3, seed=7)
    assert train1 == train2
    assert held1 == held2
    assert sorted(train1 + held1) == items
    assert len(held1) == 30


def test_split_train_held_out_with_a_different_seed_differs():
    items = list(range(100))
    _, held_a = split_train_held_out(items, 0.3, seed=1)
    _, held_b = split_train_held_out(items, 0.3, seed=2)
    assert held_a != held_b


def test_wilson_lower_bound_is_below_the_point_estimate():
    assert wilson_lower_bound(97, 100) < 0.97
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_tightens_with_more_data():
    small = wilson_lower_bound(9, 10)
    large = wilson_lower_bound(900, 1000)
    assert large > small  # same point estimate (0.9), more data narrows the interval upward


# --------------------------------------------------------------------------- fit_student determinism


def test_training_the_same_data_twice_gives_byte_identical_model_text(tmp_path):
    _, spec = make_spec(tmp_path)
    tag = spec.by_task(TASK)
    train = [_exp(spec, ctx, p, i) for i, (ctx, p, _) in enumerate(synthetic_rows(20))]

    from prismyra.learn.fit import fit_student

    booster_a, vec_a = fit_student(tag, train=train, dims=64, num_threads=1, seed=0, num_boost_round=10)
    booster_b, vec_b = fit_student(tag, train=train, dims=64, num_threads=1, seed=0, num_boost_round=10)
    assert booster_a.model_to_string() == booster_b.model_to_string()
    assert vec_a == vec_b


def test_a_different_seed_can_change_the_model(tmp_path):
    _, spec = make_spec(tmp_path)
    tag = spec.by_task(TASK)
    train = [_exp(spec, ctx, p, i) for i, (ctx, p, _) in enumerate(synthetic_rows(20))]

    from prismyra.learn.fit import fit_student

    booster_a, _ = fit_student(tag, train=train, dims=64, num_threads=1, seed=0, num_boost_round=10)
    booster_b, _ = fit_student(tag, train=train, dims=64, num_threads=1, seed=1, num_boost_round=10)
    # Not asserted to always differ (a tiny, easy dataset could coincide), but this dataset's seed-0 and
    # seed-1 trees are known to differ -- regression protection against `seed` silently not being wired in.
    assert booster_a.model_to_string() != booster_b.model_to_string()


def test_fit_student_refuses_an_empty_training_set(tmp_path):
    _, spec = make_spec(tmp_path)
    tag = spec.by_task(TASK)
    from prismyra.learn.fit import fit_student

    with pytest.raises(FitError, match="no trainable experience"):
        fit_student(tag, train=[], dims=16, num_threads=1, seed=0, num_boost_round=5)


# --------------------------------------------------------------------------- tau_sweep shape


def test_tau_sweep_reports_a_monotonic_answer_rate(tmp_path):
    _, spec = make_spec(tmp_path)
    held_out = [_exp(spec, ctx, p, i) for i, (ctx, p, _) in enumerate(synthetic_rows(10))]

    def predict_fn(context):
        return {"negative": 0.3, "positive": 0.7}

    rows = tau_sweep(held_out, predict_fn, taus=[0.0, 0.5, 0.8])
    rates = [r["answer_rate"] for r in rows]
    assert rates == sorted(rates, reverse=True)  # a higher bar answers no more than a lower one


# --------------------------------------------------------------------------- run(): end-to-end admission


def test_run_admits_a_student_that_beats_prismyra_and_agrees_on_held_out(tmp_path):
    spec_path, spec = make_spec(tmp_path, retain_days=30)
    rows = synthetic_rows(60)
    write_experience(tmp_path, spec, rows)

    eval_path = tmp_path / "eval.jsonl"
    with eval_path.open("w") as fh:
        for context, p_positive, _ in synthetic_rows(15):
            gold = "positive" if p_positive > 0.5 else "negative"
            # Prismyra's own recorded answer agrees with gold here, same as the training signal: the point of
            # this fixture is an easy dataset the student can actually win on, not an adversarial one.
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

    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0.0.0-test",
        backbone="stub/model",
        dims=128,
        held_out_fraction=0.25,
        min_match_rate=0.5,
        accuracy_margin=0.5,  # generous: an easy fixture, not a claim about a real tag's bar
        num_threads=1,
        seed=0,
        num_boost_round=30,
        min_data_in_leaf=2,  # a toy, few-dozen-row fixture: LightGBM's own default (20) would never split
    )
    assert artifact.admitted, artifact.reasons
    assert artifact.tag == TASK
    assert artifact.spec_version == spec.version
    assert artifact.package_version == "0.0.0-test"
    assert artifact.backbone == "stub/model"
    assert "held_out_agreement" in artifact.metrics
    assert "tau_sweep" in artifact.metrics


def test_run_refuses_admission_without_an_eval_set(tmp_path):
    spec_path, spec = make_spec(tmp_path, eval_set="does-not-exist.jsonl")
    write_experience(tmp_path, spec, synthetic_rows(60))
    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0",
        backbone="m",
        dims=64,
        min_match_rate=0.0,
        num_threads=1,
        seed=0,
        num_boost_round=10,
    )
    assert not artifact.admitted
    assert any("does not exist" in r for r in artifact.reasons)


def test_run_refuses_a_nonexistent_task(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    with pytest.raises(FitError, match="no entry for task"):
        run(
            spec_path=spec_path,
            task="not-a-task",
            log_dir=None,
            package_version="0",
            backbone="m",
        )


def test_the_student_artifact_round_trips_through_json_and_predicts(tmp_path):
    spec_path, spec = make_spec(tmp_path, eval_set="does-not-exist.jsonl")
    write_experience(tmp_path, spec, synthetic_rows(40))
    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0",
        backbone="m",
        dims=64,
        num_threads=1,
        seed=0,
        num_boost_round=10,
    )
    from prismyra.learn.fit import StudentArtifact

    restored = StudentArtifact.from_json(json.loads(json.dumps(artifact.to_json())))
    assert restored == artifact
    probs = restored.predict_proba("this is amazing news item 0")
    assert set(probs) == {"negative", "positive"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
