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
        min_accuracy_ratio=0.5,  # generous: an easy fixture, not a claim about a real tag's bar
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


def _write_eval_set(path, rows):
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


# --------------------------------------------------------------------------- 2026-10-11: the admission bar
# (min_match_rate/min_accuracy_ratio) moved from this script's own CLI defaults into the tag's spec entry,
# with spec.py's DEFAULT_MIN_MATCH_RATE/DEFAULT_MIN_ACCURACY_RATIO (0.90/0.90) as the single fallback. These
# tests are about *which number run() actually used*, not about distillation accuracy -- the fixture is the
# same easy, agreeing-with-gold dataset the test above uses for exactly that reason.


def test_run_uses_the_tags_own_admission_overrides_when_the_caller_passes_none(tmp_path):
    # The tag's own entry sets a bar neither spec.py's defaults nor a hand-picked override would clear by
    # accident (min_accuracy_ratio=0.5 is far more lenient than the 0.90 default; min_match_rate=0.5 likewise)
    # -- if run() silently fell back to spec.py's 0.90/0.90 instead of reading the tag, this fixture (an easy,
    # 60-row toy) would not reliably admit at the stricter bar, and the test would start failing.
    spec_path, spec = make_spec(tmp_path, retain_days=30, min_match_rate=0.5, min_accuracy_ratio=0.5)
    tag = spec.by_task(TASK)
    assert tag.min_match_rate == 0.5
    assert tag.min_accuracy_ratio == 0.5
    write_experience(tmp_path, spec, synthetic_rows(60))
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_set(eval_path, synthetic_rows(15))

    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0.0.0-test",
        backbone="stub/model",
        dims=128,
        held_out_fraction=0.25,
        # min_match_rate / min_accuracy_ratio deliberately omitted: run()'s own default (None) must mean
        # "use the tag's own spec value", not spec.py's module-level default.
        num_threads=1,
        seed=0,
        num_boost_round=30,
        min_data_in_leaf=2,
    )
    assert artifact.admitted, artifact.reasons
    assert artifact.metrics["admission_thresholds"] == {"min_match_rate": 0.5, "min_accuracy_ratio": 0.5}


def test_run_falls_back_to_the_module_default_when_the_tag_has_no_override(tmp_path):
    spec_path, spec = make_spec(tmp_path, retain_days=30)
    tag = spec.by_task(TASK)
    assert tag.min_match_rate == 0.90
    assert tag.min_accuracy_ratio == 0.90
    write_experience(tmp_path, spec, synthetic_rows(60))
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_set(eval_path, synthetic_rows(15))

    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0",
        backbone="stub/model",
        dims=128,
        held_out_fraction=0.25,
        num_threads=1,
        seed=0,
        num_boost_round=30,
        min_data_in_leaf=2,
    )
    assert artifact.metrics["admission_thresholds"] == {"min_match_rate": 0.90, "min_accuracy_ratio": 0.90}


def test_run_rejects_when_the_tags_own_bar_is_not_met(tmp_path):
    # A bar no 15-row toy eval set can be expected to clear exactly (every item would have to be correct),
    # set on the tag itself rather than passed to run() -- this is "the spec's own number governs", not "an
    # explicit kwarg governs" (that path is covered by test_run_admits_... above, which always passes one).
    spec_path, spec = make_spec(tmp_path, retain_days=30, min_match_rate=0.999, min_accuracy_ratio=1.0)
    write_experience(tmp_path, spec, synthetic_rows(60))
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_set(eval_path, synthetic_rows(15))

    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0",
        backbone="stub/model",
        dims=128,
        held_out_fraction=0.25,
        num_threads=1,
        seed=0,
        num_boost_round=30,
        min_data_in_leaf=2,
    )
    assert not artifact.admitted
    assert any("min_match_rate" in r for r in artifact.reasons)


def test_run_kwarg_overrides_the_tags_own_min_match_rate(tmp_path):
    # The tag asks for a strict bar; the caller explicitly overrides it for this one run. The override, not
    # the tag's own entry, must be what run() actually judged against.
    spec_path, spec = make_spec(tmp_path, retain_days=30, min_match_rate=0.999)
    write_experience(tmp_path, spec, synthetic_rows(60))
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_set(eval_path, synthetic_rows(15))

    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0",
        backbone="stub/model",
        dims=128,
        held_out_fraction=0.25,
        min_match_rate=0.0,  # explicit override beats the tag's own 0.999
        num_threads=1,
        seed=0,
        num_boost_round=30,
        min_data_in_leaf=2,
    )
    assert artifact.metrics["admission_thresholds"]["min_match_rate"] == 0.0
    # min_match_rate=0.0 means condition 2 clears unconditionally; the tag's own 0.999 would not have.
    assert (
        artifact.metrics["held_out_agreement"]["lower_bound_95"]
        >= artifact.metrics["admission_thresholds"]["min_match_rate"]
    )


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


# --------------------------------------------------------------------------- a tag that names a registered,
# structured feature builder instead of the hashed-TF-IDF default (learn3)


def _word_count_v1(context: str, question: str) -> dict[str, float]:
    words = context.split()
    return {
        "n_words": float(len(words)),
        "n_positive_words": float(sum(1 for w in words if w.strip(".,!") in POSITIVE_WORDS)),
        "n_negative_words": float(sum(1 for w in words if w.strip(".,!") in NEGATIVE_WORDS)),
    }


def _register_word_count_features():
    # A module-level function, not one freshly defined per call: `register_features` is a no-op for the
    # *same* function object registered twice (every test in this file that needs this tag calls this), but
    # would (rightly) refuse a second, different function trying to claim the same name.
    from prismyra.learn.features import register_features

    register_features("word-count-v1", _word_count_v1)
    return _word_count_v1


def test_fit_student_honours_a_tags_registered_features_field(tmp_path):
    _register_word_count_features()
    _spec_path, spec = make_spec(tmp_path, features="word-count-v1")
    tag = spec.by_task(TASK)
    assert tag.features == "word-count-v1"
    train = [_exp(spec, ctx, p, i) for i, (ctx, p, _) in enumerate(synthetic_rows(20))]

    from prismyra.learn.fit import fit_student

    _booster, vectorizer = fit_student(tag, train=train, dims=64, num_threads=1, seed=0, num_boost_round=10)
    assert vectorizer.features_name == "word-count-v1"
    assert vectorizer.hashing is None
    assert vectorizer.names == ("n_negative_words", "n_positive_words", "n_words")


def test_run_admits_a_structured_feature_student_and_round_trips_the_artifact(tmp_path):
    _register_word_count_features()
    spec_path, spec = make_spec(tmp_path, features="word-count-v1")
    rows = synthetic_rows(60)
    write_experience(tmp_path, spec, rows)

    eval_path = tmp_path / "eval.jsonl"
    with eval_path.open("w") as fh:
        for context, p_positive, _ in synthetic_rows(15):
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

    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0.0.0-test",
        backbone="stub/model",
        dims=128,  # unused by the structured branch, kept only as metadata
        held_out_fraction=0.25,
        min_match_rate=0.5,
        min_accuracy_ratio=0.5,
        num_threads=1,
        seed=0,
        num_boost_round=30,
        min_data_in_leaf=2,
    )
    assert artifact.features == "word-count-v1"
    assert artifact.feature_names == ("n_negative_words", "n_positive_words", "n_words")
    assert artifact.admitted, artifact.reasons

    from prismyra.learn.fit import StudentArtifact

    restored = StudentArtifact.from_json(json.loads(json.dumps(artifact.to_json())))
    assert restored == artifact
    probs = restored.predict_proba("this is amazing news item 0")
    assert set(probs) == {"negative", "positive"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_fit_student_raises_on_an_unregistered_features_name(tmp_path):
    _spec_path, spec = make_spec(tmp_path, features="definitely-not-registered-v1")
    tag = spec.by_task(TASK)
    train = [_exp(spec, ctx, p, i) for i, (ctx, p, _) in enumerate(synthetic_rows(5))]

    from prismyra.learn.features import FeatureError
    from prismyra.learn.fit import fit_student

    with pytest.raises(FeatureError, match="no feature builder named"):
        fit_student(tag, train=train, dims=16, num_threads=1, seed=0, num_boost_round=5)
