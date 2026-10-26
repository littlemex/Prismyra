"""`prismyra.learn.router`: the pure routing decision (DISTILL-RL-DESIGN-v2.md section 4), with a fake
``student_predict`` thunk standing in for a real `prismyra.learn.students.StudentHandle` -- `decide()` never
touches LightGBM itself, so none of this needs the `learn` extra installed.
"""

from __future__ import annotations

import pytest

from prismyra.learn.router import bucket_for, decide, params_version
from prismyra.learn.spec import RouterConfig, TagSpec

TAG = TagSpec(
    task="snake-move-v1",
    question="which way?",
    options=("up", "down"),
    kind="choice",
    normalize="none",
    retain_days=30,
    keep_hidden=False,
    eval_set="e.jsonl",
)


def ab(ratio: int, **kw) -> RouterConfig:
    return RouterConfig(mode="ab", student_path="s.json", ratio=ratio, **kw)


def confidence(tau: float, ratio: int = 0) -> RouterConfig:
    return RouterConfig(mode="confidence", student_path="s.json", tau=tau, ratio=ratio)


def shadow() -> RouterConfig:
    return RouterConfig(mode="shadow", student_path="s.json")


def predict_fn(probs):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return probs

    fn.calls = calls
    return fn


def test_bucket_for_is_pure_and_deterministic():
    a = bucket_for("snake-move-v1", "board one")
    b = bucket_for("snake-move-v1", "board one")
    assert a == b
    assert 0 <= a < 10_000


def test_bucket_for_is_sensitive_to_both_tag_and_input():
    # Not a claim that any one pair of buckets must differ (a hash can coincide) -- the claim is that varying
    # either the tag or the input, over a spread of values, disagrees somewhere, which a hash that silently
    # ignored one of the two arguments could never do.
    buckets_by_tag = {bucket_for(f"tag-{i}", "same input") for i in range(20)}
    assert len(buckets_by_tag) > 1
    buckets_by_input = {bucket_for("same-tag", f"input {i}") for i in range(20)}
    assert len(buckets_by_input) > 1


def test_audit_lane_uses_a_different_hash_than_the_primary_lane():
    primary = bucket_for("t", "x")
    audit = bucket_for("t", "x", salt="audit:")
    assert primary != audit  # not a mathematical certainty, but true for this fixed input/salt pair
    assert bucket_for("t", "x", salt="audit:") == bucket_for("t", "x", salt="audit:")  # still deterministic


def test_decide_is_deterministic_given_a_frozen_ratio_and_audit_fraction():
    cfg = ab(5000)
    outcomes = [
        decide(
            TAG,
            cfg,
            normalized_input="board one",
            ratio=5000,
            audit_fraction=0,
            student_version="v1",
            student_predict=predict_fn({"up": 0.9, "down": 0.1}),
        )
        for _ in range(5)
    ]
    served = {o.served_by for o in outcomes}
    buckets = {o.bucket for o in outcomes}
    assert len(served) == 1
    assert len(buckets) == 1


def test_ab_mode_splits_by_ratio_not_by_chance():
    cfg = ab(3000)
    student_fraction = 0
    n = 500
    for i in range(n):
        outcome = decide(
            TAG,
            cfg,
            normalized_input=f"board {i}",
            ratio=3000,
            audit_fraction=0,
            student_version="v1",
            student_predict=predict_fn({"up": 1.0, "down": 0.0}),
        )
        student_fraction += outcome.served_by.startswith("student")
    # 3000/10000 = 30%; a hash-based split over 500 distinct inputs should land close to that, not exactly.
    assert 0.20 < student_fraction / n < 0.40


def test_ab_mode_never_calls_the_student_when_bucket_lands_on_prismyra_and_not_audited():
    cfg = ab(0)  # ratio 0: bucket < 0 is never true
    predict = predict_fn({"up": 1.0, "down": 0.0})
    outcome = decide(
        TAG, cfg, normalized_input="x", ratio=0, audit_fraction=0, student_version="v1", student_predict=predict
    )
    assert outcome.served_by == "prismyra"
    assert outcome.needs_engine is True
    assert outcome.student_probabilities is None
    assert predict.calls["n"] == 0  # the whole saving: never pay for the student when it will not be used


def test_ab_mode_served_by_student_and_not_audited_needs_no_engine():
    cfg = ab(10_000)  # always eligible
    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=10_000,
        audit_fraction=0,
        student_version="v1",
        student_predict=predict_fn({"up": 0.8, "down": 0.2}),
    )
    assert outcome.served_by == "student:v1"
    assert outcome.needs_engine is False
    assert outcome.answer_probabilities == {"up": 0.8, "down": 0.2}
    assert outcome.audited is False


def test_an_audited_student_served_request_still_needs_the_engine_but_keeps_the_students_answer():
    cfg = ab(10_000)
    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=10_000,
        audit_fraction=10_000,  # always audited
        student_version="v1",
        student_predict=predict_fn({"up": 0.8, "down": 0.2}),
    )
    assert outcome.served_by == "student:v1"
    assert outcome.needs_engine is True
    assert outcome.audited is True
    assert outcome.answer_probabilities == {"up": 0.8, "down": 0.2}  # the caller still gets the student's answer


def test_shadow_mode_never_serves_the_student_but_always_runs_and_audits_it():
    cfg = shadow()
    predict = predict_fn({"up": 0.1, "down": 0.9})
    outcome = decide(
        TAG, cfg, normalized_input="x", ratio=0, audit_fraction=0, student_version="v1", student_predict=predict
    )
    assert outcome.served_by == "prismyra"
    assert outcome.needs_engine is True
    assert outcome.audited is True
    assert outcome.answer_probabilities is None  # the response is never the student's, even though it ran
    assert outcome.student_probabilities == {"up": 0.1, "down": 0.9}
    assert predict.calls["n"] == 1


def test_confidence_mode_serves_the_student_when_eligible_and_confident():
    cfg = confidence(tau=0.7, ratio=10_000)
    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=10_000,
        audit_fraction=0,
        student_version="v1",
        student_predict=predict_fn({"up": 0.95, "down": 0.05}),
    )
    assert outcome.served_by == "student:v1"
    assert outcome.needs_engine is False


def test_confidence_mode_falls_back_to_prismyra_when_not_confident_and_still_records_the_students_answer():
    cfg = confidence(tau=0.7, ratio=10_000)
    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=10_000,
        audit_fraction=0,  # irrelevant to this fallback: it is unconditional, per the design
        student_version="v1",
        student_predict=predict_fn({"up": 0.55, "down": 0.45}),
    )
    assert outcome.served_by == "prismyra"
    assert outcome.needs_engine is True
    assert outcome.student_probabilities == {"up": 0.55, "down": 0.45}


def test_confidence_mode_default_ratio_zero_means_no_extra_gate():
    # cfg.ratio left at 0 (the shared "ab/ramp" default); the controller resolves this to 10_000 before
    # calling decide -- this test exercises decide() itself with that already-resolved value, confirming the
    # eligibility check is `bucket < ratio` with no special-casing inside decide() itself.
    cfg = confidence(tau=0.5, ratio=0)
    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=10_000,  # what prismyra.learn.controller.Controller would have resolved cfg.ratio == 0 to
        audit_fraction=0,
        student_version="v1",
        student_predict=predict_fn({"up": 0.9, "down": 0.1}),
    )
    assert outcome.served_by == "student:v1"


def test_every_outcome_carries_a_fresh_request_id():
    cfg = ab(5000)
    outcomes = [
        decide(
            TAG,
            cfg,
            normalized_input=f"x{i}",
            ratio=5000,
            audit_fraction=0,
            student_version="v1",
            student_predict=predict_fn({"up": 1.0, "down": 0.0}),
        )
        for i in range(10)
    ]
    assert len({o.request_id for o in outcomes}) == 10


def test_params_version_changes_with_ratio_and_tau():
    a = params_version("ab", 1000, None, 50)
    b = params_version("ab", 2000, None, 50)
    c = params_version("ab", 1000, 0.5, 50)
    assert a != b
    assert a != c
    assert params_version("ab", 1000, None, 50) == a


@pytest.mark.parametrize("mode", ["ab", "ramp", "bandit"])
def test_ab_ramp_and_bandit_share_the_same_bucket_decision_shape(mode):
    cfg = RouterConfig(mode=mode, student_path="s.json", bandit_arms=(0, 5000) if mode == "bandit" else ())
    below = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=10_000,
        audit_fraction=0,
        student_version="v1",
        student_predict=predict_fn({"up": 1.0, "down": 0.0}),
    )
    above = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=0,
        audit_fraction=0,
        student_version="v1",
        student_predict=predict_fn({"up": 1.0, "down": 0.0}),
    )
    assert below.served_by.startswith("student")
    assert above.served_by == "prismyra"
