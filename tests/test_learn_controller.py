"""`prismyra.learn.controller.Controller`: ramp, bandit, rollback and audit sizing. A fake clock
(``FakeClock``) stands in for `time.time` throughout, so a tick can be driven by advancing a counter instead
of sleeping -- every test below is deterministic and runs in milliseconds.
"""

from __future__ import annotations

from dataclasses import replace

from prismyra.learn.controller import Controller, TagState
from prismyra.learn.router import decide
from prismyra.learn.spec import RouterConfig, TagSpec

TAG = TagSpec(
    task="t",
    question="q",
    options=("a", "b"),
    kind="choice",
    normalize="none",
    retain_days=30,
    keep_hidden=False,
    eval_set="e.jsonl",
)


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def predict_fn(probs):
    return lambda: probs


def test_ab_mode_ratio_is_fixed_and_never_ramps(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(mode="ab", student_path="s.json", ratio=3000)
    ratio, _ = ctrl.resolve(TAG, cfg)
    assert ratio == 3000
    clock.advance(10_000)
    ratio, _ = ctrl.resolve(TAG, cfg)
    assert ratio == 3000  # ab never ticks: no ramp_interval_s logic applies to it


def test_ramp_mode_climbs_by_step_every_interval_when_nothing_fails(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(mode="ramp", student_path="s.json", ramp_step=1000, ramp_interval_s=60, ramp_max_ratio=10_000)
    ratio, _ = ctrl.resolve(TAG, cfg)
    assert ratio == 0  # starts at ramp_min_ratio
    for expected in (1000, 2000, 3000):
        clock.advance(60)
        ratio, _ = ctrl.resolve(TAG, cfg)
        assert ratio == expected


def test_ramp_mode_stops_climbing_at_ramp_max_ratio(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(mode="ramp", student_path="s.json", ramp_step=6000, ramp_interval_s=10, ramp_max_ratio=10_000)
    for _ in range(5):
        clock.advance(10)
        ratio, _ = ctrl.resolve(TAG, cfg)
    assert ratio == 10_000


def test_rollback_fires_when_audited_agreement_drops_below_the_bar_and_resets_ratio(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(
        mode="ramp",
        student_path="s.json",
        ramp_step=5000,
        ramp_interval_s=10,
        ramp_max_ratio=10_000,
        rollback_min_match_rate=0.9,
        rollback_cooldown_s=100,
    )
    ratio, _ = ctrl.resolve(TAG, cfg)
    # Feed a run of disagreeing audited outcomes before the next tick -- `record` is what a background task
    # calls after every routed request, win or lose.
    for _ in range(30):
        outcome = decide(
            TAG,
            cfg,
            normalized_input="x",
            ratio=ratio,
            audit_fraction=10_000,
            student_version="v1",
            student_predict=predict_fn({"a": 1.0, "b": 0.0}),
        )
        ctrl.record(TAG, cfg, outcome, agreement=False, drift_vector=None)
    clock.advance(10)  # the tick that should notice the bad agreement rate and roll back
    ratio, _ = ctrl.resolve(TAG, cfg)
    assert ratio == cfg.ramp_min_ratio


def test_rollback_halts_further_ramping_for_the_cooldown_period(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(
        mode="ramp",
        student_path="s.json",
        ramp_step=5000,
        ramp_interval_s=10,
        rollback_min_match_rate=0.9,
        rollback_cooldown_s=1000,
    )
    ctrl.resolve(TAG, cfg)
    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=0,
        audit_fraction=10_000,
        student_version="v1",
        student_predict=predict_fn({"a": 1.0, "b": 0.0}),
    )
    for _ in range(30):
        ctrl.record(TAG, cfg, outcome, agreement=False, drift_vector=None)
    clock.advance(10)
    ratio, _ = ctrl.resolve(TAG, cfg)
    assert ratio == 0
    clock.advance(10)  # still well inside the cooldown
    ratio, _ = ctrl.resolve(TAG, cfg)
    assert ratio == 0  # halted, not merely reset once


def test_rollback_fires_on_drift_once_the_window_is_full(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(
        mode="ramp",
        student_path="s.json",
        ramp_step=1000,
        ramp_interval_s=10,
        rollback_max_drift=0.1,
        rollback_min_match_rate=0.0,
    )
    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=0,
        audit_fraction=0,
        student_version="v1",
        student_predict=predict_fn({"a": 1.0, "b": 0.0}),
    )
    # First window: a stable distribution, concentrated in one bucket, becomes the reference.
    for _ in range(60):
        ctrl.record(TAG, cfg, outcome, agreement=None, drift_vector=(100, 0, 0, 0))
    clock.advance(10)  # closes the window, sets the reference, ticks (no drift to compare against yet)
    ratio_after_first_tick, _ = ctrl.resolve(TAG, cfg)
    assert ratio_after_first_tick == 1000  # nothing to roll back on the very first window

    # Second window: a sharply different distribution -- this is what should trip the drift check.
    for _ in range(60):
        ctrl.record(TAG, cfg, outcome, agreement=None, drift_vector=(0, 0, 0, 100))
    clock.advance(10)
    ratio_after_drift, _ = ctrl.resolve(TAG, cfg)
    assert ratio_after_drift == cfg.ramp_min_ratio


def test_bandit_mode_tries_every_untried_arm_before_discriminating(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(mode="bandit", student_path="s.json", ramp_interval_s=10, bandit_arms=(0, 2500, 5000))
    seen_ratios = set()
    ratio, _ = ctrl.resolve(TAG, cfg)
    seen_ratios.add(ratio)
    for _ in range(len(cfg.bandit_arms) - 1):
        outcome = decide(
            TAG,
            cfg,
            normalized_input="x",
            ratio=ratio,
            audit_fraction=10_000,
            student_version="v1",
            student_predict=predict_fn({"a": 1.0, "b": 0.0}),
        )
        ctrl.record(TAG, cfg, outcome, agreement=True, drift_vector=None)
        clock.advance(10)
        ratio, _ = ctrl.resolve(TAG, cfg)
        seen_ratios.add(ratio)
    assert seen_ratios == set(cfg.bandit_arms)


def test_bandit_mode_converges_on_the_arm_with_the_higher_agreement_rate(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    # `rollback_min_match_rate=0.0`: this test is about arm *selection*, already covered by the rollback
    # tests above, and the weaker arm's own 0.5 agreement rate would otherwise trip the default 0.85 bar
    # before the bandit ever gets to compare it against the stronger one.
    cfg = RouterConfig(
        mode="bandit", student_path="s.json", ramp_interval_s=10, bandit_arms=(2500, 7500), rollback_min_match_rate=0.0
    )
    rewards = {2500: 0.5, 7500: 0.95}

    ratio, _ = ctrl.resolve(TAG, cfg)
    for _tick in range(40):
        outcome = decide(
            TAG,
            cfg,
            normalized_input="x",
            ratio=ratio,
            audit_fraction=10_000,
            student_version="v1",
            student_predict=predict_fn({"a": 1.0, "b": 0.0}),
        )
        # Simulate many requests under the currently-active arm, each independently agreeing at that arm's
        # own rate -- a plain, reproducible stand-in for a real audited stream.
        n_samples = 20
        for i in range(n_samples):
            ok = i < round(rewards[ratio] * n_samples)
            ctrl.record(TAG, cfg, outcome, agreement=ok, drift_vector=None)
        clock.advance(10)
        ratio, _ = ctrl.resolve(TAG, cfg)

    assert ratio == 7500


def test_audit_fraction_shrinks_as_volume_grows(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(mode="ab", student_path="s.json", ratio=5000, audit_min_fraction=10, audit_max_fraction=2000)
    _, cold_fraction = ctrl.resolve(TAG, cfg)
    assert cold_fraction == cfg.audit_max_fraction  # no volume signal yet -- default to the safe maximum

    outcome = decide(
        TAG,
        cfg,
        normalized_input="x",
        ratio=5000,
        audit_fraction=0,
        student_version="v1",
        student_predict=predict_fn({"a": 1.0, "b": 0.0}),
    )
    for _ in range(200):
        clock.advance(0.1)
        ctrl.record(TAG, cfg, outcome, agreement=True, drift_vector=None)
    _, hot_fraction = ctrl.resolve(TAG, cfg)
    assert hot_fraction < cold_fraction
    assert cfg.audit_min_fraction <= hot_fraction <= cfg.audit_max_fraction


def test_peek_does_not_tick_but_resolve_does(tmp_path):
    clock = FakeClock()
    ctrl = Controller(tmp_path / "state.json", now=clock)
    cfg = RouterConfig(mode="ramp", student_path="s.json", ramp_step=1000, ramp_interval_s=10)
    ctrl.resolve(TAG, cfg)
    clock.advance(10)
    peeked_ratio, _ = ctrl.peek(TAG, cfg)
    assert peeked_ratio == 0  # peek must not advance the ramp on its own
    resolved_ratio, _ = ctrl.resolve(TAG, cfg)
    assert resolved_ratio == 1000
    peeked_again, _ = ctrl.peek(TAG, cfg)
    assert peeked_again == 1000  # reads back whatever resolve already ticked to


def test_state_survives_a_flush_and_reload(tmp_path):
    clock = FakeClock()
    state_path = tmp_path / "state.json"
    ctrl = Controller(state_path, now=clock)
    cfg = RouterConfig(mode="ramp", student_path="s.json", ramp_step=1000, ramp_interval_s=10)
    ctrl.resolve(TAG, cfg)
    clock.advance(10)
    ratio_before, _ = ctrl.resolve(TAG, cfg)
    ctrl.flush()

    reloaded = Controller(state_path, now=clock)
    ratio_after, _ = reloaded.peek(TAG, cfg)
    assert ratio_after == ratio_before


def test_tagstate_round_trips_through_json():
    st = TagState(ratio=2500, drift_reference=(1, 2, 3), bandit_stats={5000: (3, 1.5)})
    restored = TagState.from_json(st.to_json())
    assert restored == replace(st)
