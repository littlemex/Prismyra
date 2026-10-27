"""The state `prismyra.learn.router.decide` reads but never writes: one tag's current ``ratio`` and
``audit_fraction``, how both move over time, and the one thing allowed to move them down on its own --
rollback. DISTILL-RL-DESIGN-v2.md section 4:

    比率の上げ下げは、影と本番の両方の評価に加え、入力の分布の変化の検出(特徴の分布の距離)でも止める。

Four jobs, one per tag, re-evaluated at most once every `prismyra.learn.spec.RouterConfig.ramp_interval_s`
(ticked lazily, on whichever request happens to arrive after that much time has passed -- no background
thread, the same choice `prismyra.learn.experience.LocalExperienceLog._maybe_purge` already made for its own
rate-limited housekeeping):

1. **Ramp.** ``mode == "ramp"``: raise ``ratio`` by `RouterConfig.ramp_step`, clamped to
   ``[ramp_min_ratio, ramp_max_ratio]``, every tick that rollback did not fire on.
2. **Bandit.** ``mode == "bandit"``: pick the next tick's ``ratio`` from `RouterConfig.bandit_arms` by UCB1
   over each arm's own observed audited agreement rate -- `_pick_arm`. The same rollback check gates this
   exactly like ramp: a vetoed arm is treated as a fired rollback, not a candidate.
3. **Rollback.** Any mode whose traffic can reach the student (every mode except ``"shadow"``, which never
   serves it) is checked every tick: the audited agreement rate since the last tick has to clear
   `RouterConfig.rollback_min_match_rate`, and the input drift (`features.population_stability_index` between
   a frozen reference window and the current one) has to stay at or under `RouterConfig.rollback_max_drift`.
   Either failing cuts ``ratio`` to ``ramp_min_ratio`` (0 unless a tag raised that floor) and halts further
   raising for `RouterConfig.rollback_cooldown_s` -- `"ab"`'s own fixed ``ratio`` is not exempt: a flat 50/50
   split is still traffic the student answers, and a rollback-worthy regression there is still a regression.
4. **Audit sizing.** `RouterConfig.audit_min_fraction`/`audit_max_fraction` bound how much of a tag's traffic
   is also checked against the other side, scaled down as a tag's own request volume (an EWMA, `_VOLUME_HALF_LIFE_S`)
   grows -- "監査のために Prismyra に回す割合は、固定の 3% ではなく、タグの量と変化の速さに合わせて変える" is
   read here as "enough audited requests per unit time to say something, not a fixed cut of a now-large
   stream".

State persists as one JSON file (atomic write: a temp file in the same directory, then `os.replace`, so a
reader never observes a half-written file) under the learn log directory, loaded once at
`prismyra.learn.hook.LearnHook` construction and written back no more often than `_SAVE_INTERVAL_S` -- a
request-level update (`record`) touches only an in-memory, lock-guarded dict; only a tick, or the rate-limited
flush, touches disk.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .features import population_stability_index
from .router import RouteOutcome
from .spec import RouterConfig, TagSpec

#: How often `Controller.resolve`'s own state is allowed to hit disk, independent of how often a tick fires.
#: A request-level `record()` call only ever mutates the in-memory dict; this is what rate-limits the flush.
_SAVE_INTERVAL_S = 10.0

#: The EWMA half-life (section 4's own "量...に合わせて変える" volume signal) used by `_audit_fraction`.
_VOLUME_HALF_LIFE_S = 300.0

#: How many requests the drift window accumulates before it is compared against the reference and reset --
#: short of this, `features.population_stability_index`'s own smoothing makes a half-full window too noisy to
#: trust, so no drift check (and no rollback-by-drift) fires on it yet.
_DRIFT_WINDOW_MIN_N = 50


@dataclass
class TagState:
    """One tag's own row of `Controller`'s state file.

    ``mode`` is the `RouterConfig.mode` that was in effect when this row's ``ratio`` (and the rest of its
    bookkeeping) was last (re)initialised -- not a mirror of whatever mode a later call happens to pass in.
    It exists to fix a real bug found running the router by hand (RUN-xgb-student.md, the "ルーター実演で
    見つかった実装上の注意点" section): a tag run first in ``"shadow"`` mode persists ``ratio=0`` (shadow's
    own `_initial_ratio`, since shadow never serves the student and so never needs a real number); switching
    that same tag to ``"confidence"`` afterwards, against the *same* state file, found that `0` already on
    disk and used it as a literal ratio, silently skipping `confidence_ratio_gate`'s "ratio left at 0 means
    no extra gate, `tau` decides alone" rule -- a rule that only ever ran for a *brand-new* tag, never for one
    whose state already existed under a different mode. `_state_for` below is what closes that: a mismatch
    between the stored ``mode`` and the mode just asked for means the stored ``ratio``'s meaning no longer
    matches what is being asked of it, so the row is reinitialised through `_initial_ratio` exactly as if the
    tag had never been seen before, rather than reinterpreting a number that was never written for this mode.
    """

    ratio: int
    mode: str | None = None
    halted_until: float = 0.0
    last_tick: float = 0.0
    last_request: float = 0.0
    volume_ewma: float = 0.0
    audit_agree: int = 0
    audit_total: int = 0
    drift_reference: tuple[int, ...] | None = None
    drift_window_sum: list[int] = field(default_factory=list)
    drift_window_n: int = 0
    bandit_arm: int | None = None
    bandit_stats: dict[int, tuple[int, float]] = field(default_factory=dict)  # arm ratio -> (n, reward_sum)
    bandit_steps: int = 0
    rollback_events: int = 0

    def to_json(self) -> dict:
        return {
            "ratio": self.ratio,
            "mode": self.mode,
            "halted_until": self.halted_until,
            "last_tick": self.last_tick,
            "last_request": self.last_request,
            "volume_ewma": self.volume_ewma,
            "audit_agree": self.audit_agree,
            "audit_total": self.audit_total,
            "drift_reference": list(self.drift_reference) if self.drift_reference is not None else None,
            "drift_window_sum": list(self.drift_window_sum),
            "drift_window_n": self.drift_window_n,
            "bandit_arm": self.bandit_arm,
            "bandit_stats": {str(k): list(v) for k, v in self.bandit_stats.items()},
            "bandit_steps": self.bandit_steps,
            "rollback_events": self.rollback_events,
        }

    @staticmethod
    def from_json(raw: dict) -> TagState:
        ref = raw.get("drift_reference")
        return TagState(
            ratio=raw["ratio"],
            # `.get(..., None)`, not a hard requirement: a state file written before this field existed
            # (RUN-xgb-student.md's own ramp/shadow/confidence demo, fixed by renaming `--learn-log-dir` per
            # phase rather than by this field) has no `mode` at all. Reading that as `None` -- "unknown mode"
            # -- rather than guessing one is what makes `_state_for` below treat an upgraded, pre-existing
            # state file exactly like a mode switch: the first `resolve()` call after the upgrade reinitialises
            # it through `_initial_ratio`, once, instead of silently trusting a number recorded under a mode
            # this field cannot name.
            mode=raw.get("mode"),
            halted_until=raw.get("halted_until", 0.0),
            last_tick=raw.get("last_tick", 0.0),
            last_request=raw.get("last_request", 0.0),
            volume_ewma=raw.get("volume_ewma", 0.0),
            audit_agree=raw.get("audit_agree", 0),
            audit_total=raw.get("audit_total", 0),
            drift_reference=tuple(ref) if ref is not None else None,
            drift_window_sum=list(raw.get("drift_window_sum", [])),
            drift_window_n=raw.get("drift_window_n", 0),
            bandit_arm=raw.get("bandit_arm"),
            bandit_stats={int(k): tuple(v) for k, v in raw.get("bandit_stats", {}).items()},
            bandit_steps=raw.get("bandit_steps", 0),
            rollback_events=raw.get("rollback_events", 0),
        )


def _initial_ratio(cfg: RouterConfig) -> int:
    if cfg.mode == "ramp":
        return cfg.ramp_min_ratio
    if cfg.mode == "bandit":
        return min(cfg.bandit_arms)
    if cfg.mode == "confidence" and cfg.ratio == 0:
        return 10_000  # confidence_ratio_gate's own reading: unset ratio means "no extra gate"
    return cfg.ratio


def _pick_arm(cfg: RouterConfig, st: TagState) -> int:
    """UCB1 over `RouterConfig.bandit_arms`, by each arm's own observed audited agreement rate. An arm with
    no observations yet is always preferred (the standard UCB1 initialisation: an untried arm's upper
    confidence bound is unbounded), so every arm gets tried at least once before the bound starts
    discriminating between them.
    """
    for arm in cfg.bandit_arms:
        if arm not in st.bandit_stats or st.bandit_stats[arm][0] == 0:
            return arm
    total = sum(n for n, _ in st.bandit_stats.values())
    best_arm, best_score = cfg.bandit_arms[0], -math.inf
    for arm in cfg.bandit_arms:
        n, reward_sum = st.bandit_stats[arm]
        mean = reward_sum / n
        bound = mean + math.sqrt(2.0 * math.log(max(total, 1)) / n)
        if bound > best_score:
            best_arm, best_score = arm, bound
    return best_arm


class Controller:
    """One instance per server process, shared by every tag that configured a `RouterConfig`."""

    def __init__(self, state_path: str | Path, *, now: Callable[[], float] = time.time):
        self.state_path = Path(state_path)
        self._now = now
        self._lock = threading.Lock()
        self._states: dict[str, TagState] = {}
        self._last_saved = 0.0
        self._load()

    def _load(self) -> None:
        if not self.state_path.is_file():
            return
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._states = {task: TagState.from_json(row) for task, row in raw.items()}

    def _save_locked(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {task: st.to_json() for task, st in self._states.items()}
        fd, tmp_name = tempfile.mkstemp(dir=self.state_path.parent, prefix=".controller-", suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp_name, self.state_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        self._last_saved = self._now()

    def _maybe_save_locked(self) -> None:
        if self._now() - self._last_saved >= _SAVE_INTERVAL_S:
            self._save_locked()

    def flush(self) -> None:
        """Force an immediate write -- `prismyra.learn.hook.LearnHook.stop`'s own shutdown path, so a
        process that exits between two rate-limited saves does not lose the state it accumulated since the
        last one."""
        with self._lock:
            self._save_locked()

    def _state_for(self, tag: str, cfg: RouterConfig) -> TagState:
        st = self._states.get(tag)
        if st is not None and st.mode != cfg.mode:
            # This tag's persisted ratio was last initialised for a *different* mode (see `TagState.mode`'s
            # own docstring for the shadow -> confidence bug this fixes) -- its numeric value carries that
            # mode's meaning, not this one's, so it is not safe to keep. `st.mode is None` (a state file
            # written before this field existed) also mismatches here, by the same logic: an unlabelled
            # ratio is exactly as untrustworthy as one labelled for a different mode. Start this mode's own
            # bookkeeping from scratch, the same way a tag this `Controller` has never seen at all would.
            st = None
        if st is None:
            st = TagState(ratio=_initial_ratio(cfg), mode=cfg.mode)
            if cfg.mode == "bandit":
                st.bandit_arm = st.ratio
            self._states[tag] = st
        return st

    def resolve(self, tag: TagSpec, cfg: RouterConfig) -> tuple[int, int]:
        """``(ratio, audit_fraction)`` for this call, ticking first if a tick is due. Called once per routed
        request, before `prismyra.learn.router.decide`.
        """
        now = self._now()
        with self._lock:
            st = self._state_for(tag.task, cfg)
            if cfg.mode in ("ramp", "bandit") and now - st.last_tick >= cfg.ramp_interval_s:
                self._tick_locked(cfg, st, now)
            halted = now < st.halted_until
            ratio = 0 if halted else st.ratio
            audit_fraction = self._audit_fraction_locked(cfg, st)
            self._maybe_save_locked()
            return ratio, audit_fraction

    def peek(self, tag: TagSpec, cfg: RouterConfig) -> tuple[int, int]:
        """`resolve`'s read-only half: the current ``(ratio, audit_fraction)`` without ticking. `/stats`
        (`prismyra.learn.hook.LearnHook.stats`) uses this -- a monitoring read must not itself be what
        advances a ramp or retires a bandit arm.
        """
        now = self._now()
        with self._lock:
            st = self._state_for(tag.task, cfg)
            halted = now < st.halted_until
            ratio = 0 if halted else st.ratio
            return ratio, self._audit_fraction_locked(cfg, st)

    def _audit_fraction_locked(self, cfg: RouterConfig, st: TagState) -> int:
        """More volume -> a smaller fraction still yields enough audited requests per unit time; little or
        no volume -> the fraction rises to `audit_max_fraction` so a cold tag is not left unauditable.
        ``target_per_window`` is "about this many audited requests per `ramp_interval_s`-sized window" --
        fixed small rather than configurable, because this is a sizing heuristic, not a tunable an operator
        should need to reason about per tag. ``st.volume_ewma`` is itself a request-per-second rate (`record`
        maintains it that way), so the comparison converts both sides to the same units: a target *rate* of
        audited requests, divided by the observed *rate* of all requests, is the fraction of them that need
        to be audited to hit the target.
        """
        if st.volume_ewma <= 0:
            return cfg.audit_max_fraction
        target_per_window = 20.0
        target_rate = target_per_window / max(cfg.ramp_interval_s, 1.0)
        fraction = round((target_rate / st.volume_ewma) * 10_000)
        return max(cfg.audit_min_fraction, min(cfg.audit_max_fraction, fraction))

    def _tick_locked(self, cfg: RouterConfig, st: TagState, now: float) -> None:
        rolled_back = self._check_rollback_locked(cfg, st, now)
        st.last_tick = now
        if rolled_back:
            return
        if cfg.mode == "ramp":
            st.ratio = min(cfg.ramp_max_ratio, st.ratio + cfg.ramp_step)
        elif cfg.mode == "bandit":
            # Attribute the window just closed to the arm that was actually live during it, then pick the
            # next one -- `_pick_arm` reads `st.bandit_stats`, which `record()` has been updating all along.
            st.bandit_steps += 1
            st.bandit_arm = _pick_arm(cfg, st)
            st.ratio = st.bandit_arm
        # Each tick closes one audited-agreement window; the next one starts counting from zero so an old
        # tick's numbers cannot keep gating a much later one.
        st.audit_agree = 0
        st.audit_total = 0

    def _check_rollback_locked(self, cfg: RouterConfig, st: TagState, now: float) -> bool:
        if cfg.mode == "shadow":
            return False  # nothing to roll back: shadow never serves the student
        fired = False
        if st.audit_total > 0:
            rate = st.audit_agree / st.audit_total
            if rate < cfg.rollback_min_match_rate:
                fired = True
        if st.drift_window_n >= _DRIFT_WINDOW_MIN_N:
            current = tuple(c for c in st.drift_window_sum)
            if st.drift_reference is None:
                st.drift_reference = current  # the first full window becomes the baseline everything else compares to
            else:
                drift = population_stability_index(st.drift_reference, current)
                if drift > cfg.rollback_max_drift:
                    fired = True
            st.drift_window_sum = [0] * len(st.drift_window_sum)
            st.drift_window_n = 0
        if fired:
            st.ratio = cfg.ramp_min_ratio if cfg.mode == "ramp" else _initial_ratio(cfg)
            st.halted_until = now + cfg.rollback_cooldown_s
            st.rollback_events += 1
        return fired

    def record(
        self,
        tag: TagSpec,
        cfg: RouterConfig,
        outcome: RouteOutcome,
        *,
        agreement: bool | None,
        drift_vector: tuple[int, ...] | None,
    ) -> None:
        """Folds one request's outcome into this tag's state: request volume (for `_audit_fraction_locked`),
        the audited agreement tally (for `_check_rollback_locked` and, in ``"bandit"`` mode, the arm that was
        live when this request was answered), and the drift window (`features.feature_summary`'s histogram,
        accumulated, never the raw context). Called from the same background task that logs the `Experience`
        row (`prismyra.learn.hook.LearnHook.record_route`), after the response has already been sent.
        """
        now = self._now()
        with self._lock:
            st = self._state_for(tag.task, cfg)
            elapsed = now - st.last_request if st.last_request else 0.0
            st.last_request = now
            if elapsed > 0:
                # A request-per-second rate, EWMA'd across irregular inter-arrival gaps: each new gap's own
                # instantaneous rate (`1 / elapsed`) is blended in, weighted by how much of the half-life has
                # passed since the last one. Keyed by *rate*, not by "did a request just arrive" (which is
                # trivially true every time this method is called at all, and so carries no rate information
                # on its own) -- `_audit_fraction_locked` depends on this being a rate to compare against a
                # target rate in the same units.
                instantaneous_rate = 1.0 / elapsed
                decay = math.exp(-elapsed / _VOLUME_HALF_LIFE_S)
                st.volume_ewma = st.volume_ewma * decay + instantaneous_rate * (1.0 - decay)
            if drift_vector is not None:
                if not st.drift_window_sum:
                    st.drift_window_sum = [0] * len(drift_vector)
                for i, v in enumerate(drift_vector):
                    st.drift_window_sum[i] += v
                st.drift_window_n += 1
            if outcome.audited and agreement is not None:
                st.audit_agree += int(agreement)
                st.audit_total += 1
                if cfg.mode == "bandit" and st.bandit_arm is not None:
                    # Credited to whichever arm is *currently* active, not gated on `served_by` -- an audited
                    # outcome exists (both sides are known) whether this particular request was served by the
                    # student or by Prismyra, and an arm whose ratio routes nothing to the student at all
                    # (`ratio == 0`) still needs a way to be marked "tried": it can only ever get one from its
                    # own audit lane, never from `served_by.startswith("student")`, which such an arm can
                    # never produce. Wrong only in the narrow window a tick just moved `bandit_arm`, which the
                    # next full window's statistics absorb.
                    n, s = st.bandit_stats.get(st.bandit_arm, (0, 0.0))
                    st.bandit_stats[st.bandit_arm] = (n + 1, s + (1.0 if agreement else 0.0))
            self._maybe_save_locked()
