"""Stage 4: deciding, for one already-tagged request, whether Prismyra or its distilled student answers it.
DISTILL-RL-DESIGN-v2.md section 4:

    経路は決定的にする: hash(tag, 正規化した入力) mod 10000 < ratio かつ、生徒の最大の確率が tau 以上なら生徒、
    それ以外は Prismyra。同じ入力と同じ表の版なら、いつも同じ経路を通る。

This module is the pure decision, not the state that feeds it. ``ratio`` (how much traffic a tag currently
sends to its student) and ``audit_fraction`` (how much of that traffic is also checked against Prismyra) are
read, never written, here -- `prismyra.learn.controller.Controller` owns both, because they move over time
(a ramp climbing, a bandit picking a new arm, a rollback cutting one back to zero) in a way a per-request
decision must not re-derive on its own. ``decide()`` below takes them as of *this* call and nothing it does
depends on when the call happens, which is what section 4's "同じ入力と同じ表の版なら、いつも同じ経路を通る"
is asking for: freeze ``ratio``/``audit_fraction`` (one snapshot, named by `params_version`) and the same tag
and the same normalized input always produce the same `RouteOutcome`.

Four strategies, all built on the same `bucket_for` hash, matching the design's own list in order:

* ``"ab"`` -- a fixed ``ratio`` (DISTILL-RL-DESIGN-v2.md's plain "決まった割合で分ける"). The controller never
  changes it on its own; an operator edits the spec to change the split.
* ``"ramp"`` -- the controller raises ``ratio`` by `prismyra.learn.spec.RouterConfig.ramp_step` every
  `ramp_interval_s`, while the audited comparison keeps clearing `rollback_min_match_rate` and
  `rollback_max_drift`, and snaps back down (and halts further climbing) the moment either does not.
* ``"shadow"`` -- never serves the student (`decide()` always returns ``served_by="prismyra"``), but always
  runs it anyway and always records the comparison (`audited` is unconditionally `True`) -- the design's own
  "生徒にも答えさせるが、返すのは Prismyra の答え。生徒の答えは比べるために記録だけする".
* ``"confidence"`` -- the student runs first; its own top probability has to clear `RouterConfig.tau` (the
  design's own "生徒の最大の確率が tau 以上なら生徒"), and ``ratio`` acts as *this mode's* eligibility gate on
  top of that, not a second source of randomness -- a request whose bucket does not fall under ``ratio`` is
  never offered to the student at all, confident or not. A tag that configures ``"confidence"`` and leaves
  ``ratio`` at its own default (0) is read as "no extra gate, tau decides alone" -- see `confidence_ratio_gate`.
* ``"bandit"`` -- structurally identical to ``"ramp"``; only how the controller picks ``ratio`` among
  `RouterConfig.bandit_arms` differs (a UCB1 pick over each arm's own observed audited agreement rate, in
  `prismyra.learn.controller`), not how a single request is routed once that choice has been made.

**The audit lane.** Any mode that can serve the student samples a second, independent bucket
(`bucket_for(..., salt="audit:")`, a different hash so the two decisions are not correlated) against
``audit_fraction``. When that lane fires *and* the student was the primary answer, Prismyra is also asked (the
request still needs the device, only for comparison -- the response the caller gets is unchanged). When the
primary answer is Prismyra, the student is cheap enough (a CPU prediction, no device) that a hit on the same
lane also runs it, purely to keep the comparison data two-sided -- section 4's own bias audit ("生徒と
Prismyra のどちらに回ったかを...集計し、特定の型が片方に固まっていないかを見る") needs both directions, not
only "was the student right when it served".
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .spec import RouterConfig, TagSpec


def bucket_for(tag: str, normalized_input: str, *, salt: str = "") -> int:
    """``hash(tag, normalized_input) mod 10000``, exactly as DISTILL-RL-DESIGN-v2.md section 4 specifies it.
    ``hashlib.sha256``, not Python's own `hash`: the same determinism concern `features._bucket` already
    documents (`hash()` is salted per-process unless `PYTHONHASHSEED` is fixed, and a routing decision that
    could change between two runs of the identical server is exactly what this function exists to rule out).
    ``salt`` separates the primary routing lane from the audit lane (`decide()` below) -- two different hashes
    of the same input, not the same hash compared against two different numbers, so that which requests land
    in the audit sample is not perfectly correlated with which ones the primary lane already routed to the
    student.
    """
    digest = hashlib.sha256(f"{salt}\x00{tag}\x00{normalized_input}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def params_version(mode: str, ratio: int, tau: float | None, audit_fraction: int) -> str:
    """A short, content-addressed name for "the routing table in effect right now" -- what a later reader of
    an `Experience` row (`route_params_version`) uses to tell whether two rows were routed under the same
    ratio/tau, the same way `prismyra.learn.spec.LearnSpec.version` already names a spec file's own content.
    """
    payload = json.dumps({"mode": mode, "ratio": ratio, "tau": tau, "audit_fraction": audit_fraction}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def confidence_ratio_gate(cfg: RouterConfig) -> int:
    """``"confidence"``'s own reading of ``ratio``: left at the shared default (0, meant for ``"ab"``/``"ramp"``/
    ``"bandit"``'s "start at zero traffic" story) it means "no extra gate, `tau` decides alone" -- a ratio of
    literally zero would otherwise make a confidence-mode tag permanently eligible for nothing, which no
    operator configuring this mode could have meant. Any explicit non-zero ``ratio`` is read exactly as every
    other mode reads it: a bucket cut, not a probability.
    """
    return 10_000 if cfg.ratio == 0 else cfg.ratio


@dataclass(frozen=True)
class RouteOutcome:
    """Everything one `decide()` call produced, ready for both halves of the request: which answer to use
    (`served_by`/`answer_probabilities`, needed before the engine would otherwise be called) and what to log
    once the response has been sent (`prismyra.learn.hook.LearnHook.record_route` reads every other field).

    ``request_id`` is minted here, once per `decide()` call, specifically so it is the *same* id whichever
    branch below returns: a caller that gets this request's answer back and a background task that logs it
    afterwards must agree on one id without either having to invent it independently. `prismyra.server`
    echoes it back to the caller (the design's own "応答にも、同じ id を返す"), and `POST /v1/feedback` is the
    one way a label or a reward can ever be attached back to this exact row.
    """

    request_id: str
    served_by: str  #: ``"prismyra"`` or ``f"student:{student_version}"``.
    mode: str
    bucket: int
    params_version: str
    audited: bool  #: this request's outcome was also checked against the other side.
    needs_engine: bool  #: the device is required to answer this request (``served_by == "prismyra"``) or to
    #: audit it (``served_by`` starts with ``"student"`` and ``audited``).
    student_probabilities: Mapping[str, float] | None  #: set whenever the student actually ran, whether or
    #: not it ended up answering (a "confidence" fallback runs it to check `tau` either way).
    answer_probabilities: Mapping[str, float] | None  #: set exactly when ``served_by`` starts with
    #: ``"student"`` -- the probabilities the caller gets back, computed before the engine is ever asked.


#: ``() -> probabilities``. A thunk, not the probabilities themselves, so `decide()` never pays for a
#: LightGBM prediction a given call turns out not to need (``"ab"``/``"ramp"``/``"bandit"`` routing Prismyra
#: with no audit hit on this request calls it zero times).
StudentPredict = Callable[[], Mapping[str, float]]


def decide(
    tag: TagSpec,
    cfg: RouterConfig,
    *,
    normalized_input: str,
    ratio: int,
    audit_fraction: int,
    student_version: str,
    student_predict: StudentPredict,
) -> RouteOutcome:
    """One routing decision. ``ratio`` and ``audit_fraction`` are the controller's current numbers for this
    tag (``"ab"`` passes `RouterConfig.ratio` unchanged forever; every other mode passes whatever
    `prismyra.learn.controller.Controller` computed for the snapshot named by the returned `params_version`).
    Pure and side-effect-free: calling this twice with the same arguments (including the same
    ``student_predict`` closure, which `decide` itself calls at most once) gives the same `RouteOutcome`,
    which is the whole of section 4's determinism contract once `ratio`/`audit_fraction` are themselves fixed.
    """
    request_id = uuid.uuid4().hex
    bucket = bucket_for(tag.task, normalized_input)
    pv = params_version(cfg.mode, ratio, cfg.tau, audit_fraction)

    if cfg.mode == "shadow":
        student_probs = student_predict()
        return RouteOutcome(
            request_id=request_id,
            served_by="prismyra",
            mode=cfg.mode,
            bucket=bucket,
            params_version=pv,
            audited=True,
            needs_engine=True,
            student_probabilities=student_probs,
            answer_probabilities=None,
        )

    if cfg.mode == "confidence":
        student_probs = student_predict()
        top_p = max(student_probs.values()) if student_probs else 0.0
        # `ratio` here is already `prismyra.learn.controller.Controller`'s resolved number -- it has already
        # applied `confidence_ratio_gate`'s "0 means no extra gate" reading and any rollback halt, so this
        # branch treats it exactly like "ab"/"ramp"/"bandit"'s own bucket cut below.
        eligible = bucket < ratio
        confident = cfg.tau is not None and top_p >= cfg.tau
        if eligible and confident:
            audited = bucket_for(tag.task, normalized_input, salt="audit:") < audit_fraction
            return RouteOutcome(
                request_id=request_id,
                served_by=f"student:{student_version}",
                mode=cfg.mode,
                bucket=bucket,
                params_version=pv,
                audited=audited,
                needs_engine=audited,
                student_probabilities=student_probs,
                answer_probabilities=student_probs,
            )
        # Not eligible, or not confident enough: fall back to Prismyra. The student already ran to check
        # `tau`, so recording its answer here is free -- this is the case coordinator feedback named
        # explicitly ("自信が足りなくて Prismyra に戻した場合は、生徒の確率も記録する").
        return RouteOutcome(
            request_id=request_id,
            served_by="prismyra",
            mode=cfg.mode,
            bucket=bucket,
            params_version=pv,
            audited=True,
            needs_engine=True,
            student_probabilities=student_probs,
            answer_probabilities=None,
        )

    # "ab" / "ramp" / "bandit": one shared shape. The controller is what makes them differ -- by how it
    # computes `ratio` across calls -- not this branch.
    if bucket < ratio:
        audited = bucket_for(tag.task, normalized_input, salt="audit:") < audit_fraction
        student_probs = student_predict()
        return RouteOutcome(
            request_id=request_id,
            served_by=f"student:{student_version}",
            mode=cfg.mode,
            bucket=bucket,
            params_version=pv,
            audited=audited,
            needs_engine=audited,
            student_probabilities=student_probs,
            answer_probabilities=student_probs,
        )
    audited = bucket_for(tag.task, normalized_input, salt="audit:") < audit_fraction
    audit_student_probs = student_predict() if audited else None
    return RouteOutcome(
        request_id=request_id,
        served_by="prismyra",
        mode=cfg.mode,
        bucket=bucket,
        params_version=pv,
        audited=audited,
        needs_engine=True,
        student_probabilities=audit_student_probs,
        answer_probabilities=None,
    )
