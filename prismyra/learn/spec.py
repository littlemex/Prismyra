"""The registered-tag spec: a closed, versioned, operator-authored vocabulary of decisions to learn from.

DISTILL-RL-DESIGN-v2.md section 0.1 is the reason this exists as data rather than as a free-text ``task`` field a
caller can set to anything: a tag that any caller could invent is a hole a caller could pour anything through, and
the three independent reviews of v1 (DISTILL-RL-REVIEW.md P0 item 1) agreed that is a contamination risk, not a
feature. So a tag exists only if an operator put it in this file before any traffic arrived, and only requests
that match an entry here -- by name or by exact (question, option set, kind) -- are ever logged. See
``prismyra.learn.tagging`` for the matching rule itself; this module is only the file format and its validation.

File shape (a JSON list, paths -- the one in ``eval_set`` -- resolved relative to the spec file, the same
convention ``prismyra.heads`` already uses):

    [{"task": "phishing-v1", "question": "Is this email a phishing attempt?", "options": ["no", "yes"],
      "kind": "noul", "normalize": "strip+lower", "retain_days": 30, "keep_hidden": false,
      "eval_set": "evals/phishing-v1-gold.jsonl"}]

Every field above is required on every entry, and every entry is checked in full before the server is allowed to
start -- a typo'd field name or a bad type is a refusal to start, not a tag that silently never matches anything.
That is the one half of the "fail loudly" contract; the other half is in ``tagging.py``: a well-formed entry that
simply does not match a request is not an error, it is an untagged request, and an untagged request is answered
exactly as if ``--learn-spec`` had never been passed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

#: The exact field set an entry must have -- no more, no fewer. Extra fields are refused rather than ignored: a
#: silently-ignored field is how a typo (``"retain_day"``) ships for weeks before anyone notices retention was
#: never set.
REQUIRED_FIELDS = frozenset(
    {"task", "question", "options", "kind", "normalize", "retain_days", "keep_hidden", "eval_set"}
)

#: Fields an entry may have in addition to ``REQUIRED_FIELDS``, checked when present but never required --
#: an existing spec file with none of these keeps meaning exactly what it meant before. ``"features"`` names a
#: registered feature builder (``prismyra.learn.features.register_features``) stage 2's student should use for
#: this tag instead of the hashed-TF-IDF default; which builders exist is a runtime fact (an operator's own
#: module registers one by importing it), not something this file-format module can check against, so only the
#: type is validated here -- an unregistered name is a refusal at fit time (``prismyra.learn.features``), not at
#: spec-load time. ``"min_match_rate"`` and ``"min_accuracy_ratio"`` override, for this one tag, the two
#: admission-condition defaults below (``DEFAULT_MIN_MATCH_RATE``, ``DEFAULT_MIN_ACCURACY_RATIO``) that
#: ``prismyra.learn.fit.run`` otherwise applies to every tag. ``"router"`` and ``"retain_context"`` are stage 4
#: (DISTILL-RL-DESIGN-v2.md section 4): a tag with neither keeps meaning exactly what it meant before either
#: field existed -- no ``"router"`` means `prismyra.learn.router.decide` is never consulted for that tag, and
#: every one of its requests goes to Prismyra exactly as stage 1/2 always sent it.
OPTIONAL_FIELDS = frozenset({"features", "min_match_rate", "min_accuracy_ratio", "router", "retain_context"})

#: Stage 4's own closed vocabulary of routing strategies (docs/LEARN.md's "Stage 4" section has the full account
#: of each). Closed for the same reason `NORMALIZE_RULES` is: a typo'd mode string is a refusal at load time,
#: not a tag that silently never routes anything.
ROUTER_MODES = frozenset({"ab", "ramp", "shadow", "confidence", "bandit"})

#: How much of a tag's own `context` survives into its logged experience (section 8's "個人情報が入りうる
#: context を残すかどうかは、既存の retain_days と keep_hidden と同じく spec で決める"). `"full"` -- the only
#: behaviour stage 1/2 ever had -- stores the raw context exactly as before. `"features_only"` drops it and
#: keeps only the already-vectorised features a loaded student's own artifact can produce (nothing to keep
#: before a tag has one, in which case this degrades to `"none"`'s own behaviour for that record). `"none"`
#: drops the context and keeps only `feature_summary`'s lossy drift histogram. See
#: `prismyra.learn.hook.LearnHook.submit` for where this is applied -- never in this module, which validates
#: the *setting*, not a request.
RETAIN_CONTEXT_RULES = frozenset({"full", "features_only", "none"})

#: The one place a student's admission bar is a number. ``prismyra.learn.fit.run`` reads these -- never a
#: literal of its own -- for any tag whose entry does not set ``"min_match_rate"``/``"min_accuracy_ratio"``.
#: 2026-10-11: lowered from the original design's 0.97/1.00 (eval accuracy at least Prismyra's own) after an
#: operator decision that distillation's point is speed and cost, not matching a frontier model's own
#: mistakes -- a correct, 70-150x-faster student that disagrees with Prismyra on one board in ten is a student
#: worth shipping. Changing either number here changes the default for every tag that does not set its own; a
#: tag that genuinely needs a stricter (or looser) bar sets ``"min_match_rate"``/``"min_accuracy_ratio"`` in its
#: own entry instead of this file being edited for everyone.
DEFAULT_MIN_MATCH_RATE = 0.90
DEFAULT_MIN_ACCURACY_RATIO = 0.90


#: The closed set of normalization rules ``tagging.normalize`` knows how to apply. Not open-ended on purpose: a
#: rule is a few lines of string handling, and an operator who needs a new one is one PR away, not a regex they
#: hand-type into a JSON file where a mistake fails silently at match time instead of loudly at load time.
NORMALIZE_RULES = frozenset({"none", "strip", "lower", "strip+lower"})


class SpecError(Exception):
    """The spec file is malformed. Raised at load time -- before the server binds a port, before the engine loads
    a model -- so a bad spec is a refusal to start, never a silent gap in what gets logged. This is deliberately a
    plain ``Exception``, not ``prismyra.schema.PrismyraError``: a spec problem is an operator's configuration
    mistake, discovered before any request exists, not a request-time refusal a caller's exception handling needs
    to recognise.
    """


#: Stage 4's own defaults, defined once here for the same reason `DEFAULT_MIN_MATCH_RATE` is: a tag's entry
#: overrides what it needs to, everything else inherits a value that is never retyped at each call site.
DEFAULT_RAMP_STEP = 500
DEFAULT_RAMP_INTERVAL_S = 300.0
DEFAULT_RAMP_MIN_RATIO = 0
DEFAULT_RAMP_MAX_RATIO = 10_000
DEFAULT_AUDIT_MIN_FRACTION = 50
DEFAULT_AUDIT_MAX_FRACTION = 2_000
DEFAULT_ROLLBACK_MIN_MATCH_RATE = 0.85
DEFAULT_ROLLBACK_MAX_DRIFT = 0.35
DEFAULT_ROLLBACK_COOLDOWN_S = 900.0


@dataclass(frozen=True)
class RouterConfig:
    """Stage 4's switch for one tag (DISTILL-RL-DESIGN-v2.md section 4). Parsed from a tag entry's own
    ``"router"`` object; absent entirely for a tag that has not opted in, which is what keeps every tag written
    before this field existed meaning exactly what it meant before -- see `prismyra.learn.router`'s own
    docstring for how each `mode` uses these numbers.

    ``ratio``, every ``*_fraction`` and every ``*_ratio`` in this dataclass is a count per 10,000
    (`DISTILL-RL-DESIGN-v2.md`'s own `hash(...) mod 10000 < ratio`), an integer, not a float: a float ratio
    compared against an integer hash bucket is exactly the kind of binary-rounding mismatch that would make
    "the same input always takes the same path" stop being quite true at the edges.
    """

    mode: str
    student_path: str
    ratio: int = 0
    tau: float | None = None
    ramp_step: int = DEFAULT_RAMP_STEP
    ramp_interval_s: float = DEFAULT_RAMP_INTERVAL_S
    ramp_min_ratio: int = DEFAULT_RAMP_MIN_RATIO
    ramp_max_ratio: int = DEFAULT_RAMP_MAX_RATIO
    audit_min_fraction: int = DEFAULT_AUDIT_MIN_FRACTION
    audit_max_fraction: int = DEFAULT_AUDIT_MAX_FRACTION
    rollback_min_match_rate: float = DEFAULT_ROLLBACK_MIN_MATCH_RATE
    rollback_max_drift: float = DEFAULT_ROLLBACK_MAX_DRIFT
    rollback_cooldown_s: float = DEFAULT_ROLLBACK_COOLDOWN_S
    bandit_arms: tuple[int, ...] = ()


@dataclass(frozen=True)
class TagSpec:
    """One registered tag, exactly as section 1 of the design describes it."""

    task: str
    question: str
    options: tuple[str, ...]
    kind: str
    normalize: str
    retain_days: int
    keep_hidden: bool
    eval_set: str
    features: str | None = None
    min_match_rate: float = DEFAULT_MIN_MATCH_RATE
    min_accuracy_ratio: float = DEFAULT_MIN_ACCURACY_RATIO
    retain_context: str = "full"
    router: RouterConfig | None = None


@dataclass(frozen=True)
class LearnSpec:
    """The whole spec file: its tags, and a version that changes exactly when the file's bytes do.

    ``version`` is not a field an operator writes. The design calls for a "spec の版" to travel in every
    experience record's version tuple, so that a later consumer can tell whether two records were collected under
    the same tag definition -- but asking an operator to bump an integer by hand on every edit is exactly the kind
    of manual step that gets forgotten the one time it matters. A content hash cannot be forgotten: it is simply
    always correct, including about edits nobody meant to make.
    """

    path: Path
    version: str
    tags: tuple[TagSpec, ...]

    def by_task(self, task: str) -> TagSpec | None:
        for t in self.tags:
            if t.task == task:
                return t
        return None


def _fail(path: Path, where: str, message: str) -> NoReturn:
    raise SpecError(f"{path}: {where}: {message}")


def _check_admission_fraction(path: Path, where: str, field: str, value: object) -> float:
    """``"min_match_rate"`` and ``"min_accuracy_ratio"`` share one shape: a number in ``(0, 1]``. ``0`` would
    admit nothing able to clear a bound of zero items correctly, which is not a bar anyone would knowingly
    set, and both are fractions, not counts, so neither can exceed ``1``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < value <= 1.0):
        _fail(path, where, f"{field!r} must be a number in (0, 1], got {value!r}")
    return float(value)


def _check_fraction_int(path: Path, where: str, field: str, value: object, *, max_value: int = 10_000) -> int:
    """``ratio`` and every stage 4 ``*_fraction`` share one shape: an integer count per 10,000, in
    ``[0, max_value]``."""
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= max_value):
        _fail(path, where, f"{field!r} must be an integer in [0, {max_value}], got {value!r}")
    return value


def _check_positive_number(path: Path, where: str, field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        _fail(path, where, f"{field!r} must be a positive number, got {value!r}")
    return float(value)


def _check_router(path: Path, where: str, raw: dict) -> RouterConfig:
    """``"router"``'s own nested object. Checked in full, the same discipline every other field on an entry
    gets: a typo'd mode or a missing ``student_path`` is a refusal to start, not a tag that silently never
    routes anything to a student. See `prismyra.learn.router` for what each ``mode`` does with these numbers.
    """
    where = f"{where}.router"
    if not isinstance(raw, dict):
        _fail(path, where, f"must be a JSON object, got {type(raw).__name__}")

    required = {"mode", "student_path"}
    optional = {
        "ratio",
        "tau",
        "ramp_step",
        "ramp_interval_s",
        "ramp_min_ratio",
        "ramp_max_ratio",
        "audit_min_fraction",
        "audit_max_fraction",
        "rollback_min_match_rate",
        "rollback_max_drift",
        "rollback_cooldown_s",
        "bandit_arms",
    }
    missing = required - raw.keys()
    extra = raw.keys() - required - optional
    if missing:
        _fail(path, where, f"missing field(s): {sorted(missing)}")
    if extra:
        _fail(path, where, f"unknown field(s): {sorted(extra)} -- the allowed fields are {sorted(required | optional)}")

    mode = raw["mode"]
    if mode not in ROUTER_MODES:
        _fail(path, where, f"'mode' must be one of {sorted(ROUTER_MODES)}, got {mode!r}")

    student_path = raw["student_path"]
    if not isinstance(student_path, str) or not student_path.strip():
        msg = f"'student_path' must be a non-empty string (a path to a StudentArtifact), got {student_path!r}"
        _fail(path, where, msg)

    ratio = _check_fraction_int(path, where, "ratio", raw.get("ratio", 0))

    tau = None
    if "tau" in raw:
        tau = raw["tau"]
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not (0.0 <= tau <= 1.0):
            _fail(path, where, f"'tau' must be a number in [0, 1], got {tau!r}")
        tau = float(tau)
    if mode == "confidence" and tau is None:
        _fail(path, where, "mode 'confidence' requires 'tau'")

    bandit_arms: tuple[int, ...] = ()
    if "bandit_arms" in raw:
        arms = raw["bandit_arms"]
        if not isinstance(arms, list) or not arms:
            _fail(path, where, f"'bandit_arms' must be a non-empty list of integers, got {arms!r}")
        bandit_arms = tuple(_check_fraction_int(path, where, "bandit_arms[]", a) for a in arms)
    if mode == "bandit" and not bandit_arms:
        _fail(path, where, "mode 'bandit' requires a non-empty 'bandit_arms'")

    ramp_step = _check_fraction_int(path, where, "ramp_step", raw.get("ramp_step", DEFAULT_RAMP_STEP))
    ramp_interval_s = _check_positive_number(
        path, where, "ramp_interval_s", raw.get("ramp_interval_s", DEFAULT_RAMP_INTERVAL_S)
    )
    ramp_min_ratio = _check_fraction_int(
        path, where, "ramp_min_ratio", raw.get("ramp_min_ratio", DEFAULT_RAMP_MIN_RATIO)
    )
    ramp_max_ratio = _check_fraction_int(
        path, where, "ramp_max_ratio", raw.get("ramp_max_ratio", DEFAULT_RAMP_MAX_RATIO)
    )
    if ramp_min_ratio > ramp_max_ratio:
        _fail(path, where, f"'ramp_min_ratio' ({ramp_min_ratio}) exceeds 'ramp_max_ratio' ({ramp_max_ratio})")

    audit_min_fraction = _check_fraction_int(
        path, where, "audit_min_fraction", raw.get("audit_min_fraction", DEFAULT_AUDIT_MIN_FRACTION)
    )
    audit_max_fraction = _check_fraction_int(
        path, where, "audit_max_fraction", raw.get("audit_max_fraction", DEFAULT_AUDIT_MAX_FRACTION)
    )
    if audit_min_fraction > audit_max_fraction:
        msg = f"'audit_min_fraction' ({audit_min_fraction}) exceeds 'audit_max_fraction' ({audit_max_fraction})"
        _fail(path, where, msg)

    rollback_min_match_rate = _check_admission_fraction(
        path, where, "rollback_min_match_rate", raw.get("rollback_min_match_rate", DEFAULT_ROLLBACK_MIN_MATCH_RATE)
    )
    rollback_max_drift = raw.get("rollback_max_drift", DEFAULT_ROLLBACK_MAX_DRIFT)
    bad_rollback_max_drift = (
        isinstance(rollback_max_drift, bool)
        or not isinstance(rollback_max_drift, (int, float))
        or not (0.0 < rollback_max_drift <= 2.0)
    )
    if bad_rollback_max_drift:
        _fail(path, where, f"'rollback_max_drift' must be a number in (0, 2], got {rollback_max_drift!r}")
    rollback_cooldown_s = _check_positive_number(
        path, where, "rollback_cooldown_s", raw.get("rollback_cooldown_s", DEFAULT_ROLLBACK_COOLDOWN_S)
    )

    return RouterConfig(
        mode=mode,
        student_path=student_path,
        ratio=ratio,
        tau=tau,
        ramp_step=ramp_step,
        ramp_interval_s=ramp_interval_s,
        ramp_min_ratio=ramp_min_ratio,
        ramp_max_ratio=ramp_max_ratio,
        audit_min_fraction=audit_min_fraction,
        audit_max_fraction=audit_max_fraction,
        rollback_min_match_rate=rollback_min_match_rate,
        rollback_max_drift=float(rollback_max_drift),
        rollback_cooldown_s=rollback_cooldown_s,
        bandit_arms=bandit_arms,
    )


def _check_entry(path: Path, index: int, raw: dict) -> TagSpec:
    where = f"entry {index}"
    if not isinstance(raw, dict):
        _fail(path, where, f"must be a JSON object, got {type(raw).__name__}")
    missing = REQUIRED_FIELDS - raw.keys()
    extra = raw.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if missing:
        _fail(path, where, f"missing field(s): {sorted(missing)}")
    if extra:
        allowed = sorted(REQUIRED_FIELDS | OPTIONAL_FIELDS)
        _fail(path, where, f"unknown field(s): {sorted(extra)} -- the allowed fields are {allowed}")

    task = raw["task"]
    if not isinstance(task, str) or not task.strip():
        _fail(path, where, f"'task' must be a non-empty string, got {task!r}")

    question = raw["question"]
    if not isinstance(question, str) or not question.strip():
        _fail(path, where, f"'question' must be a non-empty string, got {question!r}")

    options = raw["options"]
    if not isinstance(options, list) or len(options) < 2:
        _fail(path, where, f"'options' must be a list of at least two strings, got {options!r}")
    if not all(isinstance(o, str) and o.strip() for o in options):
        _fail(path, where, f"'options' must all be non-empty strings, got {options!r}")
    if len(set(options)) != len(options):
        _fail(path, where, f"'options' repeats an entry: {options!r}")

    kind = raw["kind"]
    if not isinstance(kind, str) or not kind.strip():
        _fail(path, where, f"'kind' must be a non-empty string, got {kind!r}")

    normalize = raw["normalize"]
    if normalize not in NORMALIZE_RULES:
        _fail(path, where, f"'normalize' must be one of {sorted(NORMALIZE_RULES)}, got {normalize!r}")

    retain_days = raw["retain_days"]
    if isinstance(retain_days, bool) or not isinstance(retain_days, int) or retain_days <= 0:
        _fail(path, where, f"'retain_days' must be a positive integer, got {retain_days!r}")

    keep_hidden = raw["keep_hidden"]
    if not isinstance(keep_hidden, bool):
        _fail(path, where, f"'keep_hidden' must be a boolean, got {keep_hidden!r}")

    eval_set = raw["eval_set"]
    if not isinstance(eval_set, str) or not eval_set.strip():
        _fail(path, where, f"'eval_set' must be a non-empty string (a path), got {eval_set!r}")

    features: str | None = None
    if "features" in raw:
        features = raw["features"]
        if not isinstance(features, str) or not features.strip():
            _fail(path, where, f"'features' must be a non-empty string (a registered builder's name), got {features!r}")

    min_match_rate = DEFAULT_MIN_MATCH_RATE
    if "min_match_rate" in raw:
        min_match_rate = _check_admission_fraction(path, where, "min_match_rate", raw["min_match_rate"])

    min_accuracy_ratio = DEFAULT_MIN_ACCURACY_RATIO
    if "min_accuracy_ratio" in raw:
        min_accuracy_ratio = _check_admission_fraction(path, where, "min_accuracy_ratio", raw["min_accuracy_ratio"])

    retain_context = "full"
    if "retain_context" in raw:
        retain_context = raw["retain_context"]
        if retain_context not in RETAIN_CONTEXT_RULES:
            _fail(
                path, where, f"'retain_context' must be one of {sorted(RETAIN_CONTEXT_RULES)}, got {retain_context!r}"
            )

    router = None
    if "router" in raw:
        router = _check_router(path, where, raw["router"])

    return TagSpec(
        task=task,
        question=question,
        options=tuple(options),
        kind=kind,
        normalize=normalize,
        retain_days=retain_days,
        keep_hidden=keep_hidden,
        eval_set=eval_set,
        features=features,
        min_match_rate=min_match_rate,
        min_accuracy_ratio=min_accuracy_ratio,
        retain_context=retain_context,
        router=router,
    )


def _normalize_for_dup_check(text: str, rule: str) -> str:
    # Local, tiny reimplementation rather than importing tagging.normalize: spec.py validates the file in
    # isolation (no request involved yet), and tagging.py's own docstring is about matching a *request*, which
    # would read oddly imported here just to normalize a spec's own question against itself.
    if rule == "strip+lower":
        return text.strip().lower()
    if rule == "strip":
        return text.strip()
    if rule == "lower":
        return text.lower()
    return text


def _check_no_ambiguity(path: Path, tags: Iterable[TagSpec]) -> None:
    tags = list(tags)
    seen_tasks: dict[str, int] = {}
    for i, t in enumerate(tags):
        if t.task in seen_tasks:
            _fail(path, f"entry {i}", f"task {t.task!r} is already used by entry {seen_tasks[t.task]}")
        seen_tasks[t.task] = i

    seen_shapes: dict[tuple[str, frozenset, str], int] = {}
    for i, t in enumerate(tags):
        shape = (_normalize_for_dup_check(t.question, t.normalize), frozenset(t.options), t.kind)
        if shape in seen_shapes:
            other = seen_shapes[shape]
            _fail(
                path,
                f"entry {i}",
                f"has the same (question, option set, kind) as entry {other} ({tags[other].task!r}) once "
                f"normalized -- a request matching this shape could not tell the two tags apart",
            )
        seen_shapes[shape] = i


def load_spec(path: str | Path) -> LearnSpec:
    """Read, validate in full, and version a spec file. Raises ``SpecError`` on the first problem found -- this is
    the "all fields are checked, and a bad one stops startup" half of the module docstring's contract.
    """
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as e:
        raise SpecError(f"{path}: could not read the spec file: {e}") from e
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as e:
        raise SpecError(f"{path}: not valid JSON: {e}") from e
    if not isinstance(raw, list):
        raise SpecError(f"{path}: the spec must be a JSON list of tag entries, got {type(raw).__name__}")

    tags = tuple(_check_entry(path, i, entry) for i, entry in enumerate(raw))
    _check_no_ambiguity(path, tags)
    version = hashlib.sha256(raw_bytes).hexdigest()[:16]
    return LearnSpec(path=path, version=version, tags=tags)
