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

#: The exact field set an entry must have -- no more, no fewer. Extra fields are refused rather than ignored: a
#: silently-ignored field is how a typo (``"retain_day"``) ships for weeks before anyone notices retention was
#: never set.
REQUIRED_FIELDS = frozenset(
    {"task", "question", "options", "kind", "normalize", "retain_days", "keep_hidden", "eval_set"}
)

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


def _fail(path: Path, where: str, message: str) -> None:
    raise SpecError(f"{path}: {where}: {message}")


def _check_entry(path: Path, index: int, raw: dict) -> TagSpec:
    where = f"entry {index}"
    if not isinstance(raw, dict):
        _fail(path, where, f"must be a JSON object, got {type(raw).__name__}")
    missing = REQUIRED_FIELDS - raw.keys()
    extra = raw.keys() - REQUIRED_FIELDS
    if missing:
        _fail(path, where, f"missing field(s): {sorted(missing)}")
    if extra:
        _fail(path, where, f"unknown field(s): {sorted(extra)} -- the allowed fields are {sorted(REQUIRED_FIELDS)}")

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

    return TagSpec(
        task=task,
        question=question,
        options=tuple(options),
        kind=kind,
        normalize=normalize,
        retain_days=retain_days,
        keep_hidden=keep_hidden,
        eval_set=eval_set,
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
