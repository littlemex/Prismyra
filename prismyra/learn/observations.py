"""Graded attempts in the open observation record format, written by Prismyra for any reader that consumes it.

Two producers, sharing one row writer:

* ``score`` answers a declared item set with a student artifact that is not serving, so a candidate can be measured
  before it is deployed.
* ``export`` turns a running server's logged requests (and feedback joined to them by ``request_id``) into rows, one per
  logged request, under the arm name the operator declares for the router, with the components that answered in
  ``path``.

The record format is the one tierbook publishes (format 3). This module writes plain JSON and imports nothing from
tierbook; a reader validates with its own tool (``tierbook check-observations``).

**Item ids.** An item's content is ``[context, question]`` as compact JSON in UTF-8; its digest is the hex SHA-256 of
that. Unless an items file names an ``item``, the id is ``sha256:`` plus the digest. Labels are always joined by
content, so a named items file and a server's log meet on the same item. Two items with the same content and
different labels are refused.

**Grading.** A row is ``accepted``/``rejected`` only against a label the operator supplied (an items file's ``label``,
or a feedback record's ``label``), and only when that label is one of the item's options; otherwise the input is
refused. A feedback label and a labels-file label that disagree make the row ``unobserved``, counted in the manifest.
A scalar feedback ``reward`` becomes a verdict only through an injected ``--verdict-module MODULE:FUNC``
(``FUNC(reward, experience) -> "accepted" | "rejected" | None``), named as the grader. Tied or non-finite top
probabilities make the row ``unobserved``: the writer does not pick an answer the server did not clearly give.

**Versions.** ``score`` stamps the artifact's content-addressed name (plus its feature builder's name, which names the
builder, not its code). ``export`` stamps backbone, package, spec, routing snapshot and the declared ``--student``
artifact; a student-answered row from any other artifact is refused, and a log with student-answered rows and no
``--student`` is refused. Rows the backbone answered cannot show which student was loaded; ``--since``/``--until``
separate snapshots by time.

**Not carried.** ``usage`` is empty: the server does not log tokens, so a self-hosted arm is priced from a declared
per-attempt price, not from these rows.

    python -m prismyra.learn.observations score --artifact A --items items.jsonl --task-type T --arm NAME \\
        --label-source SRC --out obs.jsonl
    python -m prismyra.learn.observations export --log-root DIR --task TAG --task-type T --arm NAME \\
        --student-arm S --fallback-arm F [--student A] [--feedback-root DIR] [--labels items.jsonl \\
        --label-source SRC] [--verdict-module M:F] [--stats stats.json] [--since TS] [--until TS] --out obs.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

#: The record format these writers emit when a row carries `arm_version` or `path` (every row they write does).
RECORD_FORMAT = 3
OUTCOMES = ("accepted", "rejected", "unobserved")
#: What the grading in this module is: an exact match of the answer against an operator-supplied label.
LABEL_EXACT = "label-exact"


class ObservationError(Exception):
    """An input this module cannot turn into rows without guessing."""


def content_digest(context: str, question: str = "") -> str:
    """The hex SHA-256 of ``[context, question]`` as compact JSON in UTF-8 (lone surrogates passed through)."""
    payload = json.dumps([context, question], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def item_id(context: str, question: str = "") -> str:
    return "sha256:" + content_digest(context, question)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def row(*, item: str, task_type: str, arm: str, outcome: str, run: str, arm_version: str, path: Iterable[str] = (),
        score: float | None = None, observed_at: str = "", label_source: str = "", grader: str = "",
        reason: str = "") -> dict:
    """One record, refused here when the reader would refuse it."""
    if outcome not in OUTCOMES:
        raise ObservationError(f"outcome must be one of {OUTCOMES}, not {outcome!r}")
    if outcome == "unobserved" and not reason:
        raise ObservationError(f"item {item!r}: an unobserved row needs a reason")
    for name, value in (("item", item), ("task_type", task_type), ("arm", arm), ("arm_version", arm_version)):
        if not isinstance(value, str) or not value:
            raise ObservationError(f"a row needs a non-empty {name}")
    path = list(path)
    if not all(isinstance(p, str) and p for p in path):
        raise ObservationError(f"path entries must be non-empty arm names, not {path!r}")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                              or not math.isfinite(score)):
        raise ObservationError(f"score {score!r} must be a finite number")
    out: dict = {"format": RECORD_FORMAT, "item": item, "task_type": task_type, "arm": arm, "outcome": outcome,
                 "usage": {}, "arm_version": arm_version}
    for key, value in (("run", run), ("score", score), ("observed_at", observed_at), ("label_source", label_source),
                       ("grader", grader), ("reason", reason)):
        if value not in (None, ""):
            out[key] = value
    if path:
        out["path"] = path
    return out


def _answer(probabilities: dict) -> tuple[str | None, float | None, str]:
    """The answer the probabilities clearly give, or None with why not."""
    vals = {k: v for k, v in probabilities.items()}
    if not vals:
        return None, None, "no probabilities logged"
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals.values()):
        return None, None, "non-finite probability"
    top = max(vals.values())
    best = [k for k, v in vals.items() if v == top]
    if len(best) > 1:
        return None, float(top), "tied top probabilities"
    return best[0], float(top), ""


def load_items(path: str | Path) -> list[dict]:
    """An items file: ``{item?, context, question?, label?}`` per line. Ids default to the content digest."""
    out, seen_ids, by_content = [], set(), {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("context"), str):
            raise ObservationError(f"{path}:{n}: an item needs a 'context' string")
        q = raw.get("question") or ""
        if not isinstance(q, str):
            raise ObservationError(f"{path}:{n}: 'question' must be a string")
        digest = content_digest(raw["context"], q)
        item = raw.get("item")
        if item is None:
            item = "sha256:" + digest
        if not isinstance(item, str) or not item:
            raise ObservationError(f"{path}:{n}: 'item' must be a non-empty string when given")
        if item in seen_ids:
            raise ObservationError(f"{path}:{n}: item {item!r} appears twice")
        label = raw.get("label")
        if label is not None and not isinstance(label, str):
            raise ObservationError(f"{path}:{n}: 'label' must be a string")
        prev = by_content.get(digest)
        if prev is not None and prev.get("label") != label:
            raise ObservationError(f"{path}:{n}: the same content as item {prev['item']!r} with a different label")
        seen_ids.add(item)
        entry = {"item": item, "context": raw["context"], "question": q, "label": label, "digest": digest}
        by_content.setdefault(digest, entry)
        out.append(entry)
    return out


def _check_label(label: str, options) -> None:
    if label not in options:
        raise ObservationError(f"label {label!r} is not one of the options {list(options)}; a label the answer can "
                               "never equal would grade every row rejected")


def score_items(artifact, items: list[dict], *, task_type: str, arm: str, label_source: str, run: str = "score",
                predict: Callable[[str, str], dict[str, float]] | None = None) -> list[dict]:
    """Answer every item with ``artifact`` and grade it against the item's label."""
    from .students import artifact_version

    predict = predict or artifact.predict_proba
    version = artifact_version(artifact)
    out = []
    for it in items:
        probs = predict(it["context"], it["question"])
        answer, top, why = _answer(probs)
        common = dict(item=it["item"], task_type=task_type, arm=arm, run=run, arm_version=version, score=top)
        if it["label"] is None:
            out.append(row(outcome="unobserved", reason="no label", **common))
            continue
        _check_label(it["label"], probs)
        if answer is None:
            out.append(row(outcome="unobserved", reason=why, label_source=label_source, **common))
        else:
            out.append(row(outcome="accepted" if answer == it["label"] else "rejected", grader=LABEL_EXACT,
                           label_source=label_source, **common))
    return out


def _verdict_fn(spec: str | None):
    if not spec:
        return None
    mod, sep, attr = spec.partition(":")
    if not sep or not mod or not attr:
        raise ObservationError(f"--verdict-module takes MODULE:FUNC, not {spec!r}")
    try:
        return getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as exc:
        raise ObservationError(f"--verdict-module {spec!r}: {exc}") from None


def export_experience(experiences: Iterable, *, task_type: str, arm: str, student_arm: str, fallback_arm: str,
                      student=None, feedback: dict | None = None, labels: list[dict] | None = None,
                      label_source: str = "", verdict: Callable | None = None, verdict_name: str = "",
                      since: float | None = None, until: float | None = None,
                      per_component: bool = False, run: str | None = None) -> tuple[list[dict], dict]:
    """Rows for a server's logged requests under the declared router ``arm``, and the counts behind them.

    With ``per_component`` the router arm is not written. Each request instead gives one row per component whose
    answer the log holds, under that component's own name and version: the student's (the artifact's
    `artifact_version`) wherever its probabilities were logged, and the fallback's (``backbone|package|spec``)
    wherever it answered or was audited. These are the versions an offline score of the same components carries,
    so a certificate over a cascade composed from parts can be checked against served traffic.

    ``run`` replaces each row's run (the request id by default) with one name for the whole export. A reader that
    composes a cascade from its parts pairs attempts by item and run, so rows meant to pair with an offline
    ``score`` of the same items must carry that score's run."""
    if verdict is not None and not verdict_name:
        raise ObservationError("a verdict function needs a name, recorded as the grader")
    if labels and not label_source:
        raise ObservationError("labels need a label_source saying where they came from")
    feedback = feedback or {}
    by_content = {it["digest"]: it for it in (labels or [])}
    declared = student_version_ = None
    if student is not None:
        from .students import artifact_version, student_version
        declared = f"student:{student_version(student)}"
        student_version_ = artifact_version(student)
    counts: Counter = Counter()
    joined: set[str] = set()
    out = []
    for exp in experiences:
        if (since is not None and exp.ts < since) or (until is not None and exp.ts >= until):
            counts["outside_window"] += 1
            continue
        if not exp.context:
            counts["skipped: context not retained"] += 1
            continue
        answered = exp.answered_by or ""
        if answered.startswith("student:"):
            if declared is None:
                raise ObservationError("the log has student-answered rows and no --student was declared; name the "
                                       "artifact the router was serving")
            if answered != declared:
                raise ObservationError(f"request {exp.request_id!r} was answered by {answered!r}, not the declared "
                                       f"student {declared!r}; use --since/--until to export one snapshot at a time")
            path = [student_arm]
        elif exp.comparison_probabilities is not None and exp.route_mode == "confidence":
            path = [student_arm, fallback_arm]
        else:
            path = [fallback_arm]
        missing = [k for k in ("backbone", "package", "spec") if not exp.versions.get(k)]
        if missing:
            counts[f"skipped: versions missing {missing}"] += 1
            continue
        serving = [f"backbone:{exp.versions['backbone']}", f"package:{exp.versions['package']}",
                   f"spec:{exp.versions['spec']}"]
        parts = serving + [f"route:{exp.route_params_version or 'none'}"]
        if student_version_:
            parts.append(student_version_)
        digest = content_digest(exp.context, exp.question or "")
        lab = by_content.get(digest)
        item = lab["item"] if lab is not None else "sha256:" + digest
        run_name = run or exp.request_id or f"ts:{exp.ts!r}"
        if per_component:
            by_student = answered.startswith("student:")
            student_probs = exp.probabilities if by_student else exp.comparison_probabilities
            fallback_probs = exp.comparison_probabilities if by_student else exp.probabilities
            if student_probs is not None and not by_student and student_version_ is None:
                raise ObservationError("the log has the student's probabilities and no --student was declared")
            components = [(student_arm, student_version_, student_probs), (fallback_arm, "|".join(serving),
                                                                             fallback_probs)]
        else:
            components = [(arm, "|".join(parts), exp.probabilities)]
        for c_arm, c_version, c_probs in components:
            if c_probs is None:
                continue
            _grade(out, counts, joined, exp, lab, feedback, label_source, verdict, verdict_name,
                   dict(item=item, task_type=task_type, arm=c_arm, arm_version=c_version,
                        **({} if per_component else {"path": path}), observed_at=_iso(exp.ts), run=run_name),
                   dict(c_probs))
    counts["feedback not joined to any exported row"] = len(set(feedback) - joined)
    return out, dict(sorted(counts.items()))


def _grade(out, counts, joined, exp, lab, feedback, label_source, verdict, verdict_name, common, probs) -> None:
    """One row for one answer to one logged request, graded the same way whichever component gave it."""
    answer, top, why = _answer(probs)
    common = {**common, "score": top}
    fb = feedback.get(exp.request_id) if exp.request_id else None
    if fb is not None:
        joined.add(exp.request_id)
    options = exp.options or tuple(exp.probabilities or ())
    fb_label = fb.label if fb is not None else None
    gold = lab["label"] if lab is not None else None
    for g in (fb_label, gold):
        if g is not None:
            _check_label(g, options)
    if fb_label is not None and gold is not None and fb_label != gold:
        counts["unobserved: feedback and labels disagree"] += 1
        out.append(row(outcome="unobserved", reason="feedback and labels disagree", **common))
        return
    truth, src = (fb_label, "feedback") if fb_label is not None else (gold, label_source)
    if truth is not None:
        if answer is None:
            out.append(row(outcome="unobserved", reason=why, label_source=src, **common))
        else:
            out.append(row(outcome="accepted" if answer == truth else "rejected", grader=LABEL_EXACT,
                           label_source=src, **common))
    elif fb is not None and fb.reward is not None and verdict is not None:
        v = verdict(fb.reward, exp)
        if v is None:
            out.append(row(outcome="unobserved", reason="verdict module returned no verdict",
                           label_source="feedback-reward", **common))
        else:
            out.append(row(outcome=v, grader=verdict_name, label_source="feedback-reward", **common))
    else:
        out.append(row(outcome="unobserved",
                       reason=("reward without a verdict module" if fb is not None and fb.reward is not None
                               else "no label"), **common))


def _write_atomic(path: str | Path, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _rows_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m prismyra.learn.observations",
                                 description="Graded attempts in the open observation record format.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="answer a declared item set with a student artifact")
    s.add_argument("--artifact", required=True)
    s.add_argument("--items", required=True, help="JSONL of {item?, context, question?, label?}")
    s.add_argument("--task-type", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--label-source", required=True, help="where the items' labels came from")
    s.add_argument("--run", default="score")
    s.add_argument("--features-module", action="append", default=[])
    s.add_argument("--out", required=True)
    e = sub.add_parser("export", help="turn a server's logged requests into rows for the router arm")
    e.add_argument("--log-root", required=True)
    e.add_argument("--task", required=True, help="the learn-spec tag whose log is read")
    e.add_argument("--task-type", required=True)
    e.add_argument("--arm", required=True, help="the router's stable arm name")
    e.add_argument("--student-arm", required=True, help="the name the student goes by in `path`")
    e.add_argument("--fallback-arm", required=True, help="the name the fallback goes by in `path`")
    e.add_argument("--student", default=None, help="the student artifact the router was serving")
    e.add_argument("--feedback-root", default=None)
    e.add_argument("--labels", default=None, help="JSONL of {item?, context, question?, label}, joined by content")
    e.add_argument("--label-source", default="", help="where --labels came from (required with --labels)")
    e.add_argument("--verdict-module", default=None, help="MODULE:FUNC turning a scalar reward into a verdict")
    e.add_argument("--stats", default=None, help="the server's GET /stats JSON, with top-level logged and dropped")
    e.add_argument("--since", type=float, default=None, help="epoch seconds, inclusive")
    e.add_argument("--until", type=float, default=None, help="epoch seconds, exclusive")
    e.add_argument("--per-component", action="store_true",
                   help="write one row per component answer (student, fallback) under its own version instead of "
                        "one row per request for the router arm")
    e.add_argument("--run", default=None,
                   help="one run name for every row instead of each request id, to pair with an offline score's run")
    e.add_argument("--features-module", action="append", default=[])
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        from .features import import_feature_modules
        from .fit import StudentArtifact

        import_feature_modules(args.features_module)
        if args.cmd == "score":
            art = StudentArtifact.from_json(json.loads(Path(args.artifact).read_text()))
            rows = score_items(art, load_items(args.items), task_type=args.task_type, arm=args.arm,
                               label_source=args.label_source, run=args.run)
            _write_atomic(args.out, _rows_text(rows))
            print(f"wrote {len(rows)} rows to {args.out}")
            return 0
        from .experience import LocalExperienceSource, load_feedback

        stats = None
        if args.stats:
            stats = json.loads(Path(args.stats).read_text())
            if not isinstance(stats, dict) or "logged" not in stats or "dropped" not in stats:
                raise ObservationError(f"--stats {args.stats} has no top-level 'logged' and 'dropped'")
        student = (StudentArtifact.from_json(json.loads(Path(args.student).read_text()))
                   if args.student else None)
        rows, counts = export_experience(
            LocalExperienceSource(args.log_root).iter_experiences(args.task), task_type=args.task_type,
            arm=args.arm, student_arm=args.student_arm, fallback_arm=args.fallback_arm, student=student,
            feedback=load_feedback(args.feedback_root) if args.feedback_root else {},
            labels=load_items(args.labels) if args.labels else None, label_source=args.label_source,
            verdict=_verdict_fn(args.verdict_module), verdict_name=args.verdict_module or "",
            since=args.since, until=args.until, per_component=args.per_component, run=args.run)
        manifest = {"rows": len(rows), "task": args.task, "arm": args.arm, "counts": counts,
                    "logged": stats["logged"] if stats else "unknown",
                    "dropped": stats["dropped"] if stats else "unknown", "stats_source": args.stats or None}
        _write_atomic(args.out, _rows_text(rows))
        _write_atomic(f"{args.out}.manifest.json", json.dumps(manifest, indent=1, sort_keys=True))
        warn = "" if stats else "; logged/dropped unknown (no --stats), so a gap in the log cannot be ruled out"
        if stats and stats["dropped"]:
            warn = f"; the server dropped {stats['dropped']} rows that are not here"
        print(f"wrote {len(rows)} rows to {args.out}{warn}")
        return 0
    except (ObservationError, OSError, json.JSONDecodeError, UnicodeError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
