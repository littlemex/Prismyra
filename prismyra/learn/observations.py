"""Graded attempts in the open observation record format, written by Prismyra for any reader that consumes it.

Two producers, sharing one row writer:

* ``score`` answers a declared item set with a student artifact that has never served (or is not serving now), so a
  candidate can be measured before it is deployed. The experiment scripts this replaces scored candidates by hand.
* ``export`` turns a running server's logged experience (and any feedback joined to it by ``request_id``) into rows,
  one per logged request, attributed to the arm the request was **assigned** to (the router, under a stable name the
  operator gives), with the components that actually answered in ``path``.

The record format is the one tierbook publishes (format 3: ``item``, ``task_type``, ``arm``, ``outcome``, ``usage``,
and optional ``arm_version``, ``path``, ``score``, ``latency_s``, ``observed_at``, ``label_source``, ``grader``,
``reason``). This module writes plain JSON and imports nothing from tierbook; a reader validates with its own tool
(``tierbook check-observations``).

What this module does not decide: whether an answer is right. A row is ``accepted``/``rejected`` only against a label
the operator supplied (an items file's ``label``, or a feedback record's ``label``). A scalar feedback ``reward`` becomes
a verdict only through an injected ``--verdict-module MODULE:FUNC`` (``FUNC(reward, experience) -> "accepted" |
"rejected" | None``), whose name is recorded as the grader. With no label and no verdict module the row is
``unobserved`` with the reason saying which was missing -- never guessed.

Item ids: an items file may name ``item`` per row. Otherwise the id is ``sha256:`` plus the hex SHA-256 of the context's
UTF-8 bytes, the same content digest a set manifest uses, so a row and a manifest entry for the same text agree.

    python -m prismyra.learn.observations score --artifact A --items items.jsonl --task-type T --arm NAME --out obs.jsonl
    python -m prismyra.learn.observations export --log-root DIR --task TAG --task-type T --arm NAME \\
        [--student A] [--feedback-root DIR] [--labels items.jsonl] [--verdict-module M:F] [--stats stats.json] \\
        --out obs.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

FORMAT = 3
OUTCOMES = ("accepted", "rejected", "unobserved")


class ObservationError(Exception):
    """An input this module cannot turn into rows without guessing."""


def item_id(context: str) -> str:
    """The content digest of a context: ``sha256:`` and the hex digest of its UTF-8 bytes."""
    return "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def row(*, item: str, task_type: str, arm: str, outcome: str, arm_version: str = "", path: Iterable[str] = (),
        score: float | None = None, latency_s: float | None = None, observed_at: str = "", label_source: str = "",
        grader: str = "", reason: str = "", run: str = "") -> dict:
    """One record. Refuses what a reader would refuse, so a bad row fails here rather than in someone else's tool."""
    if outcome not in OUTCOMES:
        raise ObservationError(f"outcome must be one of {OUTCOMES}, not {outcome!r}")
    if outcome == "unobserved" and not reason:
        raise ObservationError(f"item {item!r}: an unobserved row needs a reason")
    for name, value in (("item", item), ("task_type", task_type), ("arm", arm)):
        if not value:
            raise ObservationError(f"a row needs a non-empty {name}")
    path = [str(p) for p in path]
    out: dict = {"format": FORMAT if (arm_version or path) else 2 if score is not None else 1, "item": item,
                 "task_type": task_type, "arm": arm, "outcome": outcome, "usage": {}}
    for key, value in (("arm_version", arm_version), ("score", score), ("latency_s", latency_s),
                       ("observed_at", observed_at), ("label_source", label_source), ("grader", grader),
                       ("reason", reason), ("run", run)):
        if value not in (None, ""):
            out[key] = value
    if path:
        out["path"] = path
    return out


def _top(probabilities: dict[str, float]) -> tuple[str, float]:
    label = max(sorted(probabilities), key=lambda k: probabilities[k])
    return label, float(probabilities[label])


def _items(path: str | Path) -> list[dict]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw.get("context"), str):
            raise ObservationError(f"{path}:{n}: an item needs a 'context' string")
        raw.setdefault("item", item_id(raw["context"]))
        out.append(raw)
    return out


def artifact_version(artifact) -> str:
    """A student artifact's version as the open record carries it: the same content-addressed name Prismyra logs in
    ``served_by``, plus the feature builder's name when the artifact uses a registered one (the builder changes what a
    given model text computes)."""
    from .students import student_version

    v = f"student:{student_version(artifact)}"
    return f"{v}+features:{artifact.features}" if artifact.features else v


def score_items(artifact, items: list[dict], *, task_type: str, arm: str, label_source: str,
                predict: Callable[[str, str], dict[str, float]] | None = None) -> list[dict]:
    """Answer every item with ``artifact`` and grade it against the item's ``label`` (absent: ``unobserved``)."""
    predict = predict or artifact.predict_proba
    version = artifact_version(artifact)
    out = []
    for it in items:
        label, top = _top(predict(it["context"], it.get("question", "")))
        gold = it.get("label")
        common = dict(item=str(it["item"]), task_type=task_type, arm=arm, arm_version=version, score=top,
                      label_source=label_source)
        if gold is None:
            out.append(row(outcome="unobserved", reason="no label", **common))
        else:
            out.append(row(outcome="accepted" if label == gold else "rejected", grader="label-exact", **common))
    return out


def _verdict_fn(spec: str | None):
    if not spec:
        return None
    mod, sep, attr = spec.partition(":")
    if not sep:
        raise ObservationError(f"--verdict-module takes MODULE:FUNC, not {spec!r}")
    return getattr(importlib.import_module(mod), attr)


def export_experience(experiences: Iterable, *, task_type: str, arm: str, student=None,
                      feedback: dict | None = None, labels: dict[str, str] | None = None,
                      verdict: Callable | None = None, verdict_name: str = "") -> list[dict]:
    """Rows for a server's logged requests, attributed to the assigned ``arm``.

    ``arm_version`` names what was serving: the backbone and package, the routing snapshot (``route_params_version``)
    and, when ``student`` is given, that artifact's version. A row answered by a different student artifact is refused
    rather than relabelled. ``path`` is the components that answered: ``["student"]``, ``["prismyra"]``, or
    ``["student", "prismyra"]`` when the student ran and the request fell back.
    """
    feedback = feedback or {}
    labels = labels or {}
    want_student = artifact_version(student) if student is not None else None
    want_short = want_student.split("+", 1)[0] if want_student else None
    out = []
    for exp in experiences:
        item = labels_item = item_id(exp.context) if exp.context else ""
        if not item:
            raise ObservationError(f"request {exp.request_id!r} has no retained context, so it has no item id; "
                                   "keep the context ('retain_context') for a tag whose rows are exported")
        answered = exp.answered_by or ""
        if answered.startswith("student"):
            if want_short is not None and answered != want_short:
                raise ObservationError(f"request {exp.request_id!r} was answered by {answered!r}, not the declared "
                                       f"student {want_short!r}; export each artifact's rows separately")
            path = ["student"]
        elif exp.comparison_probabilities is not None and exp.route_mode == "confidence":
            path = ["student", "prismyra"]
        else:
            path = ["prismyra"]
        parts = [f"backbone:{exp.versions.get('backbone', '')}", f"package:{exp.versions.get('package', '')}",
                 f"route:{exp.route_params_version or 'none'}"]
        if want_student:
            parts.append(want_student)
        label, top = _top(dict(exp.probabilities))
        common = dict(item=labels_item, task_type=task_type, arm=arm, arm_version="|".join(parts), path=path,
                      score=top, observed_at=_iso(exp.ts))
        fb = feedback.get(exp.request_id) if exp.request_id else None
        gold = labels.get(item)
        if fb is not None and fb.label is not None:
            out.append(row(outcome="accepted" if label == fb.label else "rejected", grader="label-exact",
                           label_source="feedback", **common))
        elif gold is not None:
            out.append(row(outcome="accepted" if label == gold else "rejected", grader="label-exact",
                           label_source="dataset", **common))
        elif fb is not None and fb.reward is not None and verdict is not None:
            v = verdict(fb.reward, exp)
            if v is None:
                out.append(row(outcome="unobserved", reason="verdict module returned no verdict",
                               label_source="feedback", **common))
            else:
                out.append(row(outcome=v, grader=verdict_name, label_source="feedback", **common))
        else:
            why = ("reward without a verdict module" if fb is not None and fb.reward is not None
                   else "no label")
            out.append(row(outcome="unobserved", reason=why, **common))
    return out


def _write(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m prismyra.learn.observations", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="answer a declared item set with a student artifact")
    s.add_argument("--artifact", required=True)
    s.add_argument("--items", required=True, help="JSONL of {item?, context, question?, label?}")
    s.add_argument("--task-type", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--label-source", default="dataset")
    s.add_argument("--features-module", action="append", default=[])
    s.add_argument("--out", required=True)
    e = sub.add_parser("export", help="turn a server's logged experience into rows for the assigned arm")
    e.add_argument("--log-root", required=True)
    e.add_argument("--task", required=True, help="the learn-spec tag whose log is read")
    e.add_argument("--task-type", required=True)
    e.add_argument("--arm", required=True, help="the assigned arm's stable name, e.g. the router's")
    e.add_argument("--student", default=None, help="the student artifact the router was serving")
    e.add_argument("--feedback-root", default=None)
    e.add_argument("--labels", default=None, help="JSONL of {item?, context, label}: labels joined by content")
    e.add_argument("--verdict-module", default=None, help="MODULE:FUNC turning a scalar reward into a verdict")
    e.add_argument("--stats", default=None, help="the server's GET /stats JSON, for the manifest's logged/dropped")
    e.add_argument("--features-module", action="append", default=[])
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from .features import import_feature_modules
    from .fit import StudentArtifact

    import_feature_modules(args.features_module)
    try:
        if args.cmd == "score":
            art = StudentArtifact.from_json(json.loads(Path(args.artifact).read_text()))
            rows = score_items(art, _items(args.items), task_type=args.task_type, arm=args.arm,
                               label_source=args.label_source)
            _write(rows, args.out)
            print(f"wrote {len(rows)} rows to {args.out}")
            return 0
        from .experience import LocalExperienceSource, load_feedback

        student = (StudentArtifact.from_json(json.loads(Path(args.student).read_text()))
                   if args.student else None)
        labels = {it["item"]: it["label"] for it in _items(args.labels) if "label" in it} if args.labels else {}
        rows = export_experience(LocalExperienceSource(args.log_root).iter_experiences(args.task),
                                 task_type=args.task_type, arm=args.arm, student=student,
                                 feedback=load_feedback(args.feedback_root) if args.feedback_root else {},
                                 labels=labels, verdict=_verdict_fn(args.verdict_module),
                                 verdict_name=args.verdict_module or "")
        _write(rows, args.out)
        stats = json.loads(Path(args.stats).read_text()) if args.stats else None
        manifest = {"rows": len(rows), "task": args.task, "arm": args.arm,
                    "logged": (stats or {}).get("logged"), "dropped": (stats or {}).get("dropped"),
                    "stats_source": args.stats or None}
        Path(f"{args.out}.manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        tail = ("" if stats else "; logged/dropped unknown (no --stats), so a gap in the log cannot be ruled out")
        print(f"wrote {len(rows)} rows to {args.out}{tail}")
        return 0
    except (ObservationError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
