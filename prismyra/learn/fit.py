"""Stage 2 of DISTILL-RL-DESIGN-v2.md: a hand-run, CPU-only distillation of one registered tag's logged
experience into a LightGBM student, evaluated offline, saved as data, never wired into serving.

    python -m prismyra.learn.fit --learn-spec <path> --task <name> --out <path> [options]

This module is deliberately not imported by anything else in the package. `prismyra.server` never imports
`prismyra.learn.fit`, and the ``learn`` extra (``pip install "prismyra[learn]"``) that brings in LightGBM is not
part of ``server`` or ``dev`` -- running this is an operator's own, out-of-band decision, on their own schedule,
on their own CPU, with no server anywhere nearby.

**Why a student is not admitted on agreement with Prismyra alone.** DISTILL-RL-REVIEW.md's adversarial pass (P1
item 4) is blunt about why: a student that reproduces Prismyra's own mistakes has a *high* agreement rate by
construction, and a gate built only from that number cannot tell "the student learned the task" apart from "the
student learned the teacher's blind spots". So two separate conditions both have to pass (section 3):

1. On the tag's own ``eval_set`` -- a file of *gold*-labelled items, not teacher-labelled ones -- the student's
   accuracy is at least Prismyra's own accuracy minus ``--accuracy-margin``. There is no live model in this
   process to answer the eval set itself (fit.py runs on a CPU with no GPU anywhere near it, by design); an
   eval item that carries Prismyra's own prediction already (a ``prismyra_probabilities`` field, collected once,
   out of band, the same way the eval set's gold labels were) is scored against the student's. An eval item
   without one still has the student's accuracy scored, but contributes nothing to the "did we beat Prismyra"
   comparison -- see ``_score_eval``.
2. On a held-out slice of the tag's own logged experience -- withheld from training, and de-duplicated by exact
   input text before any split happens, so one popular document cannot inflate how precisely two things that
   are both measuring that same crowd agree with each other -- the one-sided Wilson lower bound of the
   student/Prismyra agreement rate clears ``--min-match-rate``. This is the "a point estimate is not a
   guarantee" discipline the shared memory for this project insists on everywhere else; it is the same reasoning
   here, inside one tag's own distillation.

Both conditions are necessary; neither is sufficient alone, which is the point of asking for both.

**Why ``tau`` is reported, not chosen.** The design (section 3, citing arXiv:2501.09345) wants the answer rate
versus error rate trade-off read directly off held-out data rather than fixed by this script -- a threshold
that trades those two off is a judgement about what an operator's traffic can tolerate, which this script has
no way to know. ``_tau_sweep`` reports the curve; nothing here picks a point on it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .experience import Experience
from .features import HashingTfidf, tokenize
from .spec import TagSpec, load_spec


class FitError(Exception):
    """A tag, a spec, or an eval set could not be made into a trainable, evaluable dataset."""


def _iter_experiences(log_dir: Path, tag: str):
    tag_dir = log_dir / tag
    if not tag_dir.is_dir():
        return
    for path in sorted(tag_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    yield Experience.from_json(json.loads(stripped))


def dedup(experiences: list[Experience]) -> list[Experience]:
    """Drop every experience after the first with an exact-text-identical ``context``. Section 3's own
    reasoning for this: a popular document otherwise counts once per time it was asked about, which inflates
    both the student's apparent accuracy and the agreement rate's confidence interval."""
    seen: set[str] = set()
    out = []
    for e in experiences:
        if e.context in seen:
            continue
        seen.add(e.context)
        out.append(e)
    return out


def split_train_held_out(
    experiences: list[Experience], held_out_fraction: float, seed: int
) -> tuple[list[Experience], list[Experience]]:
    rng = random.Random(seed)
    shuffled = list(experiences)
    rng.shuffle(shuffled)
    n_held = max(1, round(len(shuffled) * held_out_fraction)) if shuffled else 0
    return shuffled[n_held:], shuffled[:n_held]


def argmax(probs) -> str:
    """The option with the largest probability. A small helper rather than ``max(probs, key=probs.get)``
    everywhere: that spelling's `key` is a bound dict method, whose overloaded signature a type checker cannot
    match against `max`'s own -- this gives it one concrete, checkable shape instead of repeating the
    work-around at every call site."""
    return max(probs.items(), key=lambda kv: kv[1])[0]


def wilson_lower_bound(successes: int, n: int, z: float = 1.959963984540054) -> float:
    """The lower edge of a two-sided ~95% (``z=1.96``) Wilson score interval for a binomial proportion. Used,
    not a plain ``successes / n``, because a small held-out set makes a point estimate of agreement
    overconfident in exactly the way the design's acceptance condition (section 3, item 2) is written to guard
    against."""
    if n == 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - adj) / denom


@dataclass(frozen=True)
class EvalItem:
    context: str
    gold: str
    options: tuple[str, ...]
    prismyra_probabilities: dict[str, float] | None


def load_eval_set(path: Path) -> list[EvalItem]:
    """One JSONL file, one item per line: ``{"context", "gold", "options", "prismyra_probabilities"?}``.
    ``prismyra_probabilities`` is optional and, when present, is Prismyra's own answer to this exact item,
    collected once out of band -- see the module docstring for why fit.py cannot produce that figure itself.
    """
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            items.append(
                EvalItem(
                    context=raw["context"],
                    gold=raw["gold"],
                    options=tuple(raw["options"]),
                    prismyra_probabilities=raw.get("prismyra_probabilities"),
                )
            )
    return items


@dataclass(frozen=True)
class StudentArtifact:
    """What ``--out`` holds: one versioned, self-describing JSON file. Not wired into serving by anything in
    this package -- "生徒の成果物は、版付きのファイルとして保存する。配信にはまだ入れない" (section 3).
    """

    tag: str
    spec_version: str
    package_version: str
    backbone: str
    options: tuple[str, ...]
    kind: str
    dims: int
    idf: tuple[float, ...]
    model_text: str
    created_at: float
    admitted: bool
    reasons: tuple[str, ...]
    metrics: dict

    def to_json(self) -> dict:
        return {
            "tag": self.tag,
            "spec_version": self.spec_version,
            "package_version": self.package_version,
            "backbone": self.backbone,
            "options": list(self.options),
            "kind": self.kind,
            "dims": self.dims,
            "idf": list(self.idf),
            "model_text": self.model_text,
            "created_at": self.created_at,
            "admitted": self.admitted,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }

    @staticmethod
    def from_json(raw: dict) -> StudentArtifact:
        return StudentArtifact(
            tag=raw["tag"],
            spec_version=raw["spec_version"],
            package_version=raw["package_version"],
            backbone=raw["backbone"],
            options=tuple(raw["options"]),
            kind=raw["kind"],
            dims=raw["dims"],
            idf=tuple(raw["idf"]),
            model_text=raw["model_text"],
            created_at=raw["created_at"],
            admitted=raw["admitted"],
            reasons=tuple(raw["reasons"]),
            metrics=raw["metrics"],
        )

    def vectorizer(self) -> HashingTfidf:
        return HashingTfidf(dims=self.dims, idf=self.idf)

    def predict_proba(self, context: str) -> dict[str, float]:
        """Load-bearing for a later stage, not for stage 1/2: given this exact artifact, reproduce the
        probabilities it would assign a new input, without touching training again."""
        import lightgbm as lgb

        booster = lgb.Booster(model_str=self.model_text)
        vec = self.vectorizer().transform(tokenize(context))
        (probs,) = booster.predict(np.asarray([vec], dtype=np.float64))
        return dict(zip(self.options, (float(p) for p in probs), strict=True))


def _score_eval(eval_items: list[EvalItem], predict_fn) -> dict:
    student_correct = 0
    prismyra_correct = 0
    prismyra_scored = 0
    for item in eval_items:
        probs = predict_fn(item.context)
        student_choice = argmax(probs)
        student_correct += int(student_choice == item.gold)
        if item.prismyra_probabilities:
            prismyra_scored += 1
            prismyra_choice = argmax(item.prismyra_probabilities)
            prismyra_correct += int(prismyra_choice == item.gold)
    n = len(eval_items) or 1
    out = {"n": len(eval_items), "student_accuracy": student_correct / n}
    if prismyra_scored:
        out["prismyra_accuracy"] = prismyra_correct / prismyra_scored
        out["prismyra_scored"] = prismyra_scored
    return out


def tau_sweep(held_out: list[Experience], predict_fn, taus=None) -> list[dict]:
    """For each candidate ``tau``: the fraction of held-out items the student would answer (its own top
    probability at least ``tau``), and, among those, the fraction that disagree with Prismyra's own answer on
    the same item -- the trade-off curve section 3 wants reported, not collapsed into one number.
    """
    taus = taus if taus is not None else [round(0.05 * i, 2) for i in range(0, 20)]
    rows = []
    for tau in taus:
        answered = 0
        disagreed = 0
        for e in held_out:
            probs = predict_fn(e.context)
            choice, p = max(probs.items(), key=lambda kv: kv[1])
            if p < tau:
                continue
            answered += 1
            teacher_choice = argmax(e.probabilities)
            if choice != teacher_choice:
                disagreed += 1
        n = len(held_out) or 1
        rows.append(
            {
                "tau": tau,
                "answer_rate": answered / n,
                "disagreement_rate": (disagreed / answered) if answered else None,
            }
        )
    return rows


def _soft_rows(experiences: list[Experience], options: tuple[str, ...], vectorizer: HashingTfidf):
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    w_rows: list[float] = []
    for e in experiences:
        vec = vectorizer.transform(tokenize(e.context))
        for i, opt in enumerate(options):
            p = e.probabilities.get(opt, 0.0)
            if p <= 0.0:
                continue
            x_rows.append(vec)
            y_rows.append(i)
            w_rows.append(p)
    return x_rows, y_rows, w_rows


def fit_student(
    tag: TagSpec,
    *,
    train: list[Experience],
    dims: int = 256,
    num_threads: int = 1,
    seed: int = 0,
    num_boost_round: int = 100,
    min_data_in_leaf: int | None = None,
):
    """Train one LightGBM booster on ``train``'s soft labels (section 3: "柔らかいラベルは、クラスの数だけ行を
    複製し、重みを確率にする") -- deterministic given a fixed build, instruction set, thread count and seed
    (LightGBM's own documented contract for ``deterministic=True``, which is why this project's spec picked it
    over CatBoost: ``catboost#1825`` is an open, acknowledged instruction-set-dependence bug this project does
    not want to inherit).

    ``min_data_in_leaf`` defaults to LightGBM's own default (20) when left `None`. A tag with only a few dozen
    logged experiences -- the common case for a newly-registered tag, long before it has enough traffic to
    train on seriously -- cannot form a single split under that default and trains a stump that always
    predicts the majority class; passing a smaller value is how a caller fits something meaningful on a tag
    that is still young, with the obvious cost (section 3's own admission conditions exist precisely to catch
    it) that a stump fit on little data is also the one most likely to fail them.
    """
    import lightgbm as lgb

    corpus = [tokenize(e.context) for e in train]
    vectorizer = HashingTfidf.fit(corpus, dims)
    x_rows, y_rows, w_rows = _soft_rows(train, tag.options, vectorizer)
    if not x_rows:
        raise FitError(f"tag {tag.task!r} has no trainable experience (0 rows after soft-label expansion)")

    dataset = lgb.Dataset(np.asarray(x_rows, dtype=np.float64), label=y_rows, weight=w_rows, free_raw_data=True)
    params = {
        "objective": "multiclass",
        "num_class": len(tag.options),
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": num_threads,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "verbosity": -1,
    }
    if min_data_in_leaf is not None:
        params["min_data_in_leaf"] = min_data_in_leaf
    booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
    return booster, vectorizer


def run(
    *,
    spec_path: str | Path,
    task: str,
    log_dir: str | Path | None,
    package_version: str,
    backbone: str,
    dims: int = 256,
    held_out_fraction: float = 0.2,
    min_match_rate: float = 0.97,
    accuracy_margin: float = 0.0,
    num_threads: int = 1,
    seed: int = 0,
    num_boost_round: int = 100,
    min_data_in_leaf: int | None = None,
) -> StudentArtifact:
    spec = load_spec(spec_path)
    tag = spec.by_task(task)
    if tag is None:
        raise FitError(f"{spec_path}: no entry for task {task!r}")

    root = Path(log_dir) if log_dir is not None else Path(spec_path).parent / "experience"
    all_experiences = dedup(list(_iter_experiences(root, task)))
    train, held_out = split_train_held_out(all_experiences, held_out_fraction, seed)

    booster, vectorizer = fit_student(
        tag,
        train=train,
        dims=dims,
        num_threads=num_threads,
        seed=seed,
        num_boost_round=num_boost_round,
        min_data_in_leaf=min_data_in_leaf,
    )

    def predict_fn(context: str) -> dict[str, float]:
        vec = vectorizer.transform(tokenize(context))
        (probs,) = booster.predict(np.asarray([vec], dtype=np.float64))
        return dict(zip(tag.options, (float(p) for p in probs), strict=True))

    reasons: list[str] = []
    metrics: dict = {
        "train_n": len(train),
        "held_out_n": len(held_out),
        "deduped_n": len(all_experiences),
    }

    eval_path = Path(tag.eval_set)
    if not eval_path.is_absolute():
        eval_path = Path(spec_path).parent / eval_path
    condition_1 = False
    if eval_path.is_file():
        eval_items = load_eval_set(eval_path)
        eval_metrics = _score_eval(eval_items, predict_fn)
        metrics["eval"] = eval_metrics
        if "prismyra_accuracy" in eval_metrics:
            condition_1 = eval_metrics["student_accuracy"] >= eval_metrics["prismyra_accuracy"] - accuracy_margin
            if not condition_1:
                reasons.append(
                    f"eval_set: student accuracy {eval_metrics['student_accuracy']:.4f} is below Prismyra's "
                    f"{eval_metrics['prismyra_accuracy']:.4f} minus margin {accuracy_margin}"
                )
        else:
            reasons.append(f"eval_set {eval_path} has no item with 'prismyra_probabilities'; condition 1 unmet")
    else:
        reasons.append(f"eval_set {eval_path} does not exist; condition 1 unmet")

    agree = 0
    for e in held_out:
        probs = predict_fn(e.context)
        student_choice = argmax(probs)
        teacher_choice = argmax(e.probabilities)
        agree += int(student_choice == teacher_choice)
    lower_bound = wilson_lower_bound(agree, len(held_out))
    metrics["held_out_agreement"] = {
        "n": len(held_out),
        "agree": agree,
        "rate": (agree / len(held_out)) if held_out else 0.0,
        "lower_bound_95": lower_bound,
    }
    condition_2 = len(held_out) > 0 and lower_bound >= min_match_rate
    if not condition_2:
        reasons.append(
            f"held-out agreement lower bound {lower_bound:.4f} (n={len(held_out)}) is below "
            f"min_match_rate {min_match_rate}"
        )

    metrics["tau_sweep"] = tau_sweep(held_out, predict_fn)

    return StudentArtifact(
        tag=tag.task,
        spec_version=spec.version,
        package_version=package_version,
        backbone=backbone,
        options=tag.options,
        kind=tag.kind,
        dims=dims,
        idf=vectorizer.idf,
        model_text=booster.model_to_string(),
        created_at=time.time(),
        admitted=condition_1 and condition_2,
        reasons=tuple(reasons),
        metrics=metrics,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prismyra.learn.fit")
    parser.add_argument("--learn-spec", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", required=True, help="the model id these experiences were collected under")
    parser.add_argument("--dims", type=int, default=256)
    parser.add_argument("--held-out-fraction", type=float, default=0.2)
    parser.add_argument("--min-match-rate", type=float, default=0.97)
    parser.add_argument("--accuracy-margin", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=100)
    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=None,
        help="LightGBM's own default (20) is too strict for a newly-registered tag with only a little logged "
        "experience; lower this to actually fit a tree on a small training set",
    )
    args = parser.parse_args(argv)

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print('this needs the extra: pip install "prismyra[learn]"')
        return 1

    import prismyra

    artifact = run(
        spec_path=args.learn_spec,
        task=args.task,
        log_dir=args.log_dir,
        package_version=prismyra.__version__,
        backbone=args.backbone,
        dims=args.dims,
        held_out_fraction=args.held_out_fraction,
        min_match_rate=args.min_match_rate,
        accuracy_margin=args.accuracy_margin,
        num_threads=args.num_threads,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        min_data_in_leaf=args.min_data_in_leaf,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "ADMITTED" if artifact.admitted else "NOT ADMITTED"
    print(f"{args.task}: {verdict}")
    for reason in artifact.reasons:
        print(f"  - {reason}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
