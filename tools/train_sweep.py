"""learn2: does the stage-2 student (LightGBM, DISTILL-RL-DESIGN-v2.md section 3) catch up to Prismyra
as the amount of logged experience grows? A learning curve over the train-pool prefix size, at each
point (a) a small LightGBM/feature grid chosen by cross-validation *inside the training subset only*
(never touching held_out or eval -- the task's own instruction), then (b) scored once against the
fixed held_out slice (student/Prismyra agreement, Wilson lower bound) and the fixed eval slice
(student accuracy vs Prismyra accuracy against gold).

Reuses prismyra.learn.fit's own building blocks (fit_student, argmax, wilson_lower_bound) rather than
re-deriving them, so this script's numbers are the same arithmetic RUN-learn1.md's single-point check
used -- just run at more sizes, with a held-out/eval pair fixed across the whole curve instead of
re-split per call (fit.run()'s own split is appropriate for a one-off fit; a learning curve needs the
denominator to stay still while only the training set size moves).

Usage (CPU only; no GPU, no network -- pools are already on disk):
    python3 train_sweep.py --pools-dir /work/learn2/pools --out-dir /work/learn2/sweep \
      --tag snake --sizes 800,2000,5000,10000,20000 --options no,yes
    python3 train_sweep.py --pools-dir /work/learn2/pools --out-dir /work/learn2/sweep \
      --tag tweet --sizes 800,2000,5000,9784 --options negative,neutral,positive
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prismyra.learn.features import HashingTfidf, tokenize
from prismyra.learn.fit import _score_eval, _soft_rows, argmax, wilson_lower_bound


@dataclass(frozen=True)
class Rec:
    context: str
    probabilities: dict
    gold: str | None = None


def load_jsonl(path: Path) -> list[Rec]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            out.append(Rec(context=raw["context"], probabilities=raw["probabilities"], gold=raw.get("gold")))
    return out


def dedup(records: list[Rec]) -> list[Rec]:
    seen: set[str] = set()
    out = []
    for r in records:
        if r.context in seen:
            continue
        seen.add(r.context)
        out.append(r)
    return out


def fit_lgb(
    train: list[Rec],
    options: tuple[str, ...],
    *,
    dims: int,
    num_boost_round: int,
    min_data_in_leaf: int,
    num_threads: int,
    seed: int = 0,
):
    import lightgbm as lgb

    corpus = [tokenize(r.context) for r in train]
    vectorizer = HashingTfidf.fit(corpus, dims)
    x_rows, y_rows, w_rows = _soft_rows(train, options, vectorizer)  # type: ignore[arg-type]
    dataset = lgb.Dataset(np.asarray(x_rows, dtype=np.float64), label=y_rows, weight=w_rows, free_raw_data=True)
    params = {
        "objective": "multiclass",
        "num_class": len(options),
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": num_threads,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "min_data_in_leaf": min_data_in_leaf,
        "verbosity": -1,
    }
    booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
    return booster, vectorizer


def predict_fn_for(booster, vectorizer, options: tuple[str, ...]):
    def predict_fn(context: str) -> dict:
        vec = vectorizer.transform(tokenize(context))
        (probs,) = booster.predict(np.asarray([vec], dtype=np.float64))
        return dict(zip(options, (float(p) for p in probs), strict=True))

    return predict_fn


def soft_log_loss(predict_fn, fold: list[Rec]) -> float:
    """Cross-entropy of the student's predicted distribution against each held-back row's *soft*
    (teacher) label distribution -- the one criterion available during cross-validation, since the
    held_out/eval slices (gold, or kept for the final agreement number) are off limits by the task's
    own instruction ("取り置いた評価の問題で選ばず"). Lower is better."""
    eps = 1e-12
    total = 0.0
    n = 0
    for r in fold:
        pred = predict_fn(r.context)
        for opt, p_teacher in r.probabilities.items():
            if p_teacher <= 0.0:
                continue
            p_student = max(pred.get(opt, 0.0), eps)
            total += -p_teacher * np.log(p_student)
            n += 1
    return total / n if n else float("inf")


def cv_select(
    train: list[Rec], options: tuple[str, ...], grid: list[dict], *, k: int, num_threads: int, seed: int
) -> tuple[dict, list[dict]]:
    """k-fold CV *inside* `train` only. Returns (best_params, all_rows_with_cv_score)."""
    n = len(train)
    fold_idx = [i % k for i in range(n)]
    rng_rows = []
    best = None
    for params in grid:
        losses = []
        for fold in range(k):
            tr = [train[i] for i in range(n) if fold_idx[i] != fold]
            va = [train[i] for i in range(n) if fold_idx[i] == fold]
            if not tr or not va:
                continue
            booster, vectorizer = fit_lgb(tr, options, num_threads=num_threads, seed=seed, **params)
            pf = predict_fn_for(booster, vectorizer, options)
            losses.append(soft_log_loss(pf, va))
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        row = {**params, "cv_log_loss": mean_loss}
        rng_rows.append(row)
        if best is None or mean_loss < best[1]:
            best = (params, mean_loss)
    assert best is not None
    return best[0], rng_rows


def eval_point(
    train_full: list[Rec],
    held_out: list[Rec],
    eval_set: list[Rec],
    options: tuple[str, ...],
    params: dict,
    *,
    num_threads: int,
    seed: int,
) -> dict:
    t0 = time.monotonic()
    booster, vectorizer = fit_lgb(train_full, options, num_threads=num_threads, seed=seed, **params)
    fit_s = time.monotonic() - t0
    pf = predict_fn_for(booster, vectorizer, options)

    # held_out agreement with Prismyra (teacher argmax), Wilson lower bound.
    agree = 0
    for r in held_out:
        student_choice = argmax(pf(r.context))
        teacher_choice = argmax(r.probabilities)
        agree += int(student_choice == teacher_choice)
    lb = wilson_lower_bound(agree, len(held_out))

    # eval: gold accuracy, student vs Prismyra (Prismyra's own probabilities are the teacher's prob on
    # this same eval item -- collected once, same way RUN-learn1.md's eval_set did).
    eval_items = [{"context": r.context, "gold": r.gold, "prismyra_probabilities": r.probabilities} for r in eval_set]

    class _Item:
        def __init__(self, d):
            self.context = d["context"]
            self.gold = d["gold"]
            self.prismyra_probabilities = d["prismyra_probabilities"]

    eval_metrics = _score_eval([_Item(d) for d in eval_items], pf)

    # per-item student inference latency (CPU, single-row predict -- the shape a served request would
    # actually pay, not an amortised batch number).
    sample = held_out[: min(200, len(held_out))]
    t1 = time.monotonic()
    for r in sample:
        pf(r.context)
    infer_ms = (time.monotonic() - t1) / max(1, len(sample)) * 1000.0

    return {
        "train_n": len(train_full),
        "params": params,
        "fit_seconds": fit_s,
        "held_out_n": len(held_out),
        "held_out_agreement": agree / len(held_out) if held_out else 0.0,
        "held_out_lower_bound_95": lb,
        "eval_n": eval_metrics["n"],
        "student_accuracy": eval_metrics["student_accuracy"],
        "prismyra_accuracy": eval_metrics.get("prismyra_accuracy"),
        "student_infer_ms_per_item": infer_ms,
    }


DEFAULT_GRID = [
    {"dims": dims, "num_boost_round": nbr, "min_data_in_leaf": mdl}
    for dims in (128, 256, 1024)
    for nbr in (100, 300)
    for mdl in (5, 20)
]
LEARN1_DEFAULT = {"dims": 256, "num_boost_round": 100, "min_data_in_leaf": 5}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pools-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", required=True, choices=["snake", "tweet"])
    p.add_argument("--sizes", required=True, help="comma-separated train-pool prefix sizes")
    p.add_argument("--options", required=True, help="comma-separated class names, teacher's option order")
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pools = Path(args.pools_dir)
    options = tuple(args.options.split(","))
    sizes = [int(s) for s in args.sizes.split(",")]

    train_pool = dedup(load_jsonl(pools / f"{args.tag}_train_pool.jsonl"))
    held_out = load_jsonl(pools / f"{args.tag}_held_out.jsonl")
    eval_set = load_jsonl(pools / f"{args.tag}_eval.jsonl")
    print(
        f"[{args.tag}] train_pool(deduped)={len(train_pool)} held_out={len(held_out)} eval={len(eval_set)}", flush=True
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_default = []
    curve_cv = []
    cv_detail = []

    for n in sizes:
        if n > len(train_pool):
            print(f"  size {n} > available deduped pool {len(train_pool)}; skipping", flush=True)
            continue
        subset = train_pool[:n]
        print(f"  -- size={n} --", flush=True)

        t0 = time.monotonic()
        row_default = eval_point(
            subset, held_out, eval_set, options, LEARN1_DEFAULT, num_threads=args.num_threads, seed=args.seed
        )
        row_default["config"] = "learn1_default"
        curve_default.append(row_default)
        print(
            f"    default {LEARN1_DEFAULT}: student_acc={row_default['student_accuracy']:.3f} "
            f"prismyra_acc={row_default['prismyra_accuracy']}, agree_lb={row_default['held_out_lower_bound_95']:.4f}, "
            f"infer_ms={row_default['student_infer_ms_per_item']:.3f} ({time.monotonic() - t0:.1f}s)",
            flush=True,
        )

        t0 = time.monotonic()
        best_params, grid_rows = cv_select(
            subset, options, DEFAULT_GRID, k=args.cv_folds, num_threads=args.num_threads, seed=args.seed
        )
        cv_detail.append({"train_n": n, "grid": grid_rows, "chosen": best_params})
        row_cv = eval_point(
            subset, held_out, eval_set, options, best_params, num_threads=args.num_threads, seed=args.seed
        )
        row_cv["config"] = "cv_selected"
        curve_cv.append(row_cv)
        print(
            f"    cv-selected {best_params}: student_acc={row_cv['student_accuracy']:.3f} "
            f"prismyra_acc={row_cv['prismyra_accuracy']}, agree_lb={row_cv['held_out_lower_bound_95']:.4f}, "
            f"infer_ms={row_cv['student_infer_ms_per_item']:.3f} ({time.monotonic() - t0:.1f}s)",
            flush=True,
        )

        # Flush after every size so a long run's partial progress is never lost.
        (out_dir / f"{args.tag}_curve.json").write_text(
            json.dumps({"default": curve_default, "cv_selected": curve_cv}, indent=2), encoding="utf-8"
        )
        (out_dir / f"{args.tag}_cv_detail.json").write_text(json.dumps(cv_detail, indent=2), encoding="utf-8")

    print(f"wrote {out_dir / (args.tag + '_curve.json')} and {out_dir / (args.tag + '_cv_detail.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
