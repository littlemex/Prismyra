"""learn4: re-judge learn3's already-trained Snake student against a fresh, never-seen evaluation slice, under
the admission bar lowered in this change (prismyra.learn.spec.DEFAULT_MIN_MATCH_RATE/DEFAULT_MIN_ACCURACY_RATIO,
0.90/0.90 -- see RUN-learn4.md).

Why this does not retrain. The bar changed *after* RUN-learn3.md had already seen its own held-out/eval numbers
(94.70% held-out lower bound, 89.0% accuracy, both measured against the old 97% bar). Re-checking those same old
numbers against a new bar chosen with them already in view would be the judgment picking its own fixture. What
is legitimately re-usable without re-training is the *student itself*: LightGBM's `deterministic=True` with a
fixed build, instruction set, thread count and seed is a documented, and this project's own regression check
(`/Users/akazawt/tmp/smr/learn3/snake_curve_regression_check.json`) already verified byte-for-byte, contract --
so re-fitting the exact same (N=19,289, `snake_board_v1`, `num_boost_round=300`, `min_data_in_leaf=5`) model
`tools/train_sweep.py`'s own cross-validation chose for learn3 reproduces the identical student, not a new one.
Only the *evaluation data* is new: `tools/collect_fresh_eval.py`'s 1,100-row slice, excluded by exact context
text from every board learn2/learn3 ever drew.

Usage (CPU only, on the shared pod, same tools/data this change's other scripts already use):
    python3 rejudge_fresh_eval.py --pools-dir /work/learn2/pools --fresh-eval /work/learn4/snake_fresh_eval.jsonl \
      --out /work/learn4/rejudge_fresh_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import snake_features  # noqa: F401  (import registers "snake_board_v1" with prismyra.learn.features)
from train_sweep import Rec, dedup, eval_point, load_jsonl

from prismyra.learn.spec import DEFAULT_MIN_ACCURACY_RATIO, DEFAULT_MIN_MATCH_RATE

# learn3's own CV selection for N=19,289, snake_board_v1 (RUN-learn3.md section 3's table, and
# /Users/akazawt/tmp/smr/learn3/snake_cv_detail_structured.json's own "chosen" entry for train_n=19289).
# Not re-selected here: re-running cross-validation on this run's own fresh data would make the model fit to
# data this re-judgment is supposed to be independent of.
LEARN3_CV_SELECTED_PARAMS = {"dims": 1, "num_boost_round": 300, "min_data_in_leaf": 5}
TRAIN_N = 19289
OPTIONS = ("no", "yes")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pools-dir", required=True, help="learn2's own pools dir (snake_train_pool.jsonl lives here)")
    p.add_argument("--fresh-eval", required=True, help="collect_fresh_eval.py's own output (never-seen rows)")
    p.add_argument("--out", required=True)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pools = Path(args.pools_dir)
    train_pool = dedup(load_jsonl(pools / "snake_train_pool.jsonl"))
    if len(train_pool) != TRAIN_N:
        raise SystemExit(f"expected {TRAIN_N} deduped train rows (learn3's own number), got {len(train_pool)}")

    fresh: list[Rec] = load_jsonl(Path(args.fresh_eval))
    print(f"train_pool(deduped)={len(train_pool)} fresh_eval={len(fresh)}", flush=True)

    # The fresh slice plays both roles at once: condition 2 (held-out agreement with Prismyra) and condition 1
    # (gold accuracy vs Prismyra's own accuracy) are each a different reduction of the exact same rows, and
    # eval_point already computes both from two separate arguments -- passing the same list to both is how a
    # single fresh collection serves as the one "not yet used" set the task asked for, rather than two.
    row = eval_point(
        train_pool,
        held_out=fresh,
        eval_set=fresh,
        options=OPTIONS,
        params=LEARN3_CV_SELECTED_PARAMS,
        num_threads=args.num_threads,
        seed=args.seed,
        features_name="snake_board_v1",
    )

    held_out_lb = row["held_out_lower_bound_95"]
    student_acc = row["student_accuracy"]
    prismyra_acc = row["prismyra_accuracy"]
    required_acc = DEFAULT_MIN_ACCURACY_RATIO * prismyra_acc if prismyra_acc is not None else None

    condition_1 = required_acc is not None and student_acc >= required_acc
    condition_2 = held_out_lb >= DEFAULT_MIN_MATCH_RATE
    admitted = condition_1 and condition_2

    result = {
        "train_n": TRAIN_N,
        "params": LEARN3_CV_SELECTED_PARAMS,
        "features": "snake_board_v1",
        "fresh_eval_n": len(fresh),
        "admission_thresholds": {
            "min_match_rate": DEFAULT_MIN_MATCH_RATE,
            "min_accuracy_ratio": DEFAULT_MIN_ACCURACY_RATIO,
        },
        "condition_1_accuracy_vs_ratio": {
            "student_accuracy": student_acc,
            "prismyra_accuracy": prismyra_acc,
            "required_accuracy": required_acc,
            "met": condition_1,
        },
        "condition_2_held_out_agreement": {
            "lower_bound_95": held_out_lb,
            "min_match_rate": DEFAULT_MIN_MATCH_RATE,
            "met": condition_2,
        },
        "admitted": admitted,
        "raw_eval_point": row,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    verdict = "ADMITTED" if admitted else "NOT ADMITTED"
    print(f"snake-move-up-v1 (fresh eval, n={len(fresh)}): {verdict}", flush=True)
    print(
        f"  condition 1 (accuracy >= {DEFAULT_MIN_ACCURACY_RATIO} x Prismyra): "
        f"student={student_acc:.4f} prismyra={prismyra_acc:.4f} required={required_acc:.4f} -> {condition_1}",
        flush=True,
    )
    print(
        f"  condition 2 (held-out agreement lower bound >= {DEFAULT_MIN_MATCH_RATE}): "
        f"{held_out_lb:.4f} -> {condition_2}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
