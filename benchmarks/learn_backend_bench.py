"""Real-hardware comparison of `prismyra.learn.backends`' two `StudentBackend` implementations --
`"lightgbm"` (the default) and `"xgboost"` -- on identical data: training wall-clock time, single-row
prediction latency (the number that matters for stage 4's routed path, which predicts one request at a
time), and the peak resident-memory delta training and holding each model costs. Not part of the pytest
suite -- run by hand on the machine named in the run log, same convention every other number in this
project uses (`benchmarks/idle_latency_bench.py`'s own docstring).

    pip install "prismyra[learn,learn-xgb]"
    python benchmarks/learn_backend_bench.py --rows 2000 --dims 256 --num-boost-round 100 --predict-reps 200
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import time

import numpy as np

from prismyra.learn.backends import STUDENT_BACKENDS, get_backend


def _peak_rss_kib() -> int:
    # ru_maxrss is cumulative-peak-so-far on both Linux and macOS (ru_maxrss never decreases within a
    # process), which is exactly why this is read as a *delta* around each measured section below rather
    # than trusted as "this section's own cost" on its own. The unit itself is not the same on both: Linux
    # reports KiB (`man getrusage`), macOS reports bytes -- normalised to KiB here so the field name
    # ("_kib") is honest on whichever platform this ran on.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform != "darwin" else raw // 1024


def make_synthetic_rows(n_rows: int, dims: int, num_class: int, seed: int) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    x = rng.random((n_rows, dims)).tolist()
    y = rng.integers(0, num_class, size=n_rows).tolist()
    w = [1.0] * n_rows
    return x, y, w


def bench_backend(name: str, *, x, y, w, num_class: int, num_boost_round: int, predict_reps: int) -> dict:
    backend = get_backend(name)
    gc.collect()
    rss_before_train = _peak_rss_kib()
    t0 = time.perf_counter()
    model_text = backend.train(
        x, y, w, num_class=num_class, num_threads=1, seed=0, num_boost_round=num_boost_round, min_data_in_leaf=1
    )
    train_s = time.perf_counter() - t0
    rss_after_train = _peak_rss_kib()

    gc.collect()
    rss_before_load = _peak_rss_kib()
    model = backend.load(model_text)
    rss_after_load = _peak_rss_kib()

    # Single-row predictions, one at a time -- `students.StudentHandle.predict`'s own shape on the routed
    # path, not a batch call. Warm-up excluded from the timed reps the same way every other latency number
    # in this project's benchmarks excludes it.
    row = np.asarray([x[0]], dtype=np.float64)
    for _ in range(5):
        model.predict_proba(row)
    samples_ms = []
    for _ in range(predict_reps):
        t0 = time.perf_counter()
        model.predict_proba(row)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()

    def pct(p: float) -> float:
        idx = min(len(samples_ms) - 1, int(len(samples_ms) * p))
        return samples_ms[idx]

    return {
        "backend": name,
        "backend_version": backend.library_version(),
        "model_text_bytes": len(model_text.encode("utf-8")),
        "train_s": train_s,
        "train_rss_delta_kib": rss_after_train - rss_before_train,
        "load_rss_delta_kib": rss_after_load - rss_before_load,
        "predict_p50_ms": statistics.median(samples_ms),
        "predict_p99_ms": pct(0.99),
        "predict_mean_ms": statistics.mean(samples_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--dims", type=int, default=256)
    parser.add_argument("--num-class", type=int, default=3)
    parser.add_argument("--num-boost-round", type=int, default=100)
    parser.add_argument("--predict-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    x, y, w = make_synthetic_rows(args.rows, args.dims, args.num_class, args.seed)

    results = []
    for name in sorted(STUDENT_BACKENDS):
        try:
            results.append(
                bench_backend(
                    name,
                    x=x,
                    y=y,
                    w=w,
                    num_class=args.num_class,
                    num_boost_round=args.num_boost_round,
                    predict_reps=args.predict_reps,
                )
            )
        except ImportError as e:
            results.append({"backend": name, "error": str(e)})

    print(json.dumps({"rows": args.rows, "dims": args.dims, "num_class": args.num_class, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
