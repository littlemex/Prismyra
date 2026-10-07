#!/usr/bin/env python3
"""rel2: does engine.open_batch()'s own questions/second change with interleaved_fork? Builds one
engine, round-robins interleaved_fork True/False, times `with engine.open_batch(contexts) as batch:
batch.ask(...)` over GROUP_SIZE RACE documents x QN questions each, several rounds, reports
questions/second = (GROUP_SIZE * QN) / seconds for both settings.

Usage: PRISMYRA_SRC=... PRISMYRA_MODEL=... RACE150_PATH=... python3 rel_open_batch_throughput.py
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
RACE150 = os.environ.get("RACE150_PATH", "/work/recon/race150.json")
GROUP_SIZE = int(os.environ.get("REL_GROUP_SIZE", "8"))
QN = int(os.environ.get("REL_QN", "2"))
ROUNDS = int(os.environ.get("REL_ROUNDS", "15"))
DROP = int(os.environ.get("REL_DROP", "4"))


def load_contexts(n):
    data = json.load(open(RACE150))
    items = data["items"]
    return [items[i % len(items)]["context"] for i in range(n)]


def make_questions(n):
    return [Boolean(id=f"q{i}", prompt=f"この文書は{i}番目の論点を支持しているか。") for i in range(n)]


def main():
    engine = Prismyra(MODEL, require_kernels=True, paged=True, graphs=False, group=max(64, GROUP_SIZE))
    contexts = load_contexts(GROUP_SIZE)
    qs_per_doc = [make_questions(QN) for _ in range(GROUP_SIZE)]

    results = {"two_pass": [], "fused": []}
    for r in range(ROUNDS):
        for mode, flag in (("two_pass", False), ("fused", True)):
            engine.interleaved_fork = flag
            t0 = time.perf_counter()
            with engine.open_batch(contexts) as batch:
                batch.ask(qs_per_doc)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if r >= DROP:
                results[mode].append(dt)

    print(f"open_batch: {GROUP_SIZE} docs x {QN} questions, {ROUNDS - DROP} measured rounds")
    for mode, samples in results.items():
        samples = sorted(samples)
        med = statistics.median(samples)
        qps = (GROUP_SIZE * QN) / med
        print(
            f"{mode}: median={med * 1000:.1f}ms (min={samples[0] * 1000:.1f} max={samples[-1] * 1000:.1f}) "
            f"-> {qps:.2f} questions/sec"
        )


if __name__ == "__main__":
    main()
