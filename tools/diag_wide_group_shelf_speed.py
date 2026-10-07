#!/usr/bin/env python3
"""fp8spd6 (round8, task2): speed of `wide_group` combined with the Shelf/Batcher fused path. 48 questions
about one document, answered in one wide (group=64) fused pass, against the only way the *same* Batcher
could serve 48 questions before this round at all: split into two <=32-question requests (`Batcher.submit`
itself refuses a single request over `engine.group` questions without `wide_group`), each its own
single-document fused pass. Alternating, same engine, same card.
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra
from prismyra.schedule import Batcher

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
CONTEXT = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
)
ROUNDS = 11
WARMUP = 3


def questions_for(n, tag=""):
    return [Boolean(id=f"{tag}q{i}", prompt=f"条項 {i} はこの文書の主題について述べているか。") for i in range(n)]


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True, wide_group=True, paged=True)
    batcher = Batcher(engine).start()
    try:
        qs48 = questions_for(48)
        qs24a = questions_for(24, "a_")
        qs24b = questions_for(24, "b_")

        wide_ms, split_ms = [], []
        for r in range(ROUNDS):
            # A fresh context each round (a trailing round marker makes the digest unique) -- otherwise the
            # second and later rounds would find the document already resident on the shelf and take the
            # branch-only path, never exercising the fused *read* this measurement means to compare.
            ctx_wide = CONTEXT + f"\n(round {r} wide)"
            ctx_split = CONTEXT + f"\n(round {r} split)"
            t0 = time.perf_counter()
            batcher.ask(ctx_wide, qs48, timeout=60.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batcher.ask(ctx_split, qs24a, timeout=60.0)
            batcher.ask(ctx_split, qs24b, timeout=60.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            if r >= WARMUP:
                wide_ms.append((t1 - t0) * 1e3)
                split_ms.append((t2 - t1) * 1e3)
            print(f"round {r}: wide={1e3*(t1-t0):.1f}ms split={1e3*(t2-t1):.1f}ms", flush=True)

        print(f"wide:  median={statistics.median(wide_ms):.1f} min={min(wide_ms):.1f} max={max(wide_ms):.1f}")
        print(f"split: median={statistics.median(split_ms):.1f} min={min(split_ms):.1f} max={max(split_ms):.1f}")
        delta = (statistics.median(wide_ms) - statistics.median(split_ms)) / statistics.median(split_ms) * 100
        print(f"wide vs split: {delta:+.1f}%")
    finally:
        batcher.stop()


if __name__ == "__main__":
    main()
