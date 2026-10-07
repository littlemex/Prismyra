#!/usr/bin/env python3
"""recon: test the hypothesis that integ's "+23.8% worse at Q64" (RUN-integ.md 12.1, two SEPARATELY
CONSTRUCTED engines, "off"=defaults vs "on"=interleaved_fork+wide_group=True) was not a real fusion
slowdown but a METHODOLOGY confound: before this round's fix, only the "on" engine paid batch
invariance's own measured overhead (docstring: "+14.5ms/6.8% on a solo ask(), +18.9ms/1.8% on a
32-doc open_batch"), while fp8spd's own measurement (RUN-fp8spd.md round4-6, -19.8%/-9.6%) toggled
`engine.interleaved_fork` on ONE already-built engine, so both its "two_pass" and "fused" samples
paid the SAME batch-invariance state -- an apples-to-apples fusion-only comparison.

Mode env var CONFOUND_MODE:
  prefix  -- monkeypatch Prismyra.__init__ to the OLD conditional gate (paged or interleaved_fork
             only), reproducing the pre-fix world, before building either engine.
  fixed   -- (default) use this branch's actual code, unmodified.

Builds "off" (defaults) and "on" (interleaved_fork=True, wide_group=True) as two SEPARATE engine
constructions, sequentially (torn down between, to fit a single 46GB card), and times Q64 on the
same race150-concatenated document (group=32) both ways.
"""
import gc
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
RACE150 = os.environ.get("RACE150_PATH", "/work/recon/race150.json")
MODE = os.environ.get("CONFOUND_MODE", "fixed")


def build_race150_context(target_chars=22000):
    data = json.load(open(RACE150))
    items = data["items"]
    text = ""
    i = 0
    while len(text) < target_chars:
        text += items[i % len(items)]["context"] + "\n\n"
        i += 1
    return text


def make_questions(n):
    return [Boolean(id=f"q{i}", prompt=f"この文書は{i}番目の論点を支持しているか。") for i in range(n)]


if MODE == "prefix":
    import prismyra.engine as eng_mod

    _orig_init = eng_mod.Prismyra.__init__

    def _patched_init(self, *a, **kw):
        # Reproduce the pre-fix world: suppress the unconditional base claim this round added, and
        # restore the old conditional-on-interleaved_fork claim, by running the ACTUAL __init__ but
        # with _enable_batch_invariance patched to a no-op during the unconditional block only.
        real_enable = eng_mod._enable_batch_invariance
        calls = {"n": 0}

        def guarded_enable():
            calls["n"] += 1
            # First call inside __init__ is the new unconditional base claim (added this round, right
            # after self._invariance_claimed = False, before self.paged = paged). Make exactly that one
            # call look unavailable (ImportError is the one exception engine.py's own except clause
            # catches, leaving self._invariance_base_claimed at its initial False), so the later
            # interleaved_fork-only fallback block's `not self._invariance_base_claimed` guard still
            # lets it make its OWN, real claim -- reproducing the pre-fix world where only paged/
            # interleaved_fork ever claimed anything. Any later call (paged's setter, the fallback
            # block itself) goes through to the real function normally.
            if calls["n"] == 1:
                raise ImportError("recon test harness: simulating no base claim (pre-fix world)")
            real_enable()

        eng_mod._enable_batch_invariance = guarded_enable
        try:
            _orig_init(self, *a, **kw)
        finally:
            eng_mod._enable_batch_invariance = real_enable

    eng_mod.Prismyra.__init__ = _patched_init
    print("CONFOUND_MODE=prefix: reproducing the pre-fix conditional gate")
else:
    print("CONFOUND_MODE=fixed: using this branch's actual unconditional claim")


def time_q64(engine, context, rounds=8, drop=2):
    qs = make_questions(64)
    samples = []
    for r in range(rounds):
        t0 = time.perf_counter()
        engine.ask(context, qs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if r >= drop:
            samples.append(dt)
    return samples


def main():
    context_holder = {}

    print("--- building OFF (defaults) ---")
    off = Prismyra(MODEL, require_kernels=True, interleaved_fork=False, wide_group=False, group=32)
    import prismyra.engine as eng_mod

    print(f"OFF: refcount={eng_mod._BATCH_INVARIANT_REFCOUNT} base_claimed={getattr(off, '_invariance_base_claimed', None)} paged_claimed={off._invariance_claimed}")
    context = build_race150_context(22000)
    tok = len(off.tokenizer(context)["input_ids"])
    print(f"race150-concat doc tokens={tok}")
    off_samples = time_q64(off, context)
    off_med = statistics.median(off_samples)
    print(f"OFF Q64: median={off_med:.1f}ms samples={[round(s,1) for s in off_samples]}")

    del off
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("\n--- building ON (interleaved_fork=True, wide_group=True) ---")
    on = Prismyra(MODEL, require_kernels=True, interleaved_fork=True, wide_group=True, group=32)
    print(f"ON: refcount={eng_mod._BATCH_INVARIANT_REFCOUNT} base_claimed={getattr(on, '_invariance_base_claimed', None)} paged_claimed={on._invariance_claimed}")
    on_samples = time_q64(on, context)
    on_med = statistics.median(on_samples)
    print(f"ON  Q64: median={on_med:.1f}ms samples={[round(s,1) for s in on_samples]}")

    pct = (on_med - off_med) / off_med * 100
    print(f"\nOFF vs ON (two separately constructed engines, Q64, race150 doc, group=32): diff={pct:+.2f}%")


if __name__ == "__main__":
    main()
