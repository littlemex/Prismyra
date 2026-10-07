"""inv5, round 7 (remaining task 4 from the original brief): confirm what docs/PERFORMANCE.md's published
+10% (solo put-ask-drop lifecycle, 224->247ms) and -21% (open_batch throughput at group 32/64) actually
measured, against round 5 (inv4)'s own measurement of the dispatcher registration's cost in isolation
(+0.2%/+0.9%, "global" scope-of-registration vs "narrow" point-fixes -- RUN-inv.md round 5, item 33).

These are two different comparisons, not a contradiction:
  - round 5's "global vs narrow" holds the *amount of invariance protection* roughly constant (narrow already
    covers router + shared_expert_gate + GDN gate x2 directly) and asks only "does registering the global aten
    dispatcher cost more than those four direct point-fixes alone". A small number is expected because narrow
    already does most of the job.
  - PERFORMANCE.md's number was published at v0.3.0, before any narrow point-fix existed -- the global
    dispatcher was the *only* mechanism providing the guarantee at all. "A development build with it removed"
    there means removing the one and only protection, not swapping it for an equivalent-but-narrower one.

This reproduces PERFORMANCE.md's own comparison on current code and hardware: the batch-invariant dispatcher
registered (today's default, `paged=True`) against the same paged engine with the dispatcher forced off via
`_disable_batch_invariance()` right after construction (so the read path, admission, and everything else stay
identical -- only the aten::mm/addmm/bmm/linear registration differs), alternating measurement, 15 rounds with
the first 4 discarded, same as the project's own measurement discipline.

Usage: PRISMYRA_MODEL=<repo> python3 measure_dispatcher_cost_round7.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402
import prismyra.engine as _engine_mod  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
ROUNDS = 15
DISCARD = 4

CONTEXT = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
    "seller when the item is faulty and by the buyer otherwise. "
) * 40  # a few thousand tokens, closer to the model card's own context length than a one-line fixture
QUESTION = Boolean(id="q", prompt="Is a refund limited to unopened items?")


def _now() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def solo_lifecycle_ms(engine: Prismyra) -> float:
    t0 = _now()
    with engine.open_context(CONTEXT) as ctx:
        ctx.ask([QUESTION])
    t1 = _now()
    return (t1 - t0) * 1000


def open_batch_qs(engine: Prismyra, group: int, seconds: float = 3.0) -> float:
    """Questions answered per second over `seconds` of back-to-back `open_batch` calls at the given width."""
    docs = [CONTEXT] * 1  # one document, `group` questions per pass -- matches PERFORMANCE.md's "group 32/64"
    questions = [Boolean(id=f"q{i}", prompt=QUESTION.prompt) for i in range(group)]
    t0 = _now()
    answered = 0
    while True:
        with engine.open_batch(docs) as batch:
            batch.ask([questions])
        answered += group
        if time.perf_counter() - t0 >= seconds:
            break
    t1 = _now()
    return answered / (t1 - t0)


def summarize(label: str, on: list[float], off: list[float]) -> None:
    med_on, med_off = statistics.median(on), statistics.median(off)
    delta = (med_on - med_off) / med_off * 100
    print(f"\n{label}:")
    print(f"  dispatcher ON : median={med_on:.3f} min={min(on):.3f} max={max(on):.3f}")
    print(f"  dispatcher OFF: median={med_off:.3f} min={min(off):.3f} max={max(off):.3f}")
    print(f"  delta (ON vs OFF): {delta:+.1f}%")


def main():
    # The dispatcher registration is a process-wide `torch.library.Library`, not a per-engine setting (see
    # `_BATCH_INVARIANT_REFCOUNT`'s own docstring) -- two engines cannot hold it ON and OFF at once in the same
    # process. One engine is used throughout; `paged` stays True the whole time (so the read path, admission and
    # every narrow point-fix wired by rounds 1/5 are identical in both conditions) and only the module-level
    # dispatcher is toggled directly, bypassing the `paged` property (which no-ops on an unchanged value thanks
    # to round 4's `_invariance_claimed` guard).
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, "dispatcher not registered by default construction"
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    def with_dispatcher(on: bool, fn, *args):
        if on:
            if _engine_mod._BATCH_INVARIANT_REFCOUNT == 0:
                _engine_mod._enable_batch_invariance()
        else:
            while _engine_mod._BATCH_INVARIANT_REFCOUNT > 0:
                _engine_mod._disable_batch_invariance()
        return fn(*args)

    print("=== solo read's full put-ask-drop lifecycle (PERFORMANCE.md: 224 -> 247 ms, +10%) ===")
    on_times, off_times = [], []
    for i in range(ROUNDS):
        t_on = with_dispatcher(True, solo_lifecycle_ms, engine)
        t_off = with_dispatcher(False, solo_lifecycle_ms, engine)
        if i >= DISCARD:
            on_times.append(t_on)
            off_times.append(t_off)
    with_dispatcher(True, lambda: None)  # leave the dispatcher ON (the shipped default) before the next section
    summarize("solo put-ask-drop (ms, lower is better)", on_times, off_times)

    for group in (32, 64):
        print(f"\n=== open_batch throughput at group={group} "
              f"(PERFORMANCE.md: 77.7->61.7 q/s at 32, 90.2->70.3 at 64) ===")
        on_qs, off_qs = [], []
        for i in range(ROUNDS):
            q_on = with_dispatcher(True, open_batch_qs, engine, group, 2.0)
            q_off = with_dispatcher(False, open_batch_qs, engine, group, 2.0)
            if i >= DISCARD:
                on_qs.append(q_on)
                off_qs.append(q_off)
        with_dispatcher(True, lambda: None)
        summarize(f"open_batch q/s at group={group} (higher is better)", on_qs, off_qs)


if __name__ == "__main__":
    main()
