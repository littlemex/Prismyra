"""inv round 3, item 1's speed half: alternating timing of open_batch before vs after per-document padding.

Since round 3 changes the production code path unconditionally (no env toggle -- the old combined-total rounding
is gone), "before" is measured by checking out the previous commit's engine.py into a sibling import path is not
practical from inside one process; instead this compares two shapes that *should* cost the same under either
scheme (no padding either way: every document asks a count that is already a power of two) against a shape where
round 3 does more work than the old scheme would have (one document far under its neighbours' bucket, which round
3 now pads independently instead of letting it ride for free in the old shared-padding scheme). If round 3 has a
real cost, it should show up as the second shape being disproportionately slower, not just slower in absolute
terms (longer context, more questions overall also cost more).
"""
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import tasks  # noqa: E402
from prismyra import Prismyra  # noqa: E402

MODEL = os.environ["PRISMYRA_MODEL"]
N_RUNS = int(os.environ.get("PRISMYRA_N_RUNS", "6"))
items = tasks.load("race", 8, split="validation", seed=6)
engine = Prismyra(MODEL, paged=True, graphs=False, group=32)


def questions_at(item, n):
    import dataclasses

    base = item.questions
    return [dataclasses.replace(base[i % len(base)], id=f"{base[i % len(base)].id}_{i}") for i in range(n)]


shapes = {
    # Both already powers of two: round 3 pads neither document (padded_counts == counts either way), so this is
    # the "round 3 should cost nothing extra" case.
    "aligned_4_4": [(items[0], 4), (items[1], 4)],
    # 3 is not a power of two: round 3 pads *this* document to 4 on its own; the old scheme would have padded the
    # combined total (3+4=7 -> 8) instead, once, for the whole pass. Round 3 does the same amount of total padding
    # here (1 row) by coincidence of this specific shape, so this mainly checks nothing structural got slower.
    "unaligned_3_4": [(items[0], 3), (items[1], 4)],
    # 1 is far under 4: round 3 pads document 0 up to only 1 (no padding, already the floor) while document 1
    # pads to 4 -- same total either way (5 real -> 5 total under round 3's sum, vs 5 real -> 8 under the old
    # single combined-total rounding). Round 3 should be *cheaper* here, not more expensive, since the old scheme
    # over-padded a small companion to match its larger neighbour and round 3 does not.
    "lopsided_1_4": [(items[0], 1), (items[1], 4)],
}

times = {name: [] for name in shapes}
for _ in range(N_RUNS):
    for name, pairs in shapes.items():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with engine.open_batch([it.context for it, _ in pairs]) as batch:
            batch.ask([questions_at(it, n) for it, n in pairs])
        torch.cuda.synchronize()
        times[name].append((time.perf_counter() - t0) * 1000)

for name, ts in times.items():
    ts_sorted = sorted(ts)
    print(f"{name}: median={statistics.median(ts):.2f}ms min={ts_sorted[0]:.2f} max={ts_sorted[-1]:.2f} "
          f"all={['%.2f' % t for t in ts]}")
