"""inv (coordinator round 2, point 2): measure whether `PRISMYRA_ROUND_ROWS_TO_GROUP=1` (every branch pass runs at
the full `group` width, closing the qn=1-companion residual at the root) is affordable, instead of assuming it from
first principles. Two things measured on real hardware, alternating within one process:

1. Correctness: does it eliminate the qn=1-with-companion residual `audit_sm120.py` isolated?
2. Speed: median of >=5 alternating runs (default vs. forced-to-group), same process, same card, same group size.

`group` is a parameter here (not fixed at the production 32/64) because the production width did not fit this
card's memory at all once every pass pays for the full width (`_check_fits` refuses the request outright) --
that refusal is itself part of the answer, not a bug in this script.
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
GROUP = int(os.environ.get("PRISMYRA_TEST_GROUP", "8"))
N_RUNS = int(os.environ.get("PRISMYRA_N_RUNS", "6"))
os.environ.setdefault("PRISMYRA_EXPERTS", "nvfp4")

items = tasks.load("race", 6, split="validation", seed=2)
engine = Prismyra(MODEL, paged=True, graphs=False, group=GROUP)
target, companion = items[0], items[1]


def questions_at(item, n):
    import dataclasses

    base = item.questions
    return [dataclasses.replace(base[i % len(base)], id=f"{base[i % len(base)].id}_{i}") for i in range(n)]


# ---- correctness: qn=1 target + qn=1 companion, default vs forced-to-group ----
truth = engine.ask(target.context, questions_at(target, 1))
for mode, env_val in (("default", None), ("round_to_group", "1")):
    if env_val is None:
        os.environ.pop("PRISMYRA_ROUND_ROWS_TO_GROUP", None)
    else:
        os.environ["PRISMYRA_ROUND_ROWS_TO_GROUP"] = env_val
    with engine.open_batch([target.context, companion.context]) as batch:
        results = batch.ask([questions_at(target, 1), questions_at(companion, 1)])
    q = questions_at(target, 1)[0]
    got = results[0][q.id]
    want = truth[q.id]
    g_t = torch.tensor([got.probabilities[o] for o in sorted(got.probabilities)])
    w_t = torch.tensor([want.probabilities[o] for o in sorted(want.probabilities)])
    exact = torch.equal(g_t, w_t)
    move = (g_t - w_t).abs().max().item()
    print(f"correctness[{mode}]: exact={exact} move={move:.6f}", flush=True)

# ---- speed: alternate default/forced, N_RUNS each, same process ----
times = {"default": [], "round_to_group": []}
for i in range(N_RUNS):
    for mode, env_val in (("default", None), ("round_to_group", "1")):
        if env_val is None:
            os.environ.pop("PRISMYRA_ROUND_ROWS_TO_GROUP", None)
        else:
            os.environ["PRISMYRA_ROUND_ROWS_TO_GROUP"] = env_val
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with engine.open_batch([target.context, companion.context]) as batch:
            batch.ask([questions_at(target, 1), questions_at(companion, 1)])
        torch.cuda.synchronize()
        times[mode].append((time.perf_counter() - t0) * 1000)

for mode, ts in times.items():
    ts_sorted = sorted(ts)
    print(f"speed[{mode}] (ms, group={GROUP}, {N_RUNS} runs): median={statistics.median(ts):.2f} "
          f"min={ts_sorted[0]:.2f} max={ts_sorted[-1]:.2f} all={['%.2f' % t for t in ts]}", flush=True)

d_med = statistics.median(times["default"])
r_med = statistics.median(times["round_to_group"])
print(f"\nround_to_group vs default: {(r_med / d_med - 1) * 100:+.1f}% at group={GROUP}")
