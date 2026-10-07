"""inv round 3: isolate the one case per-document padding (round 3) did not close -- every document in a batch
(target and companions alike) asking exactly one question. `_round_rows(1, group)` is already 1, the floor, so
there is no padding for round 3's per-document scheme to make company-independent: the target's own row is never
padded either way, and the only thing that changes between "alone" (M=1) and "with N companions" (M=1+N, all
real, no padding anywhere) is the pass's total row count itself. Measures whether this residual's magnitude grows
with the number of same-shaped companions (model-agnostic: set PRISMYRA_MODEL and, for NVFP4, PRISMYRA_EXPERTS=nvfp4
plus the two side-file env vars before running).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import tasks  # noqa: E402
from prismyra import Prismyra  # noqa: E402

MODEL = os.environ["PRISMYRA_MODEL"]
items = tasks.load("race", 10, split="validation", seed=5)
engine = Prismyra(MODEL, paged=True, graphs=False, group=32)


def q1(item):
    return [item.questions[0]]


truth = {id(it): engine.ask(it.context, q1(it)) for it in items}
print("ask() ground truth done", flush=True)

for n_companions in (1, 2, 3, 7):
    chunk = items[: 1 + n_companions]
    with engine.open_batch([it.context for it in chunk]) as batch:
        results = batch.ask([q1(it) for it in chunk])
    target = chunk[0]
    want = truth[id(target)]
    q = q1(target)[0]
    g = results[0][q.id].probabilities
    w = want[q.id].probabilities
    g_t = torch.tensor([g[o] for o in sorted(g)])
    w_t = torch.tensor([w[o] for o in sorted(w)])
    move = (g_t - w_t).abs().max().item()
    print(f"n_companions={n_companions} (total_M={1+n_companions}): move={move:.6f} exact={move == 0.0}", flush=True)
