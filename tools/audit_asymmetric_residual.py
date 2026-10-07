"""inv round 3: the actual documented residual (tests/test_gpu.py's 0.0128/0.024, RUN-inv.md round 2's 0.057) is
specifically an *asymmetric* companion case -- a target with its own question count and a companion with a
*different* one, where the pre-round-3 combined-total `_round_rows` landed the pair on a different bucket than
either would have gotten alone (e.g. target=2 + companion=1 -> combined=3 -> old bucket 4, vs. target alone -> 2).
`audit_sm120.py`'s sweep (round 2/3) always gives every document in a chunk the *same* question count, which never
asks for this: target=1 + companion=1 is combined=2 either way (bucket 2, no padding under the old scheme either),
which is a different, more fundamental question-count-independent residual (see RUN-inv.md round 3 notes) that
round 3's per-document padding was never going to touch and does not claim to.

This script isolates the asymmetric case specifically, round-3's actual target, before vs after.
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
os.environ.setdefault("PRISMYRA_EXPERTS", "nvfp4")

items = tasks.load("race", 8, split="validation", seed=4)
engine = Prismyra(MODEL, paged=True, graphs=False, group=32)


def questions_at(item, n):
    import dataclasses

    base = item.questions
    return [dataclasses.replace(base[i % len(base)], id=f"{base[i % len(base)].id}_{i}") for i in range(n)]


print("target_q companion_q  max_move  exact  decision_flip", flush=True)
worst = 0.0
any_nonexact = False
for target_q in (1, 2, 3, 5, 7):
    for companion_q in (1, 2, 3, 5, 7):
        if target_q == companion_q:
            continue  # the symmetric case is audit_sm120.py's territory, not this script's
        target, companion = items[0], items[1]
        want = engine.ask(target.context, questions_at(target, target_q))
        with engine.open_batch([target.context, companion.context]) as batch:
            got = batch.ask([questions_at(target, target_q), questions_at(companion, companion_q)])[0]
        max_move = 0.0
        flip = False
        for q in questions_at(target, target_q):
            g = got[q.id].probabilities
            w = want[q.id].probabilities
            g_t = torch.tensor([g[o] for o in sorted(g)])
            w_t = torch.tensor([w[o] for o in sorted(w)])
            max_move = max(max_move, (g_t - w_t).abs().max().item())
            flip = flip or got[q.id].option != want[q.id].option
        exact = max_move == 0.0
        worst = max(worst, max_move)
        any_nonexact = any_nonexact or not exact
        print(f"{target_q:8d} {companion_q:12d}  {max_move:.6f}  {exact!s:5}  {flip!s}", flush=True)

print(f"\nworst asymmetric move: {worst:.6f}, any non-exact: {any_nonexact}")
