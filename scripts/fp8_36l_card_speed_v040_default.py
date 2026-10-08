"""Does the published fp8-36l model card's Latency table reflect v0.4.0's DEFAULT construction?

Background (RUN-tok.md, SYNTHESIS-v2.md 0-2 for the nvfp4 sibling of this same bug): the card
(`FP8-36l-MODEL-CARD-a8-draft.md`, live at HF commit 9d7cd0bc) reports 96.8/204.2/411.4 ms for 1/16/64
questions. The script that produced those numbers, `acc-wt/scripts/a_speed_interleave.py` (commit
`5622132`), builds its engine with `Prismyra(MODEL, group=32)` -- no `paged=True`. Read directly from that
commit's own `prismyra/engine.py`, the batch-invariance claim at that point in history was gated
`if paged and on_cuda:` -- so that measurement never claimed batch-invariance at all. The current
`origin/main` (v0.4.0, b31b80f) claims it unconditionally (`engine.py:672`, `if on_cuda:`) for every CUDA
engine regardless of `paged`/`interleaved_fork`. This script re-measures the SAME protocol (same race150
document, same widths, same alternation discipline) on the current v0.4.0 default construction -- no
`paged=`, no `graphs=`, no env overrides, exactly `Prismyra(MODEL, require_kernels=True)` -- to see whether
the card's numbers still hold.

Protocol (BRIEF-COMMON.md rule 3): one round = 1Q, then 16Q, then 64Q, in that order; 15 rounds; the first
4 rounds are discarded as warm-up; report median/min/max of the remaining 11 per width. A <5% difference is
not claimed as real without this alternation (also satisfied here: all three widths measured in the same
process, close together in time, so GPU/driver warm-up drift is shared across them).

Usage: fp8_36l_card_speed_v040_default.py OUT.json
"""
import os, sys, json, statistics, time, uuid

os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from prismyra import Prismyra, Choice

MODEL = os.environ.get("FP8_SPEED_MODEL", "/work/next/models/p35-l36-kd_a8_2xrows")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/tok/results/fp8_36l_card_speed_v040_default.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
WIDTHS = (1, 16, 64)
N_ROUNDS = 15
DISCARD_ROUNDS = 4
L = "ABCDEFGH"

data = json.load(open("/work/items/race150.json"))
ctx, qs = [], []
for it in data["items"]:
    ctx.append(it["context"])
    qs.extend(it["questions"])
    if sum(map(len, ctx)) >= 5300:
        break
context = "\n\n".join(ctx)
print(f"context chars: {len(context)}", flush=True)


def make_questions(width):
    sel = [qs[i % len(qs)] for i in range(width)]
    return [
        Choice(
            id=f"q{i}",
            prompt=q["question"] + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(q["options"])),
            choices=list(L[: len(q["options"])]),
        )
        for i, q in enumerate(sel)
    ]


questions_by_width = {w: make_questions(w) for w in WIDTHS}

# v0.4.0 DEFAULT construction. Deliberately nothing else: no paged=, no graphs=, no env var overrides. The
# only non-default flag is require_kernels=True, so a silent fallback to the unfused path (which would make
# "does this measure the fused kernels" moot) fails loudly instead of quietly passing as "fast enough".
eng = Prismyra(MODEL, require_kernels=True)
print("engine.paged =", eng.paged, "engine.interleaved_fork =", eng.interleaved_fork, flush=True)

samples = {w: [] for w in WIDTHS}
with torch.inference_mode():
    for round_i in range(N_ROUNDS):
        for w in WIDTHS:
            cq = questions_by_width[w]
            torch.cuda.synchronize()
            t = time.perf_counter()
            eng.ask(f"[{uuid.uuid4().hex}]\n" + context, cq)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
            samples[w].append(ms)
            print(f"round {round_i} width {w}: {ms:.2f} ms", flush=True)

kept = {w: samples[w][DISCARD_ROUNDS:] for w in WIDTHS}
summary = {
    "model": MODEL,
    "context_chars": len(context),
    "n_rounds": N_ROUNDS,
    "discard_rounds": DISCARD_ROUNDS,
    "engine_paged": eng.paged,
    "engine_interleaved_fork": eng.interleaved_fork,
    "by_width": {
        str(w): {
            "median_ms": round(statistics.median(kept[w]), 2),
            "min_ms": round(min(kept[w]), 2),
            "max_ms": round(max(kept[w]), 2),
            "spread_pct": round((max(kept[w]) - min(kept[w])) / statistics.median(kept[w]) * 100, 2),
            "n_kept": len(kept[w]),
            "all_samples_ms": [round(x, 2) for x in samples[w]],
        }
        for w in WIDTHS
    },
}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "by_width"}, indent=1), flush=True)
print(json.dumps(summary["by_width"], indent=1), flush=True)
print("wrote", OUT, flush=True)
