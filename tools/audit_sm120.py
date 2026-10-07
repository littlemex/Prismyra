"""inv task item 1: SYNTHESIS Section 0's unmeasured claim -- does nvfp4-36l's companion-invariance guarantee hold
on the RTX PRO 4500 (sm_120, Blackwell)? Dossier 0-2/0-3 say the dispatcher registration that makes the FP8/L40S
checkpoint invariant is skipped on sm_120 (`is_device_capability_family(80)` in `engine._enable_batch_invariance`)
and nothing has replaced it there; 0-4 adds the NVFP4 MoE kernel's own per-M tactic bucketing as a second, separate
candidate cause. This script is the first real measurement of either on this card.

Ground truth: `engine.ask(context, questions)` alone, once per document (never shares a pass with anything else).
Compared against: `open_batch` with companion counts in {1, 2, 3, 8} documents and question counts in
{1, 2, 3, 31, 32, 33} per document (crossing the group=32 boundary on purpose). Comparison is `torch.equal` on the
full probability tensor per question, not just the argmax decision -- the standing rule is bit-identical, not
merely same-answer.

Usage: PRISMYRA_MODE={raw,fix} python3 audit_sm120.py
  raw -- shipped v0.3.1 code, unmodified (what ships today).
  fix -- this branch's sm_120 fix applied (see engine.py/nvfp4.py diffs); only meaningful once that code exists.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import tasks  # noqa: E402
from prismyra import Prismyra  # noqa: E402
from prismyra.schema import PrismyraError  # noqa: E402


def _round_rows(rows: int, cap: int) -> int:
    """Mirrors `prismyra.engine._round_rows` exactly, so this script can predict (not just discover by exception)
    which companion/question combinations round 3's per-document padding will accept."""
    if rows >= cap:
        return cap
    return min(cap, 1 << (max(rows, 1) - 1).bit_length())

MODEL = os.environ.get("PRISMYRA_MODEL")
N_DOCS = int(os.environ.get("PRISMYRA_N_DOCS", "24"))
QCOUNTS = [1, 2, 3, 31, 32, 33]
GROUP_SIZES = [1, 2, 3, 8]

os.environ.setdefault("PRISMYRA_EXPERTS", "nvfp4")

print(f"PRISMYRA_MODE={os.environ.get('PRISMYRA_MODE', 'raw')} model={MODEL}", flush=True)
print(f"torch={torch.__version__} device_cap={torch.cuda.get_device_capability(0)} name={torch.cuda.get_device_name(0)}", flush=True)

items = tasks.load("race", N_DOCS, split="validation", seed=1)
print(f"{len(items)} documents loaded", flush=True)

#: group=64 so a 33-question request (a `_round_rows` 32->64 bucket crossing, deliberately in QCOUNTS) still fits
#: in one `open_batch` pass -- `_answer_batch` refuses a request wider than `group` outright, by design.
engine = Prismyra(MODEL, paged=True, graphs=False, group=64)


def questions_at(item, n):
    """`n` Question objects derived from this item's own questions, cycling if it has fewer than n and renumbering
    `id` so a cycled-around set never collides (`engine.validate` rejects duplicate ids)."""
    import dataclasses

    base = item.questions
    out = []
    for i in range(n):
        q = base[i % len(base)]
        out.append(dataclasses.replace(q, id=f"{q.id}_{i}"))
    return out


# ---- ground truth: ask() alone, one document, one call, never shares a pass ----
truth = {}
for item in items:
    for n in QCOUNTS:
        qs = questions_at(item, n)
        truth[(id(item), n)] = engine.ask(item.context, qs)
print("ask() ground truth done", flush=True)

report = {"model": MODEL, "mode": os.environ.get("PRISMYRA_MODE", "raw"), "n_docs": len(items), "mismatches": []}
total_checks = 0
total_exact = 0

skipped_capacity = []
for group_size in GROUP_SIZES:
    # `_answer_batch` refuses a batch whose *padded* question count across every document in the pass exceeds
    # `self.group` (64 here) -- a real constraint, not a test artefact, and since round 3 (per-document padding,
    # not combined-total padding) it is computed per document and summed, which rejects a few combinations the
    # combined-total version used to accept (predicted here with the same `_round_rows`, not discovered by
    # exception, so a genuine regression in the admission check itself still shows up as an unexpected raise below).
    qcounts_here = [n for n in QCOUNTS if group_size * _round_rows(n, engine.group) <= engine.group]
    for qn in qcounts_here:
        for start in range(0, len(items) - group_size + 1, group_size):
            chunk = items[start : start + group_size]
            try:
                with engine.open_batch([it.context for it in chunk]) as batch:
                    results = batch.ask([questions_at(it, qn) for it in chunk])
            except PrismyraError as e:
                skipped_capacity.append({"group_size": group_size, "qn": qn, "error": str(e)})
                continue
            for item, got in zip(chunk, results, strict=True):
                qs = questions_at(item, qn)
                want = truth[(id(item), qn)]
                for q in qs:
                    total_checks += 1
                    g = got[q.id].probabilities
                    w = want[q.id].probabilities
                    g_t = torch.tensor([g[o] for o in sorted(g)])
                    w_t = torch.tensor([w[o] for o in sorted(w)])
                    exact = torch.equal(g_t, w_t)
                    if exact:
                        total_exact += 1
                    else:
                        move = (g_t - w_t).abs().max().item()
                        decision_flip = got[q.id].option != want[q.id].option
                        report["mismatches"].append(
                            {
                                "group_size": group_size,
                                "qn": qn,
                                "doc": item.context[:50],
                                "qid": q.id,
                                "move": move,
                                "decision_flip": decision_flip,
                            }
                        )

print(f"\ntotal probability checks: {total_checks}, bit-exact: {total_exact}, non-exact: {total_checks - total_exact}")
decision_flips = sum(1 for m in report["mismatches"] if m["decision_flip"])
worst = max((m["move"] for m in report["mismatches"]), default=0.0)
print(f"decision flips: {decision_flips}, worst probability move: {worst:.6f}")

by_group = {}
for m in report["mismatches"]:
    by_group.setdefault(m["group_size"], 0)
    by_group[m["group_size"]] += 1
print(f"non-exact count by companion group_size: {by_group}")

by_qn = {}
for m in report["mismatches"]:
    by_qn.setdefault(m["qn"], 0)
    by_qn[m["qn"]] += 1
print(f"non-exact count by question-count: {by_qn}")

print(f"\nunexpectedly-refused combinations (admission bug if nonzero): {len(skipped_capacity)}")
for s in skipped_capacity[:5]:
    print(f"  group_size={s['group_size']} qn={s['qn']}: {s['error'][:120]}")

report["skipped_capacity"] = skipped_capacity
out_path = Path(os.environ.get("PRISMYRA_REPORT", "/tmp/audit_sm120_report.json"))
out_path.write_text(json.dumps(report, indent=2))
print(f"\nfull report written to {out_path}")
