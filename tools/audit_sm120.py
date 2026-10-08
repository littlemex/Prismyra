"""Does nvfp4-36l's companion-invariance guarantee hold on the RTX PRO 4500 (sm_120, Blackwell)? This script is the
real measurement, meant to be run against two engine states: before `engine._enable_batch_invariance` extended its
dispatcher registration past the SM80 family (`is_device_capability_family(80)` used to gate it to that one family
alone, leaving sm_120 with nothing in its place) and after. The NVFP4 MoE kernel's own per-M tactic bucketing is a
second, separate candidate cause, measured by the same comparison.

Ground truth: `engine.ask(context, questions)` alone, once per document (never shares a pass with anything else).
Compared against: `open_batch` with companion counts in {1, 2, 3, 8} documents and question counts in
{1, 2, 3, 31, 32, 33} per document (crossing the group=32 boundary on purpose). Comparison is `torch.equal` on the
full probability tensor per question, not just the argmax decision -- the standing rule is bit-identical, not
merely same-answer.

A second, independent matrix (below the first, its own report section `"paged_vs_joined"`) answers a different
question that this file did not used to ask at all: does the *constructor flag* `paged` move a solo, companion-free
single question's answer, on top of (not instead of) the companion/question-count matrix above? `ask()`'s own
docstring already says the one-pass path (`paged=False`, question count 1, `_ask_in_one_pass`) and the forked path
(`paged=True`, any question count, a branch pass through the page pool) "agree to the bound batching already
allows" -- this section is what measures that bound on both supported cards, rather than leaving it asserted but
unmeasured. See `tests/test_gpu.py`'s `PAGED_VS_JOINED_MOVEMENT` for the regression test this feeds and
`docs/PERFORMANCE.md`'s "Which engine construction built the answer" section for why `paged` is scoped out of that
section's "closed" claim on the strength of this measurement.

Usage: PRISMYRA_MODE={raw,fix} python3 audit_sm120.py
  raw -- the engine state before `_enable_batch_invariance` covered sm_120; only meaningful checked out against that
  code.
  fix -- the engine state after, which is what every release ships today.
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
    # `self.group` (64 here) -- a real constraint, not a test artefact. Padding is computed per document and
    # summed (not from a combined-total padding), which rejects a few combinations a combined-total version
    # would accept (predicted here with the same `_round_rows`, not discovered by exception, so a genuine
    # regression in the admission check itself still shows up as an unexpected raise below).
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

# ---- second matrix: does `paged` itself (a constructor flag, not a companion/question-count axis) move a solo,
# companion-free answer? Flips the *same* engine's `paged` property rather than constructing a second one, so this
# never holds two copies of the checkpoint at once (a second `Prismyra(...)` of either supported checkpoint does not
# fit beside the first on either card this project ships for -- see RUN-srv.md section 3's "same engine" fix for the
# same OOM). `engine.paged = False` / `= True` round-trips bit-identically on its own (closed by
# `_invariance_base_claimed`, `prismyra/engine.py`'s `__init__`) -- what this measures is *not* that round trip, it
# is the one-pass path (`_ask_in_one_pass`, reached only when `paged` is `False` and the question count is 1)
# against the forked path (`_answer`'s branch pass through the page pool, reached at every question count when
# `paged` is `True`) on the *same* document and question.
print("\n--- paged vs joined (same engine, same weights, paged flipped) ---", flush=True)
was_paged = engine.paged
engine.paged = False
joined_truth = {}
for item in items:
    for n in QCOUNTS:
        qs = questions_at(item, n)
        joined_truth[(id(item), n)] = engine.ask(item.context, qs)
engine.paged = was_paged
print("joined (paged=False) truth done", flush=True)

pvj_checks = 0
pvj_exact = 0
pvj_mismatches = []
for item in items:
    for n in QCOUNTS:
        qs = questions_at(item, n)
        joined = joined_truth[(id(item), n)]
        paged = truth[(id(item), n)]
        for q in qs:
            pvj_checks += 1
            g = joined[q.id].probabilities
            w = paged[q.id].probabilities
            g_t = torch.tensor([g[o] for o in sorted(g)])
            w_t = torch.tensor([w[o] for o in sorted(w)])
            exact = torch.equal(g_t, w_t)
            if exact:
                pvj_exact += 1
            else:
                move = (g_t - w_t).abs().max().item()
                decision_flip = joined[q.id].option != paged[q.id].option
                pvj_mismatches.append(
                    {"qn": n, "doc": item.context[:50], "qid": q.id, "move": move, "decision_flip": decision_flip}
                )

pvj_worst = max((m["move"] for m in pvj_mismatches), default=0.0)
pvj_flips = sum(1 for m in pvj_mismatches if m["decision_flip"])
pvj_by_qn = {}
for m in pvj_mismatches:
    pvj_by_qn.setdefault(m["qn"], []).append(m["move"])
print(
    f"paged vs joined: {pvj_checks} checks, {pvj_exact} bit-exact, {pvj_checks - pvj_exact} non-exact, "
    f"{pvj_flips} decision flips, worst move {pvj_worst:.6f}"
)
for qn, moves in sorted(pvj_by_qn.items()):
    print(f"  qn={qn}: {len(moves)} non-exact, worst {max(moves):.6f}, median {sorted(moves)[len(moves) // 2]:.6f}")

report["paged_vs_joined"] = {
    "checks": pvj_checks,
    "exact": pvj_exact,
    "decision_flips": pvj_flips,
    "worst_move": pvj_worst,
    "mismatches": pvj_mismatches,
}

out_path = Path(os.environ.get("PRISMYRA_REPORT", "/tmp/audit_sm120_report.json"))
out_path.write_text(json.dumps(report, indent=2))
print(f"\nfull report written to {out_path}")
