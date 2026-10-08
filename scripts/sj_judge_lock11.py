"""sj: judge sj1/sj2 (and sj3 if present) against a8 on lock11, using the project's existing paired
sign-test tool (next/ana.py's compare(), the same one acc used for lock9/lock10/lock5), then apply a
Holm correction across the (<=3) candidate p-values. Pre-registered in RUN-sj.md section 1 "合格の決まり":
Holm-corrected sign test vs a8, winner = largest effect size among those that pass Holm.

  sj_judge_lock11.py A8.jsonl SJ1.jsonl [SJ2.jsonl] [SJ3.jsonl]

Each file is one JSON object per line (as produced by sj_score_lock11.py): {"i", "family", "gold", "p",
"ok", "ms"} or {"i", "error": ...}. Rows with an "error" key, or whose index errored in ANY of the
files being compared, are excluded from the paired comparison for THAT pair (same convention acc used:
"1問... 全ての構成で測れず" rows get dropped, n is reported so the exclusion is visible).
"""
import json, sys
sys.path.insert(0, "/Users/akazawt/tmp/smr/next")
from ana import compare

def load(p):
    rows = {}
    for l in open(p):
        r = json.loads(l)
        rows[r["i"]] = r
    return rows

def paired(ref_map, new_map):
    idx = sorted(set(ref_map) & set(new_map))
    ref, new = [], []
    for i in idx:
        a, b = ref_map[i], new_map[i]
        if "error" in a or "error" in b:
            continue
        ref.append(a); new.append(b)
    return ref, new

def holm(pvals, alpha=0.05):
    """Returns a list of booleans (same order as pvals) marking which hypotheses pass Holm's step-down
    procedure at the given alpha. Standard Holm: sort ascending, compare the k-th smallest to
    alpha/(m-k+1); stop (reject none further) at the first failure."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    passed = [False] * m
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        if pvals[i] <= thresh:
            passed[i] = True
        else:
            break  # step-down: once one fails, no later (larger-p) one can pass either
    return passed

a8_path = sys.argv[1]
cand_paths = sys.argv[2:]
a8 = load(a8_path)
results = []
for cp in cand_paths:
    cand = load(cp)
    ref, new = paired(a8, cand)
    r = compare(ref, new)
    results.append((cp, r))

pvals = [r["sign_p"] for _, r in results]
passed = holm(pvals, alpha=0.05)

print(json.dumps({"n_candidates": len(results), "alpha": 0.05}, indent=1))
for (cp, r), ok in zip(results, passed):
    print(json.dumps({"candidate": cp, **r, "holm_pass": ok}, indent=1))

winners = [(cp, r) for (cp, r), ok in zip(results, passed) if ok and r["delta"] > 0]
if winners:
    best = max(winners, key=lambda cr: cr[1]["delta"])
    print(json.dumps({"winner": best[0], "delta": best[1]["delta"], "sign_p": best[1]["sign_p"]}, indent=1))
else:
    print(json.dumps({"winner": None, "reason": "no candidate passed Holm with delta>0"}, indent=1))
