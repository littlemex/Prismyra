"""A1 (AIR SYNTHESIS.md): scan EXISTING eval logs (no GPU) for a per-choice-position bias: does the
model pick position k more or less often than position k is actually gold, aggregated over everything
already measured? Only reads result jsonl files that already exist on disk (nothing is re-run).

Row schema expected (as produced by probes/scripts/evalrows.py and next/*/scripts over the years):
  {"i":..., "gold": <int index>, "p": [float, ...]}      (declared-option-order probabilities)
Rows without both "gold" (int) and "p" (list of >=2 floats, finite, roughly summing to 1) are skipped.
Grouped by n = len(p) (number of options), since a 2-way boolean row and a 5-way aqua row are different
distributions and should not be pooled before checking for a per-position bias within each n.
"""
import glob, json, os, sys, collections

ROOTS = [
    "/Users/akazawt/tmp/smr/next/results",
    "/Users/akazawt/tmp/smr/probes/res",
    "/Users/akazawt/tmp/smr/ceil/results",
    "/Users/akazawt/tmp/smr/ceil/res",
    "/Users/akazawt/tmp/smr/ceil/res_a6b",
]

def iter_rows(path):
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return

def valid(r):
    if not isinstance(r, dict):
        return False
    p = r.get("p"); g = r.get("gold")
    if not isinstance(p, list) or len(p) < 2 or not isinstance(g, (int, bool)):
        return False
    if not (0 <= int(g) < len(p)):
        return False
    try:
        s = sum(float(x) for x in p)
    except Exception:
        return False
    return 0.9 <= s <= 1.1 and all(isinstance(x, (int, float)) for x in p)

def am(p):
    return max(range(len(p)), key=lambda i: p[i])

files = []
for root in ROOTS:
    files += glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
files = sorted(set(f for f in files if os.path.isfile(f) and os.path.getsize(f) < 5 * 10**8))
print(f"scanning {len(files)} files under {len(ROOTS)} roots", file=sys.stderr)

by_n = collections.defaultdict(lambda: {"pred": collections.Counter(), "gold": collections.Counter(), "n_rows": 0, "files": set()})
n_files_used = 0
n_rows_total = 0
for fp in files:
    used_here = False
    for r in iter_rows(fp):
        if not valid(r):
            continue
        p = [float(x) for x in r["p"]]; g = int(r["gold"]); n = len(p)
        a = am(p)
        d = by_n[n]
        d["pred"][a] += 1
        d["gold"][g] += 1
        d["n_rows"] += 1
        d["files"].add(fp)
        used_here = True
        n_rows_total += 1
    if used_here:
        n_files_used += 1

print(f"files with usable rows: {n_files_used} / {len(files)}; total usable rows: {n_rows_total}", file=sys.stderr)

out = {}
for n, d in sorted(by_n.items()):
    rows = d["n_rows"]
    if rows < 200:
        continue
    pred_rate = [d["pred"].get(i, 0) / rows for i in range(n)]
    gold_rate = [d["gold"].get(i, 0) / rows for i in range(n)]
    bias = [round(pred_rate[i] - gold_rate[i], 4) for i in range(n)]
    out[n] = {"n_rows": rows, "n_files": len(d["files"]), "pred_rate": [round(x, 4) for x in pred_rate],
               "gold_rate": [round(x, 4) for x in gold_rate], "pred_minus_gold": bias}
    print(f"n_options={n} rows={rows} files={len(d['files'])}")
    print(f"  pred_rate = {out[n]['pred_rate']}")
    print(f"  gold_rate = {out[n]['gold_rate']}")
    print(f"  pred-gold = {bias}")

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/Users/akazawt/tmp/smr/acc-wt/scripts/a1_letter_bias_result.json", "w"), indent=1)
