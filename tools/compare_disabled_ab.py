"""Compares two `tools/bench_disabled_ab.py` outputs: median-latency drift and bit-identical probabilities.

python3 tools/compare_disabled_ab.py results_main.json results_mine.json
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    a_path, b_path = sys.argv[1], sys.argv[2]
    with open(a_path) as fh:
        a = json.load(fh)
    with open(b_path) as fh:
        b = json.load(fh)

    print(f"a={a['label']} ({a_path})  b={b['label']} ({b_path})  model={a['model']}")
    ok = True
    for size in sorted(a["results"], key=int):
        ra, rb = a["results"][size], b["results"][size]
        med_a, med_b = ra["median_ms"], rb["median_ms"]
        pct = (med_b - med_a) / med_a * 100
        flag = "" if abs(pct) < 1.0 else "  <-- OVER 1%"
        print(
            f"size={size:>3}  a: median={med_a:8.2f} min={ra['min_ms']:8.2f} max={ra['max_ms']:8.2f}  "
            f"b: median={med_b:8.2f} min={rb['min_ms']:8.2f} max={rb['max_ms']:8.2f}  diff={pct:+6.2f}%{flag}"
        )
        if abs(pct) >= 1.0:
            ok = False
        if ra["answers"] != rb["answers"]:
            ok = False
            print(f"  answers differ at size={size}:")
            for qid in ra["answers"]:
                if ra["answers"].get(qid) != rb["answers"].get(qid):
                    print(f"    {qid}: a={ra['answers'].get(qid)} b={rb['answers'].get(qid)}")
        else:
            print(f"  answers bit-identical (post-rounding) at size={size}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
