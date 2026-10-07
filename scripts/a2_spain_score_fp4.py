"""A2/A4 (acc2): score a lock*_rows.json file through the real Prismyra engine on the Spain RTX PRO 4500
pod, NVFP4 routed experts (PRISMYRA_EXPERTS=nvfp4). Same arm-A rendering as a2_spain_score.py (FP8 version)
and probes/scripts/evalrows.py's "A" arm / common.py's question()/probs() logic, so FP8 and FP4 results
are directly comparable row-for-row.
  a2_spain_score_fp4.py MODEL CALIB_JSON EXPERTS.safetensors ROWS.json OUT.jsonl
"""
import json, os, sys, time

MODEL, CALIB_JSON, EXPERTS_SFT, ROWS, OUT = sys.argv[1:6]
os.environ["PRISMYRA_EXPERTS"] = "nvfp4"
os.environ["PRISMYRA_NVFP4_CALIB"] = CALIB_JSON
os.environ["PRISMYRA_NVFP4_EXPERTS"] = EXPERTS_SFT

import torch
from prismyra import Prismyra
from prismyra.schema import Choice, Boolean

L = "ABCDEFGHIJKLMNOP"

def question(x):
    if x["kind"] == "boolean":
        return Boolean(id="q", prompt=x["question"])
    return Choice(id="q", prompt=x["question"] + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(x["options"])),
                  choices=list(L[: len(x["options"])]))

def probs(x, r):
    n = x.get("n_opt", len(x.get("options", [])))
    return [r["no"], r["yes"]] if x["kind"] == "boolean" else [r[L[j]] for j in range(n)]

def pick(p):
    return max(range(len(p)), key=p.__getitem__)

eng = Prismyra(MODEL, group=32)
X = json.load(open(ROWS))
done = set()
try:
    for l in open(OUT):
        done.add(json.loads(l)["i"])
except FileNotFoundError:
    pass
f = open(OUT, "a")
t0 = time.time()
with torch.inference_mode():
    for i, x in enumerate(X):
        if i in done:
            continue
        q = question(x)
        rec = {"i": i, "family": x.get("family"), "gold": x["gold"]}
        try:
            t = time.perf_counter()
            out = eng.ask(x["context"], [q])
            ms = round((time.perf_counter() - t) * 1000, 2)
            p = probs(x, out["q"].probabilities)
            rec.update(p=p, ok=pick(p) == x["gold"], ms=ms)
        except Exception as e:
            rec["error"] = repr(e)[:300]
        f.write(json.dumps(rec) + "\n"); f.flush()
        if i % 200 == 0:
            print(i, len(X), f"{time.time()-t0:.0f}s", flush=True)
print("done", len(X), f"{time.time()-t0:.0f}s", flush=True)
