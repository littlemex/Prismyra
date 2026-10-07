"""A2 (acc): score a lock*_rows.json file through the real Prismyra engine on the Spain RTX PRO 4500
pod (NVFP4 native kernels), arm A (plain single read), self-contained (no /work/probes/src dependency
-- that only exists on the Tokyo EFS). Mirrors probes/scripts/evalrows.py's "A" arm and common.py's
question()/probs() logic.
  a2_spain_score.py MODEL ROWS.json OUT.jsonl
"""
import json, sys, time
import torch
from prismyra import Prismyra
from prismyra.schema import Choice, Boolean

MODEL, ROWS, OUT = sys.argv[1:4]
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
