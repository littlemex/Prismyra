"""acc3: interleaved 1/16/64-question speed measurement for NVFP4 checkpoints, same construction as
a_speed_interleave.py (race150's first few articles, ~5300 chars) but with PRISMYRA_EXPERTS=nvfp4 and
the two NVFP4 side files set via env before importing prismyra, so the fused NVFP4 expert path is used.
  a_speed_interleave_fp4.py MODEL EXPERTS_SFT CALIB_JSON WIDTH OUT.jsonl
"""
import json, os, sys, time, uuid

MODEL, EXPERTS_SFT, CALIB_JSON, WIDTH, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
os.environ["PRISMYRA_EXPERTS"] = "nvfp4"
os.environ["PRISMYRA_NVFP4_EXPERTS"] = EXPERTS_SFT
os.environ["PRISMYRA_NVFP4_CALIB"] = CALIB_JSON

import torch
from prismyra import Prismyra, Choice

L = "ABCDEFGH"

data = json.load(open("/work/items/race150.json"))
ctx, qs = [], []
for it in data["items"]:
    ctx.append(it["context"]); qs.extend(it["questions"])
    if sum(map(len, ctx)) >= 5300:
        break
context = "\n\n".join(ctx)
qs = [qs[i % len(qs)] for i in range(WIDTH)]
cq = [Choice(id=f"q{i}", prompt=q["question"] + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(q["options"])),
             choices=list(L[: len(q["options"])])) for i, q in enumerate(qs)]

eng = Prismyra(MODEL, group=32)
with torch.inference_mode():
    for _ in range(2):  # warm up
        eng.ask(f"[{uuid.uuid4().hex}]\n" + context, cq)
    torch.cuda.synchronize()
    t = time.perf_counter()
    eng.ask(f"[{uuid.uuid4().hex}]\n" + context, cq)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000

rec = {"model": MODEL, "width": WIDTH, "ms": round(ms, 2), "ts": time.time()}
with open(OUT, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(json.dumps(rec), flush=True)
