"""Task 3 (acc2): one single call's wall time at 1/16/64 questions, same ~5.3k-token document (race150's
first few articles concatenated, same construction as the published model card's own Latency section),
same GPU, one model per process invocation so each run gets a clean process (no cross-model CUDA context
reuse). Run this once per (model, width) combination and interleave the invocations from the shell so the
measurements that get compared are close together in time (cancels GPU/driver warm-up drift).
  a_speed_interleave.py MODEL WIDTH OUT.jsonl
"""
import json, sys, time, uuid
import torch
from prismyra import Prismyra, Choice

MODEL, WIDTH, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
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
