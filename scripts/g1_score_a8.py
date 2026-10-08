"""Experiment 4, G1: score the a8 checkpoint on the devset built by g1_build_devset.py.

Usage: g1_score_a8.py DEVSET.json OUT.jsonl
"""
import os, sys, json

os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
from prismyra import Prismyra, Boolean, Choice

MODEL = os.environ.get("G1_MODEL", "/work/next/models/p35-l36-kd_a8_2xrows")
DEVSET, OUT = sys.argv[1], sys.argv[2]
L = "ABCDEFGHIJKLMNOP"

rows = json.load(open(DEVSET))
print(f"{len(rows)} rows", flush=True)

eng = Prismyra(MODEL, require_kernels=True)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        done.add(json.loads(line)["i"])
fo = open(OUT, "a")

n_correct = 0
n_scored = 0
with torch.inference_mode():
    for i, r in enumerate(rows):
        if i in done:
            n_scored += 1
            continue
        if r["kind"] == "boolean":
            q = Boolean(id="q", prompt=r["question"])
        else:
            q = Choice(
                id="q",
                prompt=r["question"] + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(r["options"])),
                choices=list(L[: len(r["options"])]),
            )
        out = eng.ask(r["context"], [q])["q"]
        pred_index = ["no", "yes"].index(out.option) if r["kind"] == "boolean" else L.index(out.option)
        correct = pred_index == r["gold"]
        n_scored += 1
        n_correct += int(correct)
        fo.write(json.dumps({"i": i, "family": r["family"], "gold": r["gold"], "pred": pred_index, "correct": correct}) + "\n")
        if i % 200 == 0:
            fo.flush()
            print(f"row {i}/{len(rows)} running_acc={n_correct / max(1, n_scored - len(done)):.4f}", flush=True)

fo.close()
print(f"DONE n_scored={n_scored}", flush=True)
