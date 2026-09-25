"""Hidden states Prismyra already computes while reading a context, captured for probing, next to the asked read-out.

For each item: the residual stream at the last context token and its mean over the context, at selected layers,
from the context read (prefill) -- the zero-cost position a probe would read in serving -- and the probability the
read-out gives to one asked yes/no question about the same text, from a branch of that same read.
Usage: collect.py <model> <items.json> <text key> <question> <out.npz>"""

import json, os, sys, numpy as np, torch
from prismyra import Prismyra, Boolean

model, items_path, key, question, out = sys.argv[1:6]
items = json.load(open(items_path))
if isinstance(items, dict):
    items = items["items"]
eng = Prismyra(model, group=4)
eng._check_fits = lambda *a, **k: None
layers = eng.backbone.language_model.layers if hasattr(eng.backbone, "language_model") else eng.backbone.layers
N = len(layers)
PICK = [l for l in (8, 12, 16, 20, 24, 28, 30, 31, 32, 34, 36, 40) if l <= N]
cap = {}


def hook(i):
    def f(mod, inp, outp):
        h = outp[0] if isinstance(outp, tuple) else outp
        if h.shape[0] == 1 and h.shape[1] > 1 and "reading" in cap:  # the context read only
            cap[i] = (h[0, -1].float().cpu(), h[0].float().mean(0).cpu())

    return f


for l in PICK:
    layers[l - 1].register_forward_hook(hook(l))
last, mean, asked = [], [], []
for n, it in enumerate(items):
    text = it[key]
    cap.clear()
    cap["reading"] = True
    with eng.open_context(text) as c:
        cap.pop("reading")
        r = c.ask([Boolean(id="q", prompt=question)])
    last.append(np.stack([cap[l][0].numpy() for l in PICK]).astype(np.float32))
    mean.append(np.stack([cap[l][1].numpy() for l in PICK]).astype(np.float32))
    asked.append(r["q"].probabilities["yes"])
    if n % 200 == 0:
        print(n, flush=True)
ids = np.array([str(it.get("id", i)) for i, it in enumerate(items)])
np.savez(
    out,
    last=np.stack(last),
    mean=np.stack(mean),
    asked=np.array(asked),
    layers=np.array(PICK),
    ids=ids,
    depth=np.array(N),
    hidden_size=np.array(last[0].shape[-1]),
)
print("saved", out, len(items))
