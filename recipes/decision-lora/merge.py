"""Fold a trained LoRA into the FP8 checkpoint ahead of time, so serving pays nothing for it.

Usage:

    python merge.py <base checkpoint dir> lora.pt <output dir>

Then serve the output directory as any other checkpoint: `Prismyra("<output dir>")`.

Only the adapted matrices change. Each is dequantised, W + (alpha/r) B A is added, and it is re-quantised with the
checkpoint's own scheme (e4m3, one scale per 128x128 block, stored as weight_scale_inv). Every other tensor -- the
256 routed experts included -- is copied byte for byte, and config.json is untouched, so the result has the exact
shape Prismyra's MoE kernels check for.
"""

import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

src, lora_path, dst = sys.argv[1], sys.argv[2], sys.argv[3]
BLOCK = 128
FP8 = torch.float8_e4m3fn
FMAX = torch.finfo(FP8).max
ck = torch.load(lora_path, map_location="cpu")
lora = ck["lora"]
scale = ck["alpha"] / ck["rank"]
mods = sorted({k.rsplit(".", 1)[0] for k in lora})  # e.g. layers.3.self_attn.q_proj
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]


def key(m):
    return f"model.language_model.{m}.weight"


by_file = {}
for m in mods:
    by_file.setdefault(idx[key(m)], []).append(m)
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(src, f)):
        shutil.copy2(os.path.join(src, f), dst)
changed = 0
worst = 0.0
for f in sorted({v for v in idx.values()}):
    if f not in by_file:
        if not os.path.exists(os.path.join(dst, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        continue
    t = load_file(os.path.join(src, f))
    for m in by_file[f]:
        w, s = t[key(m)], t.get(key(m) + "_scale_inv")
        O, I = w.shape
        if s is not None and (O % BLOCK or I % BLOCK):
            # The block scale grid would need a padded last block, which this script does not write. Refuse rather
            # than re-quantise a matrix into a layout the loader would read differently.
            raise SystemExit(f"{m}: {O}x{I} is not a multiple of {BLOCK} in both dimensions")
        deq = (
            (w.float().view(O // BLOCK, BLOCK, I // BLOCK, BLOCK) * s.float().view(O // BLOCK, 1, I // BLOCK, 1)).view(
                O, I
            )
            if s is not None
            else w.float()
        )
        delta = (lora[m + ".B"].float() @ lora[m + ".A"].float()) * scale
        new = deq + delta
        if s is None:
            t[key(m)] = new.to(w.dtype)
        else:
            blocks = new.view(O // BLOCK, BLOCK, I // BLOCK, BLOCK)
            amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
            sc = (amax / FMAX).float()
            q = (blocks / sc.view(O // BLOCK, 1, I // BLOCK, 1)).clamp(-FMAX, FMAX).to(FP8).view(O, I)
            back = (q.float().view(O // BLOCK, BLOCK, I // BLOCK, BLOCK) * sc.view(O // BLOCK, 1, I // BLOCK, 1)).view(
                O, I
            )
            worst = max(worst, ((back - new).abs().max() / new.abs().max()).item())
            t[key(m)] = q
            t[key(m) + "_scale_inv"] = sc.to(s.dtype)
        changed += 1
    save_file(t, os.path.join(dst, f), metadata={"format": "pt"})
print(json.dumps({"changed_matrices": changed, "of": len(mods), "worst_relative_requant_error": worst, "out": dst}))
