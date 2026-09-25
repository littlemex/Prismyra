"""Write a trained adapter in its published form: `adapter.safetensors` plus `adapter_config.json`.

Usage:

    python export.py lora.pt <output dir> [base model id]

The tensors are the adapter's `A` and `B` for each adapted projection, named after the module they adapt
(`layers.<i>.<path>.A` / `.B`), in float32 as trained. The config carries what `merge.py` needs -- rank and alpha --
and what a reader needs to know what the adapter is for: the base checkpoint and the adapted modules.
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3.6-35B-A3B-FP8"
ck = torch.load(src, map_location="cpu")
os.makedirs(dst, exist_ok=True)
save_file({k: v.contiguous() for k, v in ck["lora"].items()}, os.path.join(dst, "adapter.safetensors"))
config = {
    "base_model": base,
    "rank": ck["rank"],
    "alpha": ck["alpha"],
    "target_suffixes": ck["targets"],
    "adapted_modules": ck["wrapped"],
    "format": "prismyra-decision-lora: W + (alpha / rank) * B @ A, folded in by merge.py",
}
with open(os.path.join(dst, "adapter_config.json"), "w") as f:
    json.dump(config, f, indent=1)
print(json.dumps({"tensors": len(ck["lora"]), "rank": ck["rank"], "alpha": ck["alpha"], "out": dst}))
