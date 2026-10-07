"""A5 (acc): live diagnostic -- does prismyra.kernels.autotune.pin() actually land on the pinned
num_warps for merge_16x16_to_64x64_inverse_kernel on THIS GPU, now that the installed fla package's
version of that kernel autotunes over DOT_PRECISION too (every candidate's kwargs is
{'DOT_PRECISION': 'ieee'} since FLA_TRIL_PRECISION is never set anywhere in this project), while the
pinned sm_<arch>.json files record "kwargs": {} (empty) for this kernel? If matches() requires an exact
kwargs dict match, {'DOT_PRECISION': 'ieee'} != {}, so pin() may be silently falling back to the autotune
list's FIRST declared candidate (num_warps=2, num_stages=2) instead of the intended num_warps=4 (sm_89) /
num_warps=2 (sm_120, which happens to coincide) -- a real determinism regression from a dependency
version drift, not a hypothetical.

Prints: fla version, the kernel's actual autotune config list (first 6), the device's arch, the pinned
table entry for this kernel, and the live Pinned.as_dict() result of actually calling pin() after one
real forward pass has triggered the kernel's import/first use.
"""
import os, sys, json
os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/work/probes/src")
sys.path.insert(0, "/work/next/scripts")

import torch
import fla
print("fla version:", fla.__version__, flush=True)
from fla.ops.utils.solve_tril import merge_16x16_to_64x64_inverse_kernel, DOT_PRECISION_AUTOTUNE_LIST, FLA_TRIL_PRECISION
print("FLA_TRIL_PRECISION:", FLA_TRIL_PRECISION, "DOT_PRECISION_AUTOTUNE_LIST:", DOT_PRECISION_AUTOTUNE_LIST, flush=True)
from fla.utils import IS_TMA_SUPPORTED
print("IS_TMA_SUPPORTED on this GPU:", IS_TMA_SUPPORTED, flush=True)
_k = merge_16x16_to_64x64_inverse_kernel
while not hasattr(_k, "configs") and hasattr(_k, "fn"):
    _k = _k.fn
configs = _k.configs
print(f"merge_16x16_to_64x64_inverse_kernel has {len(configs)} candidate configs; first 6:")
for c in configs[:6]:
    print("  num_warps=", c.num_warps, "num_stages=", c.num_stages, "kwargs=", dict(c.kwargs))
print("  ... last 2:")
for c in configs[-2:]:
    print("  num_warps=", c.num_warps, "num_stages=", c.num_stages, "kwargs=", dict(c.kwargs))

arch = f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
print("device arch:", arch, "name:", torch.cuda.get_device_name(0), flush=True)
pinned_path = f"/work/prismyra/prismyra/kernels/pinned/{arch}.json"
if not os.path.exists(pinned_path):
    # fall back to the installed package copy
    import prismyra.kernels.autotune as pa
    pinned_path = str(pa.PINNED_DIR / f"{arch}.json")
print("pinned file:", pinned_path, "exists:", os.path.exists(pinned_path))
if os.path.exists(pinned_path):
    tab = json.load(open(pinned_path))["kernels"].get("merge_16x16_to_64x64_inverse_kernel")
    print("pinned table entry for this kernel:", tab)

# Trigger one real forward so the Autotuner object exists in the process, matching how the engine
# itself calls pin() right after model construction (see engine.py / n2_fit.py's own first-forward-then-pin pattern).
M = os.environ.get("PIN_DIAG_MODEL", "/work/models/p35-l36_kd025_7k_cap32k")
from prismyra import Prismyra
from common import question, load as cmload
eng = Prismyra(M, group=32, pin_autotune=False)   # pin_autotune=False: we pin manually below to inspect the result
rows = cmload("calib")[:2]
with torch.inference_mode():
    for r in rows:
        eng.ask(r["context"], [question(r)])
print("after warmup forward, autotuner objects found:", flush=True)
import prismyra.kernels.autotune as pa
tuners = pa.autotuners()
for t in tuners:
    name = pa.kernel_name(t)
    if "merge_16x16_to_64x64" in name or "inverse" in name:
        print(" tuner:", name, "n_configs=", len(getattr(t, "configs", [])), "cache keys so far:", list(getattr(t, "cache", {}).keys())[:3])

result = pa.pin(0)
print("pin() result:", json.dumps(result.as_dict(), indent=1))
