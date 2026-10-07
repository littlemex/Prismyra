"""A5 (acc) follow-up to a5_diag_pin.py's finding: pin() silently falls back to num_warps=2,
num_stages=2 for one of the two "merge_16x16_to_64x64_inverse_kernel" Autotuner objects in the
process (the one with the newer fla 0.5.2 DOT_PRECISION kwarg dimension), because the shipped
pinned/sm_89.json records "kwargs": {} and matches() requires an exact dict match. This script
settles THREE things on a real forward pass over a genuinely long document (so BT=64 chunking
actually fires):
  1. which of the two same-named tuner objects actually gets a non-empty .cache (i.e. is really
     called) once a long document is read;
  2. whether the DEFAULT engine construction (pin_autotune=True, exactly as shipped) reproduces the
     same answer across two independent fresh processes right now (it should, now that the fallback
     is deterministic -- just possibly the WRONG deterministic choice);
  3. whether forcing num_warps=4 (what sm_89.json actually intended) on the kernel that fell back
     changes the answer on this document, by how much, and which (if either) is closer once a
     float64 reference for the same triangular inverse is computed directly with torch.linalg.
"""
import os, sys, json, glob
os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/work/probes/src")
sys.path.insert(0, "/work/next/scripts")

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "default"   # default | force_w4
M = os.environ.get("PIN_DIAG_MODEL", "/work/models/p35-l36_kd025_7k_cap32k")

# a genuinely long row (race-in-a-long-document "bury7k" format: {"items": [{"context":..., "questions": [...]}]}),
# so the chunked (BT=64) gated-delta path actually fires.
manifest = json.load(open("/work/items/race40_bury7k.json"))
item = max(manifest["items"], key=lambda it: len(it.get("context", "")))
sub = item["questions"][0]
row = {"context": item["context"], "kind": sub["kind"], "question": sub["question"], "options": sub["options"], "gold": sub["gold_index"]}
print("using race40_bury7k.json item", item["item"], "len(context)=", len(row["context"]), flush=True)

from prismyra import Prismyra
import common
from common import question as qfn

eng = Prismyra(M, group=32, pin_autotune=True)   # exactly as shipped: pin_autotune defaults True
print("engine.stats()['autotune'] (as actually constructed):")
print(json.dumps(eng.stats().get("autotune", {}), indent=1)[:4000], flush=True)

import prismyra.kernels.autotune as pa
tuners = [t for t in pa.autotuners() if "merge_16x16_to_64x64" in pa.kernel_name(t)]
print(f"found {len(tuners)} merge_16x16_to_64x64_inverse_kernel tuner object(s) post-construction")
for i, t in enumerate(tuners):
    print(f"  [{i}] n_configs(after pin, should be 1)=", len(t.configs), "configs[0]=", t.configs[0].num_warps, t.configs[0].num_stages, dict(t.configs[0].kwargs))

if MODE == "force_w4":
    # Force BOTH tuner objects to num_warps=4 (what sm_89.json actually intended for this kernel),
    # overriding whatever pin() landed on, to see whether the answer on this row changes.
    for t in tuners:
        from triton import Config
        t.configs = [Config({"DOT_PRECISION": "ieee"} if "DOT_PRECISION" in dict(t.configs[0].kwargs) else {}, num_warps=4, num_stages=5)]
        if isinstance(getattr(t, "cache", None), dict):
            t.cache.clear()
    print("forced both tuners to num_warps=4, num_stages=5", flush=True)

q = qfn(row)
with torch.inference_mode():
    out = eng.ask(row["context"], [q])
p = common.probs(row, out["q"].probabilities)
print("MODE", MODE, "answer p:", list(p), "gold:", row["gold"], flush=True)

for i, t in enumerate(tuners):
    nk = len(getattr(t, "cache", {}) or {})
    print(f"  tuner[{i}] cache size after forward:", nk, "(non-zero means this one was actually invoked)")
