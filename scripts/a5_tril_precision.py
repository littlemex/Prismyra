"""A5 (acc), the real measurement: for 50 long documents, capture every real `A` matrix the gated-delta
read path feeds into fla's triangular-inverse solve ((I+A)^-1, strictly-lower-triangular A, chunks of
BT=64), then compare FOUR ways of computing that inverse against a float64 `torch.linalg.inv` reference
computed on the exact same captured matrices:
  (a) num_warps=4, num_stages=5, DOT_PRECISION='ieee'  -- sm_89.json's INTENDED pin for L40S
  (b) num_warps=2, num_stages=2, DOT_PRECISION='ieee'  -- what pin() actually falls back to right now
      (a5_diag_pin.py's finding: fla 0.5.2 added a DOT_PRECISION kwarg to this kernel's autotune
      candidates; sm_89.json's recorded "kwargs": {} no longer matches any candidate's actual kwargs
      dict, so pin() silently uses the autotune list's FIRST candidate instead of the named one)
  (c) num_warps=4, num_stages=5, DOT_PRECISION='tf32'  -- the TF32 case A5 was asked to check; not
      reachable in current production (DOT_PRECISION_AUTOTUNE_LIST is forced to ['ieee'] whenever
      IS_TMA_SUPPORTED is False, true on this L40S and probably on sm_120 too since nothing sets
      FLA_TRIL_PRECISION=tf32 anywhere in this project) but forced here directly to answer the
      question asked, not just to report the premise doesn't currently apply.
  (d) plain torch.float32 matmul-based Gauss-Jordan elimination done in pure PyTorch (sanity check
      that the float64 reference and the kernel's own math agree in shape/convention).

Reports, per document-length octile (50 long docs -> ~6-7 per octile): max |error| and mean |error|
of (a)/(b)/(c) against (d)=float64, and the resulting change in the FINAL read-out probability
(gold-option prob) between (a) and (b) on the real question for each document.
"""
import os, sys, json, glob, copy
os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/work/probes/src")
sys.path.insert(0, "/work/next/scripts")

import torch
import triton
from triton import Config

M = os.environ.get("PIN_DIAG_MODEL", "/work/models/p35-l36_kd025_7k_cap32k")
N_DOCS = int(os.environ.get("A5_N_DOCS", "50"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/next/results/a5_tril_precision.json"

# ---- gather up to N_DOCS long documents from the project's own long-document sources (bury7k/10k +
# JevBench's long octile), longest-context item from each manifest first.
docs = []
for path in ["/work/items/race40_bury7k.json", "/work/items/race40_bury10k.json"]:
    man = json.load(open(path))
    for it in sorted(man["items"], key=lambda x: -len(x.get("context", ""))):
        sub = it["questions"][0]
        docs.append({"context": it["context"], "kind": sub["kind"], "question": sub["question"],
                     "options": sub["options"], "gold": sub["gold_index"], "src": f"{os.path.basename(path)}#{it['item']}"})
jevsel = json.load(open("/work/next/data/jevsel.json"))
for x in sorted(jevsel, key=lambda r: -len(r.get("context", ""))):
    d = dict(x); d["src"] = "jevsel#" + str(x.get("jev_id", "?"))
    docs.append(d)
docs = docs[:N_DOCS]
print(f"using {len(docs)} documents; length range chars: "
      f"{min(len(d['context']) for d in docs)}..{max(len(d['context']) for d in docs)}", flush=True)

from prismyra import Prismyra
import common
from common import question as qfn

eng = Prismyra(M, group=32, pin_autotune=True)

# ---- capture: monkeypatch solve_tril at every module that imported it by name, recording every real
# call's A tensor (cloned to CPU as float32) plus which document index triggered it.
import fla.ops.utils.solve_tril  # noqa: F401 - ensures sys.modules has the real submodule object
st_mod = sys.modules["fla.ops.utils.solve_tril"]  # fla.ops.utils/__init__.py's `from .solve_tril import
# solve_tril` overwrites the `fla.ops.utils.solve_tril` ATTRIBUTE with the function itself, shadowing
# the submodule there -- so grab the submodule from sys.modules directly instead of by attribute access.
orig_solve_tril = st_mod.solve_tril
captured = []
cur_doc = {"i": None}

def capturing_solve_tril(A, *a, **kw):
    if A.shape[-1] == 64:
        captured.append((cur_doc["i"], A.detach().to(torch.float32).cpu().clone()))
    return orig_solve_tril(A, *a, **kw)

patched_modules = []
import importlib, pkgutil
for modname in list(sys.modules):
    if modname.startswith("fla.") and hasattr(sys.modules[modname], "solve_tril"):
        if sys.modules[modname].solve_tril is orig_solve_tril:
            sys.modules[modname].solve_tril = capturing_solve_tril
            patched_modules.append(modname)
st_mod.solve_tril = capturing_solve_tril
print("patched solve_tril in modules:", patched_modules, flush=True)

results = []
with torch.inference_mode():
    for i, d in enumerate(docs):
        cur_doc["i"] = i
        q = qfn(d)
        n_before = len(captured)
        out = eng.ask(d["context"], [q])
        p = common.probs(d, out["q"].probabilities)
        n_after = len(captured)
        results.append({"i": i, "src": d["src"], "chars": len(d["context"]), "gold": d["gold"],
                         "p": p, "gold_p": p[d["gold"]] if d["kind"] != "boolean" else p[int(d["gold"])],
                         "n_tril_calls": n_after - n_before})
        if i % 5 == 0:
            print(f"doc {i}/{len(docs)} chars={len(d['context'])} tril_calls_so_far={len(captured)}", flush=True)

for modname in patched_modules:
    sys.modules[modname].solve_tril = orig_solve_tril
st_mod.solve_tril = orig_solve_tril
print(f"total captured tril calls (BT=64 only): {len(captured)}", flush=True)

# ---- offline precision comparison on every captured A -------------------------------------------------
from fla.ops.utils.solve_tril import merge_16x16_to_64x64_inverse_kernel as K64

def get_autotuner(kernel_obj):
    k = kernel_obj
    while not hasattr(k, "configs") and hasattr(k, "fn"):
        k = k.fn
    return k

tuner = get_autotuner(K64)
base_configs = list(tuner.configs)  # whatever pin() left it as; we override per-call below and restore after

def run_variant(A_gpu, num_warps, num_stages, dot_precision):
    tuner.configs = [Config({"DOT_PRECISION": dot_precision}, num_warps=num_warps, num_stages=num_stages)]
    if isinstance(getattr(tuner, "cache", None), dict):
        tuner.cache.clear()
    return orig_solve_tril(A_gpu).to(torch.float32)

dev = eng.torch_device if hasattr(eng, "torch_device") else "cuda:0"
errs = {"a_w4_ieee": [], "b_w2_ieee": [], "c_w4_tf32": []}
per_doc_err = {}
sample_n = 0
MAX_MATS_PER_DOC = 200  # cap work; still many thousands of 64x64 inversions total across 50 docs
for doc_i, A_cpu in captured:
    if per_doc_err.setdefault(doc_i, 0) >= MAX_MATS_PER_DOC:
        continue
    per_doc_err[doc_i] += 1
    A_gpu = A_cpu.to(dev)
    # float64 reference: (I + A)^-1 computed directly, no Triton, no kernel-specific chunk-merge algebra
    I = torch.eye(A_gpu.shape[-1], dtype=torch.float64, device=dev).expand(*A_gpu.shape[:-1], -1, -1)
    ref64 = torch.linalg.inv(I + A_gpu.double())
    for key, (nw, ns, dp) in {"a_w4_ieee": (4, 5, "ieee"), "b_w2_ieee": (2, 2, "ieee"), "c_w4_tf32": (4, 5, "tf32")}.items():
        out = run_variant(A_gpu, nw, ns, dp)
        err = (out.double() - ref64).abs()
        errs[key].append((float(err.max()), float(err.mean())))
    sample_n += 1

tuner.configs = base_configs
if isinstance(getattr(tuner, "cache", None), dict):
    tuner.cache.clear()

summary = {"n_docs": len(docs), "n_tril_calls_captured": len(captured), "n_matrices_compared": sample_n}
for key, lst in errs.items():
    if not lst:
        summary[key] = None
        continue
    maxs = [x[0] for x in lst]; means = [x[1] for x in lst]
    summary[key] = {"max_abs_err_over_all": max(maxs), "mean_of_max_abs_err": sum(maxs) / len(maxs),
                     "mean_abs_err": sum(means) / len(means)}
summary["per_document_gold_prob"] = results
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_document_gold_prob"}, indent=1), flush=True)
print("wrote", OUT, flush=True)
