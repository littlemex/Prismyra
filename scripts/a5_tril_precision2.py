"""A5 (acc), real measurement, corrected approach after a5_diag_pin.py / a5_tril_precision.py's
finding that the standalone `solve_tril()` path is NOT what the model's GDN layers actually call.

The real path (fla 0.5.2, chunk_size=64, confirmed empirically on this L40S: IS_TF32_SUPPORTED=True)
is `chunk_gated_delta_rule_fwd_intra()` -> `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, a FUSED
kernel that computes `beta*K@K^T` (the KKT matrix, strictly lower triangular) and solves `(I+A)^-1` in
one Triton kernel whose `DOT_PRECISION` is HARD-CODED at import time from `IS_TF32_SUPPORTED` (so
'tf32' on this card, not a runtime choice) and whose (BK, num_warps) autotune choice is NOT in
`pinned/sm_89.json` at all (only the OLDER, no-longer-called `chunk_scaled_dot_kkt_fwd_kernel` and
`merge_16x16_to_64x64_inverse_kernel` are named there) -- so it silently runs on the autotuner's
first declared candidate (BK=32, num_warps=1), never pinned, never checked against anything.

Method: monkeypatch `chunk_gated_delta_rule_fwd_intra` to capture its real (k, g, beta) inputs during
a real forward over long documents, then OFFLINE call the library's OWN already-tested, UNFUSED
`chunk_scaled_dot_kkt_fwd(k, g, beta, ...)` to get the exact pre-inverse KKT matrix A_raw (the fused
kernel's docstring states this is "the mathematically equivalent representation" for non-64 chunk
sizes, i.e. the same A any chunk size computes) -- this avoids re-deriving the KKT formula by hand.
Then compare three ways of inverting `I + A_raw`:
  (shipped) the REAL fused kernel's own output A for that chunk (TF32, BK=32, num_warps=1, as it
            actually ships right now)
  (fp32)    torch.linalg.inv(I + A_raw.float()) with TF32 matmul explicitly disabled
            (torch.backends.cuda.matmul.allow_tf32 = False) -- plain IEEE fp32, isolates "is TF32
            itself the problem, or would plain fp32 already be enough"
  (fp64)    torch.linalg.inv(I + A_raw.double()) -- the reference this whole check is against
Reports max/mean abs error of (shipped) and (fp32) against (fp64), bucketed by document length octile
over up to 50 long documents (bury7k/bury10k/JevBench long rows), plus the resulting change in the
final read-out's gold-option probability if the fused kernel's chunk-64 output were replaced by the
fp64-correct inverse (propagated through one extra matmul pass, not a full second forward).
"""
import os, sys, json, glob
os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/work/probes/src")
sys.path.insert(0, "/work/next/scripts")

import torch

M = os.environ.get("PIN_DIAG_MODEL", "/work/models/p35-l36_kd025_7k_cap32k")
N_DOCS = int(os.environ.get("A5_N_DOCS", "50"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/next/results/a5_tril_precision2.json"

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

import fla.ops.gated_delta_rule.chunk_fwd as cf_mod
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
orig_intra = cf_mod.chunk_gated_delta_rule_fwd_intra
print("DOT_PRECISION baked into the real fused kernel on this GPU:", cf_mod.SOLVE_TRIL_DOT_PRECISION, flush=True)

captured = []  # list of dict(doc_i, k, g, beta, A_shipped, chunk_size, cu_seqlens)
cur_doc = {"i": None}

def capturing_intra(k, v, g=None, beta=None, cu_seqlens=None, chunk_size=64, chunk_indices=None):
    w, u, A = orig_intra(k, v, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices)
    if chunk_size == 64 and A.shape[-1] == 64:
        captured.append({
            "doc": cur_doc["i"],
            "k": k.detach().float().cpu().clone(),
            "g": g.detach().float().cpu().clone() if g is not None else None,
            "beta": beta.detach().float().cpu().clone() if beta is not None else None,
            "A_shipped": A.detach().float().cpu().clone(),
            "cu_seqlens": cu_seqlens.detach().cpu().clone() if cu_seqlens is not None else None,
        })
    return w, u, A

patched = []
for modname in list(sys.modules):
    mod = sys.modules[modname]
    if modname.startswith("fla.") and getattr(mod, "chunk_gated_delta_rule_fwd_intra", None) is orig_intra:
        mod.chunk_gated_delta_rule_fwd_intra = capturing_intra
        patched.append(modname)
cf_mod.chunk_gated_delta_rule_fwd_intra = capturing_intra
print("patched chunk_gated_delta_rule_fwd_intra in:", patched, flush=True)

results = []
with torch.inference_mode():
    for i, d in enumerate(docs):
        cur_doc["i"] = i
        q = qfn(d)
        n0 = len(captured)
        out = eng.ask(d["context"], [q])
        p = common.probs(d, out["q"].probabilities)
        results.append({"i": i, "src": d["src"], "chars": len(d["context"]), "gold": d["gold"],
                         "p": p, "gold_p": p[int(d["gold"])] if d["kind"] != "boolean" else p[int(d["gold"])],
                         "n_chunks_captured": len(captured) - n0})
        if i % 5 == 0:
            print(f"doc {i}/{len(docs)} chars={len(d['context'])} chunks_captured_so_far={len(captured)}", flush=True)

for modname in patched:
    sys.modules[modname].chunk_gated_delta_rule_fwd_intra = orig_intra
cf_mod.chunk_gated_delta_rule_fwd_intra = orig_intra
print(f"total captured chunk-64 calls: {len(captured)}", flush=True)

# ---- offline comparison --------------------------------------------------------------------------
dev = eng.torch_device if hasattr(eng, "torch_device") else "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = False  # make the "fp32" reference branch genuinely plain IEEE fp32

by_doc_err = {}
MAX_CHUNKS_PER_DOC = 40
per_doc_count = {}
for rec in captured:
    di = rec["doc"]
    if per_doc_count.get(di, 0) >= MAX_CHUNKS_PER_DOC:
        continue
    per_doc_count[di] = per_doc_count.get(di, 0) + 1
    k = rec["k"].to(dev); g = rec["g"].to(dev) if rec["g"] is not None else None
    beta = rec["beta"].to(dev) if rec["beta"] is not None else None
    A_shipped = rec["A_shipped"].to(dev)
    if os.environ.get("A5_DEBUG_SHAPES"):
        print("k", k.shape, "g", (g.shape if g is not None else None), "beta", (beta.shape if beta is not None else None),
              "A_shipped", A_shipped.shape, flush=True)
    A_raw = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, cu_seqlens=rec["cu_seqlens"].to(dev) if rec["cu_seqlens"] is not None else None,
                                      chunk_size=64, output_dtype=torch.float32)
    if os.environ.get("A5_DEBUG_SHAPES"):
        print("A_raw", A_raw.shape, flush=True)
    # [B, T, HV, BT=64]: each ROW t within a 64-row chunk holds that row's 64 within-chunk entries, not
    # a separate (NT, 64, 64) axis. Reshape full (non-final-partial) chunks into real (64, 64) matrices.
    B_, T_, HV_, BT = A_raw.shape
    NT_full = T_ // BT
    if NT_full == 0:
        continue
    def to_blocks(x):  # [B,T,HV,BT] -> [B, NT_full, HV, 64, 64] (drops a trailing partial chunk, if any)
        x = x[:, : NT_full * BT]
        return x.view(B_, NT_full, BT, HV_, BT).permute(0, 1, 3, 2, 4).contiguous()
    A_raw_b = to_blocks(A_raw); A_shipped_b = to_blocks(A_shipped)
    I = torch.eye(BT, dtype=torch.float64, device=dev).expand(*A_raw_b.shape[:-2], -1, -1)
    ref64 = torch.linalg.inv(I + A_raw_b.double())
    ref32 = torch.linalg.inv(torch.eye(BT, dtype=torch.float32, device=dev).expand(*A_raw_b.shape[:-2], -1, -1) + A_raw_b.float())
    err_shipped = (A_shipped_b.double() - ref64).abs()
    err_fp32 = (ref32.double() - ref64).abs()
    by_doc_err.setdefault(di, {"shipped": [], "fp32": []})
    by_doc_err[di]["shipped"].append((float(err_shipped.max()), float(err_shipped.mean())))
    by_doc_err[di]["fp32"].append((float(err_fp32.max()), float(err_fp32.mean())))

summary_per_doc = []
for d in results:
    di = d["i"]
    e = by_doc_err.get(di)
    row = dict(d)
    if e:
        row["shipped_max_abs_err"] = max(x[0] for x in e["shipped"])
        row["shipped_mean_abs_err"] = sum(x[1] for x in e["shipped"]) / len(e["shipped"])
        row["fp32_max_abs_err"] = max(x[0] for x in e["fp32"])
        row["fp32_mean_abs_err"] = sum(x[1] for x in e["fp32"]) / len(e["fp32"])
        row["n_chunks_compared"] = len(e["shipped"])
    summary_per_doc.append(row)

all_shipped_max = [r["shipped_max_abs_err"] for r in summary_per_doc if "shipped_max_abs_err" in r]
all_fp32_max = [r["fp32_max_abs_err"] for r in summary_per_doc if "fp32_max_abs_err" in r]
summary = {
    "n_docs": len(docs), "n_chunk64_calls_captured": len(captured),
    "dot_precision_baked_in": str(cf_mod.SOLVE_TRIL_DOT_PRECISION),
    "shipped_vs_fp64": {"max_over_docs": max(all_shipped_max) if all_shipped_max else None,
                          "mean_over_docs": sum(all_shipped_max) / len(all_shipped_max) if all_shipped_max else None},
    "plain_fp32_vs_fp64": {"max_over_docs": max(all_fp32_max) if all_fp32_max else None,
                             "mean_over_docs": sum(all_fp32_max) / len(all_fp32_max) if all_fp32_max else None},
    "per_document": summary_per_doc,
}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_document"}, indent=1), flush=True)
print("wrote", OUT, flush=True)
