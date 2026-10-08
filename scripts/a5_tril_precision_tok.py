"""Experiment 1 (tok, SYNTHESIS-v2.md 0-1/1): the final A5 check, on the kernel actually used by the
shipped engine.

0-1 found that prismyra's real GDN path never imports bare `fla`; `_borrowed_delta()` only ever does
`importlib.import_module("vllm.third_party.flash_linear_attention.ops.chunk")` and takes
`chunk_gated_delta_rule` off that module. That function (`ops/chunk.py:chunk_gated_delta_rule_fwd`) calls
`solve_tril(A=..., output_dtype=k.dtype)` (`ops/solve_tril.py`), which dispatches to
`merge_16x16_to_64x64_inverse_kernel` for BT=64 -- the kernel already named in `pinned/sm_89.json` /
`pinned/sm_120.json`. `solve_tril.py`'s own `FLA_TRIL_PRECISION` is read once from the environment at
import time, default `"ieee"`; this script does not set that variable, so it measures whatever a caller
who does nothing special gets.

Unlike the predecessor script (a5_tril_precision2.py), which patched the *fused* `fla.ops.gated_delta_
rule.chunk_fwd.chunk_gated_delta_rule_fwd_intra` (a function that does not exist in vLLM's vendored copy,
and so is never on the real forward path), this one patches `solve_tril` itself, in the module namespace
`chunk_gated_delta_rule_fwd` actually calls it through. That also removes a step: `solve_tril`'s own `A`
argument already *is* the raw KKT matrix after `chunk_scaled_dot_kkt_fwd` + the gate cumsum, so there is no
need to recompute it offline -- it is captured for free from the real call.

Pre-registered rule (BRIEF-COMMON.md rule 3, SYNTHESIS-v2.md experiment 1): 50 long documents
(race40_bury7k + race40_bury10k longest items, plus JevBench's longest rows), compare the real shipped
output (ieee, output_dtype=k.dtype, i.e. whatever the model's own working dtype is) against a float64
reference inverse of the same captured `I + A`. Max abs error < 1e-5 over all 50 docs closes A5 ("no
problem, under threshold"). Max abs error > 1e-4 is reported to the chair immediately (a real issue in the
shipped kernel). No lock9/lock10 used (this is a numerical measurement, not an accuracy judgement).
"""
import os, sys, json

os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/work/next/scripts")

import torch

M = os.environ.get(
    "A5_MODEL", "/work/next/models/p35-l36-kd_a8_2xrows"
)  # the fp8-36l a8 checkpoint actually live on HF (9d7cd0bc) -- same depth/same GDN layers as any other 36l checkpoint
N_DOCS = int(os.environ.get("A5_N_DOCS", "50"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/tok/results/a5_tril_precision_tok.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

docs = []
for path in ["/work/items/race40_bury7k.json", "/work/items/race40_bury10k.json"]:
    man = json.load(open(path))
    for it in sorted(man["items"], key=lambda x: -len(x.get("context", ""))):
        sub = it["questions"][0]
        docs.append(
            {
                "context": it["context"],
                "kind": sub["kind"],
                "question": sub["question"],
                "options": sub["options"],
                "gold": sub["gold_index"],
                "src": f"{os.path.basename(path)}#{it['item']}",
            }
        )
jevsel = json.load(open("/work/next/data/jevsel.json"))
for x in sorted(jevsel, key=lambda r: -len(r.get("context", ""))):
    d = dict(x)
    d["src"] = "jevsel#" + str(x.get("jev_id", "?"))
    docs.append(d)
docs = docs[:N_DOCS]
print(
    f"using {len(docs)} documents; length range chars: "
    f"{min(len(d['context']) for d in docs)}..{max(len(d['context']) for d in docs)}",
    flush=True,
)

from prismyra import Prismyra
import common
from common import question as qfn

# Pure v0.4.0 default construction (rule 11): no paged=, no graphs=, no env overrides. require_kernels=True
# only so a silent fallback to the unfused PyTorch path (which would make this whole measurement moot)
# fails loudly instead of quietly.
eng = Prismyra(M, require_kernels=True)

import importlib

chunk_mod = importlib.import_module("vllm.third_party.flash_linear_attention.ops.chunk")
solve_tril_mod = importlib.import_module("vllm.third_party.flash_linear_attention.ops.solve_tril")
print("FLA_TRIL_PRECISION baked in at import time:", solve_tril_mod.FLA_TRIL_PRECISION, flush=True)
assert solve_tril_mod.FLA_TRIL_PRECISION == "ieee", (
    "this run set FLA_TRIL_PRECISION away from the shipped default -- that is a different experiment"
)
orig_solve_tril = chunk_mod.solve_tril

captured = []  # list of (A, Ai_shipped) GPU tensor pairs for the document currently being asked about -- cleared
                # after every document (see errors_for_doc() below), so this is at most one document's worth
cur_doc = {"i": None}


# tok's own fix on top of acc's original design: acc's capture-everything-then-compare-offline approach
# (which this script otherwise follows) holds every captured [B,T,H,64] tensor for every GDN layer, every
# document, in memory at once -- fine on next2/next3 (300Gi RAM, 4 GPUs), but this experiment is assigned
# to `tput` (50Gi RAM, 1 GPU, BRIEF-COMMON.md rule 5), where a 29k-41k-char document's own [1, ~8000-10000,
# 16, 64] tensor is already ~40-50MB and there are ~30 GDN layers per document -- holding all 50 documents'
# worth at once OOM-killed the pod on the first attempt (see RUN-tok.md). So: compare and discard per
# document, immediately after that document's `ask()`, on the GPU (no `.cpu()` round-trip needed either --
# the reference inverse is tiny compute, not a memory problem), and keep only the resulting scalars plus
# at most MAX_CHUNKS_PER_DOC capped captures per document (matching acc's own analysis-time cap, just
# applied before the memory is spent rather than after).
MAX_CHUNKS_PER_DOC = 40
torch.backends.cuda.matmul.allow_tf32 = False  # makes the "fp32" reference branch genuinely plain IEEE fp32
dev = eng.torch_device


def capturing_solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float):
    Ai = orig_solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=output_dtype)
    if A.shape[-1] == 64 and len(captured) < MAX_CHUNKS_PER_DOC:
        captured.append((A.detach().float(), Ai.detach().float()))  # stays on GPU, no clone to host
    return Ai


chunk_mod.solve_tril = capturing_solve_tril


def to_blocks(x, B_, NT_full, H_, BT):  # [B,T,H,BT] -> [B, NT_full, H, 64, 64] (drops a trailing partial chunk)
    x = x[:, : NT_full * BT]
    return x.view(B_, NT_full, BT, H_, BT).permute(0, 1, 3, 2, 4).contiguous()


def errors_for_doc():
    """Consume `captured` (this document's chunks only) and return (shipped_max, shipped_mean, fp32_max,
    fp32_mean) over all of them, then free the tensors."""
    shipped_pairs, fp32_pairs = [], []
    for A, Ai_shipped in captured:
        B_, T_, H_, BT = A.shape
        NT_full = T_ // BT
        if NT_full == 0:
            continue
        A_b = to_blocks(A, B_, NT_full, H_, BT)
        Ai_shipped_b = to_blocks(Ai_shipped, B_, NT_full, H_, BT)
        I = torch.eye(BT, dtype=torch.float64, device=dev).expand(*A_b.shape[:-2], -1, -1)
        ref64 = torch.linalg.inv(I + A_b.double())
        ref32 = torch.linalg.inv(
            torch.eye(BT, dtype=torch.float32, device=dev).expand(*A_b.shape[:-2], -1, -1) + A_b.float()
        )
        err_shipped = (Ai_shipped_b.double() - ref64).abs()
        err_fp32 = (ref32.double() - ref64).abs()
        shipped_pairs.append((float(err_shipped.max()), float(err_shipped.mean())))
        fp32_pairs.append((float(err_fp32.max()), float(err_fp32.mean())))
        del A_b, Ai_shipped_b, I, ref64, ref32, err_shipped, err_fp32
    captured.clear()
    if not shipped_pairs:
        return None
    return {
        "shipped_max_abs_err": max(x[0] for x in shipped_pairs),
        "shipped_mean_abs_err": sum(x[1] for x in shipped_pairs) / len(shipped_pairs),
        "fp32_max_abs_err": max(x[0] for x in fp32_pairs),
        "fp32_mean_abs_err": sum(x[1] for x in fp32_pairs) / len(fp32_pairs),
        "n_chunks_compared": len(shipped_pairs),
    }


summary_per_doc = []
with torch.inference_mode():
    for i, d in enumerate(docs):
        cur_doc["i"] = i
        q = qfn(d)
        captured.clear()
        out = eng.ask(d["context"], [q])
        p = common.probs(d, out["q"].probabilities)
        n_captured_this_doc = len(captured)
        e = errors_for_doc()  # consumes and frees `captured`
        row = {
            "i": i,
            "src": d["src"],
            "chars": len(d["context"]),
            "gold": d["gold"],
            "p": p,
            "n_chunks_captured": n_captured_this_doc,
        }
        if e:
            row.update(e)
        summary_per_doc.append(row)
        torch.cuda.empty_cache()
        if i % 5 == 0:
            print(f"doc {i}/{len(docs)} chars={len(d['context'])} chunks_captured={n_captured_this_doc}", flush=True)

chunk_mod.solve_tril = orig_solve_tril
n_total_captured = sum(r.get("n_chunks_captured", 0) for r in summary_per_doc)
print(f"total captured BT=64 solve_tril calls (capped at {MAX_CHUNKS_PER_DOC}/doc): {n_total_captured}", flush=True)
if n_total_captured == 0:
    raise SystemExit(
        "0 chunks captured -- the real forward never hit a BT=64 solve_tril call; investigate before trusting any number below"
    )

all_shipped_max = [r["shipped_max_abs_err"] for r in summary_per_doc if "shipped_max_abs_err" in r]
all_fp32_max = [r["fp32_max_abs_err"] for r in summary_per_doc if "fp32_max_abs_err" in r]
overall_max = max(all_shipped_max) if all_shipped_max else None
summary = {
    "n_docs": len(docs),
    "n_bt64_calls_captured": n_total_captured,
    "fla_tril_precision": str(solve_tril_mod.FLA_TRIL_PRECISION),
    "shipped_vs_fp64": {
        "max_over_docs": overall_max,
        "mean_over_docs": sum(all_shipped_max) / len(all_shipped_max) if all_shipped_max else None,
    },
    "plain_fp32_vs_fp64": {
        "max_over_docs": max(all_fp32_max) if all_fp32_max else None,
        "mean_over_docs": sum(all_fp32_max) / len(all_fp32_max) if all_fp32_max else None,
    },
    "prereg_verdict": (
        "A5_CLOSED_under_1e-5" if overall_max is not None and overall_max < 1e-5
        else "ESCALATE_over_1e-4" if overall_max is not None and overall_max > 1e-4
        else "between_1e-5_and_1e-4_see_chair_note" if overall_max is not None
        else "no_data"
    ),
    "per_document": summary_per_doc,
}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_document"}, indent=1), flush=True)
print("wrote", OUT, flush=True)
