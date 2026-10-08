"""Follow-up to a5_tril_precision_tok.py's headline number (shipped ieee kernel vs fp64: max abs err
0.00195, over the 1e-4 escalation line). Before reporting that to the chair as "the ieee Triton dot is
imprecise", decompose where the 0.00195 actually comes from: `solve_tril()`'s own `output_dtype` defaults
to `k.dtype` (bfloat16, this model's working dtype -- `engine.py`'s `self.dtype`), so the real shipped
`Ai` is rounded to bf16 regardless of how precise the internal "ieee" `tl.dot` computation was. This
script calls the *same* kernel on the *same* captured `A`, once more with `output_dtype=torch.float32`
(no extra forward pass -- this is an offline re-run of the kernel on inputs already captured from the real
forward), to see whether the 0.00195 is (a) the bf16 output rounding (expected, same noise floor as every
other number in this model) or (b) the "ieee" `tl.dot` computation itself being imprecise (a real
problem, independent of output dtype). Smaller N (10 long docs, not 50) -- this is a diagnostic, not the
pre-registered measurement, which a5_tril_precision_tok.py already closed.
"""
import os, sys, json

os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/work/next/scripts")

import torch

M = os.environ.get("A5_MODEL", "/work/next/models/p35-l36-kd_a8_2xrows")
N_DOCS = int(os.environ.get("A5_N_DOCS", "10"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/tok/results/a5_tril_precision_tok_decompose.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

docs = []
for path in ["/work/items/race40_bury7k.json", "/work/items/race40_bury10k.json"]:
    man = json.load(open(path))
    for it in sorted(man["items"], key=lambda x: -len(x.get("context", ""))):
        sub = it["questions"][0]
        docs.append({"context": it["context"], "kind": sub["kind"], "question": sub["question"],
                     "options": sub["options"], "gold": sub["gold_index"], "src": f"{os.path.basename(path)}#{it['item']}"})
docs = docs[:N_DOCS]

from prismyra import Prismyra
import common
from common import question as qfn

eng = Prismyra(M, require_kernels=True)

import importlib
chunk_mod = importlib.import_module("vllm.third_party.flash_linear_attention.ops.chunk")
solve_tril_mod = importlib.import_module("vllm.third_party.flash_linear_attention.ops.solve_tril")
orig_solve_tril = chunk_mod.solve_tril

captured = []
cur_doc = {"i": None}
MAX_CHUNKS_PER_DOC = 40


def capturing_solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float):
    Ai = orig_solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=output_dtype)
    if A.shape[-1] == 64 and len(captured) < MAX_CHUNKS_PER_DOC:
        # Re-run the SAME kernel on the SAME real A, asking for fp32 output instead of bf16, purely to see
        # where the rounding happens. This re-uses the real captured A -- it does not change what the real
        # forward pass sees or answers (Ai, the real bf16 result, is what gets returned to the caller).
        Ai_fp32out = orig_solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=torch.float32)
        captured.append((A.detach().float(), Ai.detach().float(), Ai_fp32out.detach().float()))
    return Ai


chunk_mod.solve_tril = capturing_solve_tril

torch.backends.cuda.matmul.allow_tf32 = False
dev = eng.torch_device


def to_blocks(x, B_, NT_full, H_, BT):
    x = x[:, : NT_full * BT]
    return x.view(B_, NT_full, BT, H_, BT).permute(0, 1, 3, 2, 4).contiguous()


def errors_for_doc():
    bf16out_pairs, fp32out_pairs = [], []
    for A, Ai_bf16out, Ai_fp32out in captured:
        B_, T_, H_, BT = A.shape
        NT_full = T_ // BT
        if NT_full == 0:
            continue
        A_b = to_blocks(A, B_, NT_full, H_, BT)
        Ai_bf16out_b = to_blocks(Ai_bf16out, B_, NT_full, H_, BT)
        Ai_fp32out_b = to_blocks(Ai_fp32out, B_, NT_full, H_, BT)
        I = torch.eye(BT, dtype=torch.float64, device=dev).expand(*A_b.shape[:-2], -1, -1)
        ref64 = torch.linalg.inv(I + A_b.double())
        err_bf16out = (Ai_bf16out_b.double() - ref64).abs()
        err_fp32out = (Ai_fp32out_b.double() - ref64).abs()
        bf16out_pairs.append((float(err_bf16out.max()), float(err_bf16out.mean())))
        fp32out_pairs.append((float(err_fp32out.max()), float(err_fp32out.mean())))
        del A_b, Ai_bf16out_b, Ai_fp32out_b, I, ref64, err_bf16out, err_fp32out
    captured.clear()
    if not bf16out_pairs:
        return None
    return {
        "shipped_bf16out_max_abs_err": max(x[0] for x in bf16out_pairs),
        "shipped_bf16out_mean_abs_err": sum(x[1] for x in bf16out_pairs) / len(bf16out_pairs),
        "same_kernel_fp32out_max_abs_err": max(x[0] for x in fp32out_pairs),
        "same_kernel_fp32out_mean_abs_err": sum(x[1] for x in fp32out_pairs) / len(fp32out_pairs),
        "n_chunks_compared": len(bf16out_pairs),
    }


summary_per_doc = []
with torch.inference_mode():
    for i, d in enumerate(docs):
        cur_doc["i"] = i
        q = qfn(d)
        captured.clear()
        out = eng.ask(d["context"], [q])
        n_captured = len(captured)
        e = errors_for_doc()
        row = {"i": i, "src": d["src"], "chars": len(d["context"]), "n_chunks_captured": n_captured}
        if e:
            row.update(e)
        summary_per_doc.append(row)
        torch.cuda.empty_cache()
        print(f"doc {i}/{len(docs)} chars={len(d['context'])} chunks={n_captured} -> {e}", flush=True)

chunk_mod.solve_tril = orig_solve_tril

bf16_max = [r["shipped_bf16out_max_abs_err"] for r in summary_per_doc if "shipped_bf16out_max_abs_err" in r]
fp32_max = [r["same_kernel_fp32out_max_abs_err"] for r in summary_per_doc if "same_kernel_fp32out_max_abs_err" in r]
summary = {
    "n_docs": len(docs),
    "reading": (
        "if same_kernel_fp32out max is close to the plain-fp32-no-tf32 reference (~1e-6/1e-7, from "
        "a5_tril_precision_tok.py), the 0.00195 found there is bf16 OUTPUT ROUNDING (output_dtype=k.dtype), "
        "not the 'ieee' tl.dot computation itself being imprecise -- i.e. the same noise floor as every "
        "other bf16 number in this model, not a new problem. If same_kernel_fp32out is ALSO far from fp64, "
        "the ieee tl.dot computation inside the kernel is itself imprecise, which IS a new problem."
    ),
    "shipped_bf16out_vs_fp64": {"max_over_docs": max(bf16_max) if bf16_max else None},
    "same_kernel_fp32out_vs_fp64": {"max_over_docs": max(fp32_max) if fp32_max else None},
    "per_document": summary_per_doc,
}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_document"}, indent=1), flush=True)
print("wrote", OUT, flush=True)
