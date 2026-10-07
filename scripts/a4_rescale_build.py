"""A4 (acc) deliverable 3: build TWO alternative NVFP4 side-files that differ from the shipped
`/work/next/models/nvfp4_experts_36l.safetensors` ONLY in the down-projection (w2) per-block scale
(the one A4's saturation check found clipping on real long documents, every one of 36 layers) --
w1/s1/g1/g2 and the activation global scales stay untouched, so any KL change is attributable only to
the w2 block-scale method.

Confirmed (NVFP4-NEXT.md:472-474) that the SHIPPED file was built by the plain, non-importance-weighted
`prismyra.kernels.nvfp4.quantize()` (max of each 16-block, divided by 6, under one global per-expert
scale) -- "重要度重み付き探索なし". The two alternatives:
  (search) `quantize_search()`, ALREADY implemented in nvfp4.py (used by prep_nvfp4_search.py when an
           importance file is given) but called here with importance=None, i.e. the plain-squared-
           error version of the "誤差最小" (error-minimizing) method asked for: grid-search 7 fractions
           of each block's own max (1.0..0.7) and keep whichever minimises squared reconstruction error.
  (fourover six) a new, small function below implementing the task's literal "4/6 方式": per 16-block,
           try mapping the block's max to E2M1's grid value 4 AND to grid value 6 (today's only choice),
           keep whichever has lower squared error -- the mechanism EXTERNAL.md's Four-Over-Six summary
           describes (adaptive per-block M=4-or-6 choice), implemented directly rather than imported
           (no such function exists in nvfp4.py).
"""
import json, os, sys
sys.path.insert(0, "/work/next/src")
sys.path.insert(0, "/work/inv")  # must win over /work/next/src/prismyra, which lacks kernels/nvfp4.py
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from prismyra.kernels.nvfp4 import _dequant_fp8, quantize, quantize_search, _e2m1, FP8

SRC = os.environ.get("A4_FP8_SRC", "/work/models/p35-l36_kd025_7k_cap32k")
SHIPPED = "/work/next/models/nvfp4_experts_36l.safetensors"
OUT_SEARCH = sys.argv[1] if len(sys.argv) > 1 else "/work/next/models/nvfp4_experts_36l_w2search.safetensors"
OUT_46 = sys.argv[2] if len(sys.argv) > 2 else "/work/next/models/nvfp4_experts_36l_w2fourover6.safetensors"


def quantize_fourover6(w: torch.Tensor, global_scale: torch.Tensor):
    """Per 16-block: try mapping the block's own max to E2M1 grid value 4, and to grid value 6 (what
    `quantize()` always does); keep whichever minimises squared reconstruction error for that block."""
    rows, cols = w.shape
    blocks = w.view(rows, cols // 16, 16)
    amax = blocks.abs().amax(-1)
    table = _e2m1(w.device)
    best_err = best_scale = None
    for grid_top in (4.0, 6.0):
        scale = (amax / grid_top * global_scale).clamp(max=448.0).to(FP8)
        eff = (scale.float() / global_scale).unsqueeze(-1)
        x = torch.where(eff > 0, blocks / eff, torch.zeros_like(blocks)).clamp(-6, 6)
        idx = (x.abs().unsqueeze(-1) - table).abs().argmin(-1)
        back = table[idx] * torch.where(x < 0, -1.0, 1.0) * eff
        err = ((back - blocks) ** 2).sum(-1)
        if best_err is None:
            best_err, best_scale = err, scale.float()
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err); best_scale = torch.where(better, scale.float(), best_scale)
    scale = best_scale.to(FP8)
    eff = (scale.float() / global_scale).unsqueeze(-1)
    x = torch.where(eff > 0, blocks / eff, torch.zeros_like(blocks)).clamp(-6, 6)
    code = (x.abs().unsqueeze(-1) - table).abs().argmin(-1) | ((x < 0).to(torch.long) << 3)
    code = code.to(torch.uint8).view(rows, cols)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), scale.contiguous()


idx = json.load(open(f"{SRC}/model.safetensors.index.json")); wm = idx["weight_map"]
cfg = json.load(open(f"{SRC}/config.json")); tc = cfg.get("text_config", cfg); L, E = tc["num_hidden_layers"], tc["num_experts"]
def get(k):
    with safe_open(f"{SRC}/{wm[k]}", "pt") as f:
        return f.get_tensor(k)

shipped = load_file(SHIPPED)
out_search, out_46 = dict(shipped), dict(shipped)  # start as full copies; only w2/s2 get replaced below
dev = "cuda"
mse_report = {}
for l in range(L):
    p = f"model.language_model.layers.{l}.mlp.experts"
    w2 = torch.stack([get(f"{p}.{e}.down_proj.weight") for e in range(E)])
    s2 = torch.stack([get(f"{p}.{e}.down_proj.weight_scale_inv") for e in range(E)])
    g2 = shipped[f"{l}.g2"].to(dev)
    shipped_w2, shipped_s2 = shipped[f"{l}.w2"].to(dev), shipped[f"{l}.s2"].to(dev)
    new_w2s, new_s2s, new_w2f, new_s2f = [], [], [], []
    mses = {"shipped": [], "search": [], "fourover6": []}
    for e in range(E):
        d = _dequant_fp8(w2[e].to(dev), s2[e].to(dev))
        ge = g2[e]
        ws, ss = quantize_search(d, ge, None)
        wf, sf = quantize_fourover6(d, ge)
        new_w2s.append(ws); new_s2s.append(ss); new_w2f.append(wf); new_s2f.append(sf)
        # reconstruction MSE for a quick same-layer sanity readout (not the final deliverable, just a check)
        from prismyra.kernels.nvfp4 import _dequant_fp8 as _dq
    out_search[f"{l}.w2"] = torch.stack(new_w2s).cpu(); out_search[f"{l}.s2"] = torch.stack(new_s2s).cpu()
    out_46[f"{l}.w2"] = torch.stack(new_w2f).cpu(); out_46[f"{l}.s2"] = torch.stack(new_s2f).cpu()
    print("layer", l, "done", flush=True)

save_file(out_search, OUT_SEARCH)
save_file(out_46, OUT_46)
print("wrote", OUT_SEARCH, OUT_46)
