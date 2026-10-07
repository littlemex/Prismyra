"""inv5, round 7, chair's leading hypothesis: `in_proj_qkv` (GDN layer 0's own q/k/v projection, confirmed by
dumping `engine.backbone`'s named_modules: `language_model.layers.0.linear_attn.in_proj_qkv` is `Fp8Linear`,
not a plain bf16 `nn.Linear`) calls `w8a8_triton_block_scaled_mm` directly
(`prismyra/kernels/qwen3_moe.py:Fp8Linear.forward`) -- a raw Triton function, not `aten::mm`/`addmm`/`linear`,
so the batch-invariant dispatcher registration (`_enable_batch_invariance`, which only overrides those four
aten ops) never sees it. The Triton kernel's own tile config (BLOCK_M/N/K, num_warps, num_stages) is chosen
from an M-bucketed JSON table (`fp8_tuning.py`, built by fp4spd/fp8spd), so a different total row count M
(the branch's row count, which is the companion count for question-count=1) can select a different config and
move row 0's result -- exactly the mechanism `diag_gdn_core_real_inputs_round7.py` found upstream of GDN's own
core (q/k/v already differ before `chunk_gated_delta_rule` is even called).

Cheap, direct test (chair's own suggested check): hold row 0's input fixed, vary total M (1, 2, 3, 7, 8, 16,
32), call `in_proj_qkv` alone, and compare row 0's output by `torch.equal`.

Usage: PRISMYRA_MODEL=<repo> python3 diag_fp8linear_m_bucket_round7.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Prismyra  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    import prismyra.engine as _engine_mod
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, "dispatcher not registered -- see round-7 13.5 note"
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    target = None
    for name, m in engine.backbone.named_modules():
        if name == "language_model.layers.0.linear_attn.in_proj_qkv":
            target = m
            break
    assert target is not None, "could not find layer 0's in_proj_qkv"
    print(f"target module: {type(target).__name__}, in_features={target.in_features}, "
          f"out_features={target.out_features}", flush=True)

    gen = torch.Generator(device="cuda").manual_seed(42)
    row0 = torch.randn(1, target.in_features, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.1

    def run(m_total: int) -> torch.Tensor:
        if m_total == 1:
            x = row0.clone()
        else:
            gen_c = torch.Generator(device="cuda").manual_seed(777)
            rest = torch.randn(m_total - 1, target.in_features, device="cuda", dtype=torch.bfloat16,
                                generator=gen_c) * 0.1
            x = torch.cat([row0, rest], dim=0)
        with torch.inference_mode():
            out = target(x)
        return out[:1].detach().clone()

    baseline = run(1)
    print("baseline (M=1) computed.\n", flush=True)
    for m_total in (2, 3, 7, 8, 16, 32, 33, 64):
        out = run(m_total)
        eq = torch.equal(baseline, out)
        tag = "IDENTICAL" if eq else f"DIFFERS max|diff|={(baseline - out).abs().float().max().item():.3e}"
        print(f"M={m_total:3d}: row 0 {tag}", flush=True)


if __name__ == "__main__":
    main()
