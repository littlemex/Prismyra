"""inv5, round 7, chair step 2: with activation quantization already cleared (`diag_nvfp4_scale_swizzle_round7.py`:
the context row's de-swizzled scale and packed value are bit-identical whether a companion shares the call), hold
the context row's own `xq`/weights/expert-ids fixed and vary only how many *other* rows share the same expert
assignment, to tell apart:

  - per-expert row count M_e (how many rows of the whole call land on each expert) changing the grouped GEMM's
    own tile/CTA schedule for that expert -- would show up only when a companion is actually ASSIGNED to one of
    the context row's own 8 experts, not otherwise.
  - something keyed on the *total* M of the call regardless of overlap (e.g. a workspace size or a global
    swizzle bound) -- would show up even when every companion is assigned to experts the context row never
    touches.

Calls `FusedExpertsFp4.routed()` directly (bypassing the router) with explicit `ids`/`weights`, so the context
row's own expert assignment is identical in every scenario below; only the companions' assignment changes.

Usage: python3 diag_nvfp4_moe_expert_overlap_round7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra.kernels import nvfp4  # noqa: E402

torch.manual_seed(0)

E, TOPK, K, N = 256, 8, 2048, 512
DEVICE = "cuda"


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(K, E, bias=False)
        self.shared_expert_gate = torch.nn.Linear(K, 1, bias=False)
        self.shared_expert = torch.nn.Sequential(torch.nn.Linear(K, N), torch.nn.SiLU(), torch.nn.Linear(N, K))


def build_layer() -> nvfp4.FusedExpertsFp4:
    import os

    os.environ["PRISMYRA_NVFP4_FUSED_FINALIZE"] = "0"
    w13 = (torch.randn(E, 2 * N, K, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
    s13 = torch.ones(E, (2 * N) // 128, K // 128, device=DEVICE)
    w2 = (torch.randn(E, K, N, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
    s2 = torch.ones(E, K // 128, N // 128, device=DEVICE)
    prepared = nvfp4.prepare_layer(w13, s13, w2, s2, DEVICE)
    act = {"w13_in_amax": 1.0, "w2_in_amax": [1.0]}
    block = FakeBlock().to(DEVICE, torch.bfloat16)
    return nvfp4.FusedExpertsFp4(block, TOPK, prepared, act, DEVICE)


CONTEXT_EXPERTS = list(range(8))  # the context row's own fixed top-8 expert assignment, every scenario


def run(layer: nvfp4.FusedExpertsFp4, n_companions: int, companion_experts: list[int], label: str) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    x_context = torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE, generator=gen) * 0.1
    gen3 = torch.Generator(device=DEVICE).manual_seed(777)
    x_comp = torch.randn(n_companions, K, dtype=torch.bfloat16, device=DEVICE, generator=gen3) * 0.1 if n_companions else x_context[:0]
    x_all = torch.cat([x_context, x_comp], dim=0)

    # Realistic (non-round) softmax-like weights, not exact powers of two -- uniform 1/8 weights masked the
    # effect entirely (every scenario came back IDENTICAL with them), which this script's first run found.
    gen_w = torch.Generator(device=DEVICE).manual_seed(1234)
    ids_context = torch.tensor([CONTEXT_EXPERTS], device=DEVICE, dtype=torch.int32)
    weights_context = torch.softmax(torch.randn(1, TOPK, device=DEVICE, generator=gen_w), dim=-1).to(torch.float32)
    if n_companions:
        ids_comp = torch.tensor([companion_experts] * n_companions, device=DEVICE, dtype=torch.int32)
        gen_wc = torch.Generator(device=DEVICE).manual_seed(5678)
        weights_comp = torch.softmax(
            torch.randn(n_companions, TOPK, device=DEVICE, generator=gen_wc), dim=-1
        ).to(torch.float32)
        ids_all = torch.cat([ids_context, ids_comp], dim=0)
        weights_all = torch.cat([weights_context, weights_comp], dim=0)
    else:
        ids_all, weights_all = ids_context, weights_context

    with torch.inference_mode():
        out = layer.routed(x_all, weights_all, ids_all)
    return out[:1].detach().clone()


def main():
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    layer = build_layer()

    baseline = run(layer, 0, [], "alone")
    print("baseline (no companion) computed.\n", flush=True)

    disjoint_experts = list(range(200, 208))  # 8 experts the context row never touches
    overlapping_experts = CONTEXT_EXPERTS  # exactly the context row's own 8 experts

    for n in (1, 7, 31, 63):
        out_disjoint = run(layer, n, disjoint_experts, "disjoint")
        eq_d = torch.equal(baseline, out_disjoint)
        d_d = 0.0 if eq_d else (baseline - out_disjoint).abs().max().item()
        out_overlap = run(layer, n, overlapping_experts, "overlap")
        eq_o = torch.equal(baseline, out_overlap)
        d_o = 0.0 if eq_o else (baseline - out_overlap).abs().max().item()
        print(f"n_companions={n:3d}  disjoint-experts: {'IDENTICAL' if eq_d else f'DIFFERS max|diff|={d_d:.3e}'}"
              f"   overlapping-experts: {'IDENTICAL' if eq_o else f'DIFFERS max|diff|={d_o:.3e}'}", flush=True)

    print("\nVerdict: if 'disjoint-experts' stays IDENTICAL while 'overlapping-experts' DIFFERS, the cause is "
          "per-expert row count M_e (hypothesis 2) -- a companion only matters when it lands on one of the "
          "context row's own experts. If BOTH differ equally, the cause is keyed on the call's total M "
          "regardless of overlap (e.g. workspace sizing or a global swizzle bound), not per-expert M_e alone.")


if __name__ == "__main__":
    main()
