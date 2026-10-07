"""inv5, round 7 (chair's 3 hypotheses, 2026-10-07): fp8spd4 found that on sm_120, `FusedExpertsFp4` (the NVFP4
routed-expert layer) is the FIRST op to diverge when an unrelated branch row is concatenated alongside a context
row -- the CONTEXT row's own output changes, even though nothing about its own content changed. The chair asks
for the smallest possible repro, at one MoE layer, with no real model/checkpoint: a fixed row and an unrelated
row concatenated, compared by `torch.equal`, and three candidate mechanisms to tell apart:

  (1) the non-fused finalize sums a token's own top-k expert contributions in expert-sorted order, and that
      order shifts when other rows are present (unfused != row-count invariant, only run-to-run deterministic --
      `prismyra/kernels/nvfp4.py`'s own comment on `fused_finalize` only promises the latter).
  (2) per-expert row counts (how many tokens of the whole batch land on each expert) change the grouped GEMM's
      own tile/split schedule.
  (3) NVFP4 activation quantisation (16-element blocks) pads or straddles a block differently depending on which
      row is adjacent to it in the batch.

Uses `prismyra.kernels.nvfp4.prepare_layer`/`FusedExpertsFp4` directly, with small random FP8 expert weights (no
checkpoint download needed) so this can run before/independently of this round's full-model generation.

Usage: python3 diag_nvfp4_moe_minimal_round7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from prismyra.kernels import nvfp4  # noqa: E402

torch.manual_seed(0)

E, TOPK, K, N = 256, 8, 2048, 512  # matches the real nvfp4-36l config.json exactly (num_experts, top_k, hidden_size, moe_intermediate_size)
DEVICE = "cuda"


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(K, E, bias=False)
        self.shared_expert_gate = nn.Linear(K, 1, bias=False)
        self.shared_expert = nn.Sequential(nn.Linear(K, N), nn.SiLU(), nn.Linear(N, K))


def build_layer(fused_finalize: bool) -> nvfp4.FusedExpertsFp4:
    import os

    os.environ["PRISMYRA_NVFP4_FUSED_FINALIZE"] = "1" if fused_finalize else "0"
    w13 = (torch.randn(E, 2 * N, K, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
    s13 = torch.ones(E, (2 * N) // 128, K // 128, device=DEVICE)
    w2 = (torch.randn(E, K, N, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
    s2 = torch.ones(E, K // 128, N // 128, device=DEVICE)
    prepared = nvfp4.prepare_layer(w13, s13, w2, s2, DEVICE)
    act = {"w13_in_amax": 1.0, "w2_in_amax": [1.0]}
    block = FakeBlock().to(DEVICE, torch.bfloat16)
    return nvfp4.FusedExpertsFp4(block, TOPK, prepared, act, DEVICE)


def run_once(layer: nvfp4.FusedExpertsFp4, m_context: int, m_companion: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (context-row output when alone, context-row output when a `m_companion`-row companion shares the
    same forward call), same context rows both times (same RNG state consumed identically before either call)."""
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    x_context = torch.randn(m_context, K, dtype=torch.bfloat16, device=DEVICE, generator=gen) * 0.1

    with torch.inference_mode():
        alone = layer.forward(x_context.clone())

    gen2 = torch.Generator(device=DEVICE).manual_seed(42)
    x_context2 = torch.randn(m_context, K, dtype=torch.bfloat16, device=DEVICE, generator=gen2) * 0.1
    assert torch.equal(x_context, x_context2), "context rows must be identical input in both calls"
    gen3 = torch.Generator(device=DEVICE).manual_seed(777)
    x_companion = torch.randn(m_companion, K, dtype=torch.bfloat16, device=DEVICE, generator=gen3) * 0.1
    with torch.inference_mode():
        together = layer.forward(torch.cat([x_context2, x_companion], dim=0))[:m_context]

    return alone, together


def main():
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"E={E} top_k={TOPK} K={K} N={N}\n", flush=True)

    print("=== hypothesis test: does an unrelated companion row change the context row's own MoE output? ===")
    layer = build_layer(fused_finalize=False)
    for m_context, m_companion in ((1, 1), (1, 7), (1, 31), (1, 63), (2, 6), (2, 30), (8, 24)):
        alone, together = run_once(layer, m_context, m_companion)
        eq = torch.equal(alone, together)
        tag = "IDENTICAL" if eq else f"DIFFERS max|diff|={(alone - together).abs().max().item():.3e}"
        print(f"[unfused finalize] m_context={m_context} m_companion={m_companion}: {tag}", flush=True)

    print("\n=== same, with fused_finalize=1 (known non-deterministic run-to-run per the code's own comment; "
          "included only to see whether it is even MORE sensitive, not as a candidate fix) ===")
    layer_fused = build_layer(fused_finalize=True)
    for m_context, m_companion in ((1, 1), (1, 7), (1, 31)):
        alone, together = run_once(layer_fused, m_context, m_companion)
        eq = torch.equal(alone, together)
        tag = "IDENTICAL" if eq else f"DIFFERS max|diff|={(alone - together).abs().max().item():.3e}"
        print(f"[fused finalize] m_context={m_context} m_companion={m_companion}: {tag}", flush=True)


if __name__ == "__main__":
    main()
