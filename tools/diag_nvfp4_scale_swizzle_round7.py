"""inv5, round 7, chair step 1: `diag_nvfp4_moe_minimal_round7.py` found that `ops.scaled_fp4_quant`'s packed
values (`xq`) for the context row are bit-identical whether a companion shares the call, but its scale-factor
tensor (`xsf`, the NVFP4 "swizzled blockscale" layout, padded to (128, K_padded)) is not. This script de-swizzles
`xsf` back to the logical (row, 16-element-block) order for the context row only, and compares that against the
*computed* per-block scale (the same formula `prismyra/kernels/nvfp4.py:quantize()` uses: block amax / 6 * global
scale, clamped, cast to e4m3), which depends only on the context row's own data -- never on any companion.

If the de-swizzled value for the context row matches between "alone" and "with a companion" (and matches the
data-only computed scale), the swizzle buffer carries the *same* logical content in both cases and the chair's
conclusion follows: whatever differs is placement, not value, and the main cause sits downstream in the GEMM.
If the de-swizzled value itself differs, the activation quantization step is where the row-count dependence
enters, before any GEMM runs.

Usage: python3 diag_nvfp4_scale_swizzle_round7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale  # noqa: E402

from prismyra.kernels.nvfp4 import FP8  # noqa: E402

K = 2048
DEVICE = "cuda"


def unswizzle_blockscale(swizzled: torch.Tensor, m_logical: int, k_logical_blocks: int) -> torch.Tensor:
    """Invert `swizzle_blockscale`'s pad+reshape+permute exactly, then slice back to the logical shape."""
    from vllm.utils.math_utils import round_up

    m_padded = round_up(m_logical, 128)
    k_padded = round_up(k_logical_blocks, 4)
    assert tuple(swizzled.shape) == (m_padded, k_padded), (swizzled.shape, m_padded, k_padded)
    # forward: (m_padded//128, 4, 32, k_padded//4, 4) --permute(0,1,4,3,2,5)--> (m_padded//128, k_padded//4, 32, 4, 4)
    reshaped = swizzled.reshape(m_padded // 128, k_padded // 4, 32, 4, 4)
    # invert permute(0,1,4,3,2,5) on the 6-d (with leading batch=1) tensor: axes (0,1,4,3,2,5) -> original (0..5)
    # here without the batch dim, axes map (0,1,2,3,4) of `reshaped` back to (0,3,2,1,4) of the pre-permute tensor.
    original = reshaped.permute(0, 3, 2, 1, 4).contiguous()
    flat = original.reshape(m_padded, k_padded)
    return flat[:m_logical, :k_logical_blocks]


def logical_scale_for_row(x_row: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """The per-16-block e4m3 scale `quantize()` would compute for this row, from data alone -- no companion, no
    kernel, just the formula in `prismyra/kernels/nvfp4.py:quantize()`."""
    cols = x_row.shape[-1]
    blocks = x_row.view(-1, cols // 16, 16)
    scale = (blocks.abs().amax(-1) / 6.0 * global_scale).clamp(max=448.0).to(FP8)
    return scale


def main():
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    a1_gscale = torch.tensor([2.0], device=DEVICE)  # a fixed, data-independent global scale, as the real layer uses

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    x_context = (torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE, generator=gen) * 0.1)

    with torch.inference_mode():
        xq_a, xsf_a = ops.scaled_fp4_quant(x_context.contiguous(), a1_gscale)

    gen2 = torch.Generator(device=DEVICE).manual_seed(42)
    x_context2 = (torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE, generator=gen2) * 0.1)
    assert torch.equal(x_context, x_context2)
    gen3 = torch.Generator(device=DEVICE).manual_seed(777)
    x_comp = (torch.randn(31, K, dtype=torch.bfloat16, device=DEVICE, generator=gen3) * 0.1)
    x_all = torch.cat([x_context2, x_comp], dim=0)
    with torch.inference_mode():
        xq_t, xsf_t = ops.scaled_fp4_quant(x_all.contiguous(), a1_gscale)

    print(f"xsf_a shape={tuple(xsf_a.shape)}  xsf_t shape={tuple(xsf_t.shape)}")
    print(f"xq row0 equal: {torch.equal(xq_a[:1], xq_t[:1])}")
    print(f"raw xsf (full swizzled buffer) equal: {torch.equal(xsf_a, xsf_t[: xsf_a.shape[0]])}")

    k_blocks = K // 16
    computed = logical_scale_for_row(x_context[0], a1_gscale)
    deswizzled_alone = unswizzle_blockscale(xsf_a, 1, k_blocks)[0]
    deswizzled_together = unswizzle_blockscale(xsf_t, 32, k_blocks)[0]

    print(f"\nde-swizzled row-0 scale, alone vs data-only-computed: "
          f"{'MATCH' if torch.equal(deswizzled_alone, computed) else 'DIFFER'}")
    print(f"de-swizzled row-0 scale, together vs data-only-computed: "
          f"{'MATCH' if torch.equal(deswizzled_together, computed) else 'DIFFER'}")
    print(f"de-swizzled row-0 scale, alone vs together: "
          f"{'MATCH' if torch.equal(deswizzled_alone, deswizzled_together) else 'DIFFER'}")
    if not torch.equal(deswizzled_alone, deswizzled_together):
        d = (deswizzled_alone.float() - deswizzled_together.float()).abs()
        print(f"  nonzero positions: {int((d > 0).sum())}/{d.numel()}, "
              f"first few alone={deswizzled_alone[:8].tolist()} together={deswizzled_together[:8].tolist()}")


if __name__ == "__main__":
    main()
