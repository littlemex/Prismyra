"""Routed experts in W4A16 (4-bit weights, bf16 activations), on vLLM's Marlin MoE kernel. Experimental.

Selected with `PRISMYRA_EXPERTS=int4_marlin`. Unlike the NVFP4 path (`kernels/nvfp4.py`), this is a weight-only
quantization -- the activations stay bf16, so no calibration file of per-layer input maxima is needed. The point is a
checkpoint small enough for a 24 GiB Ada/Ampere card (an L4), where the FP8 experts already overrun 32 GiB
(`/Users/akazawt/tmp/smr/air/RUN-q3.md` section 3).

The recipe: each expert's FP8 blocks are restored to the values served today (`nvfp4._dequant_fp8`, reused rather than
duplicated), then quantized round-to-nearest in groups of 128 along the input (reduction) dimension with one
`uint4b8` scale per group -- vLLM's own name for a symmetric 4-bit code biased by 8, which needs no stored zero
point (`vllm.scalar_type.scalar_types.uint4b8`). Packing into the AutoGPTQ int32 layout and the group-scale layout
is vLLM's own tested code (`quantize_weights`/`pack_rows`), not reimplemented here: the risk this project's own
record (`RUN-q3.md` section 5) calls out explicitly -- "getting the bit order of the packing wrong even once produces
a wrong answer without crashing" -- is the risk of writing that packing by hand, which this file does not do.
`gptq_marlin_moe_repack`/`marlin_moe_permute_scales` turn that AutoGPTQ layout into the Marlin kernel's own tile
layout, again vLLM's own code.

Converted one layer at a time from the host, the same reason `nvfp4.py`'s module docstring gives: the FP8 experts
never have to fit on the device beside the packed int4 result.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from .nvfp4 import _dequant_fp8, tiny_experts  # noqa: F401 - tiny_experts re-exported for engine.py

#: Along the reduction (input) dimension of each matrix, matching this checkpoint's hidden size (2048) and routed
#: expert intermediate size (512) exactly (16 and 4 groups respectively, no remainder) -- so the Marlin thread-tile
#: padding `marlin_moe_padded_intermediate` would otherwise add never applies to this checkpoint; the forward below
#: still calls `fused_marlin_moe` the same way a padded checkpoint would, pad or no pad is a fact about the
#: checkpoint's own shapes, not about this file's own correctness.
GROUP_SIZE = 128

#: W4A16 compute runs the activations through unmodified (no `marlin_quant_input` call site in this file at all;
#: `fused_marlin_moe`'s own `input_dtype=None` default is what selects that path), so the per-group scale and the
#: packed weight only ever meet a bf16 input tensor. Keeping the scale itself in bf16, the model's own compute dtype,
#: is the natural reading of "weight-only": the only quantity this file invents is the weight's int4 code, and the
#: scale that undoes it should round the same way everything else in a bf16 forward pass does.
SCALE_DTYPE = torch.bfloat16


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, in) float -> AutoGPTQ-packed int4 qweight (in//8, out) int32 and per-group scales (in//128, out).

    Round-to-nearest, symmetric, group_size=128, `uint4b8` (vLLM's bias-128... bias-8 convention: packable
    0..15 with no separate zero point). `w` is `nn.Linear`-shaped (out_features, in_features); GPTQ's own
    convention is (K, N) = (in_features, out_features), the transpose, which is what every function called
    here expects and what `pack_rows`/`gptq_marlin_moe_repack` below assume without re-checking.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import pack_rows, quantize_weights
    from vllm.scalar_type import scalar_types

    out_features, in_features = w.shape
    w_kn = w.t().contiguous()  # (K, N) = (in_features, out_features)
    _, w_q, w_s, _ = quantize_weights(w_kn, scalar_types.uint4b8, GROUP_SIZE)
    packed = pack_rows(w_q, 4, in_features, out_features)
    return packed, w_s.to(SCALE_DTYPE)


class FusedExpertsMarlinInt4(nn.Module):
    """The routed path in W4A16; router and shared expert untouched, as in `FusedExperts`/`FusedExpertsFp4`."""

    def __init__(
        self,
        block: nn.Module,
        top_k: int,
        w13: torch.Tensor,
        s13: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
        device,
    ):
        super().__init__()
        from vllm._custom_ops import gptq_marlin_moe_repack
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_moe_permute_scales
        from vllm.scalar_type import scalar_types

        self.gate = block.gate
        self.shared_expert = block.shared_expert
        self.shared_expert_gate = block.shared_expert_gate
        self.top_k = top_k
        self.quant_type_id = scalar_types.uint4b8.id

        e = w13.shape[0]
        gate_up_n, hidden = w13.shape[1], w13.shape[2]  # N, K for the gate+up projection
        down_n, intermediate = w2.shape[1], w2.shape[2]  # N, K for the down projection

        packed1 = torch.empty(e, hidden // 8, gate_up_n, dtype=torch.int32, device=device)
        scale1 = torch.empty(e, hidden // GROUP_SIZE, gate_up_n, dtype=SCALE_DTYPE, device=device)
        packed2 = torch.empty(e, intermediate // 8, down_n, dtype=torch.int32, device=device)
        scale2 = torch.empty(e, intermediate // GROUP_SIZE, down_n, dtype=SCALE_DTYPE, device=device)
        for i in range(e):
            d1 = _dequant_fp8(w13[i].to(device), s13[i].to(device))
            packed1[i], scale1[i] = quantize(d1)
            d2 = _dequant_fp8(w2[i].to(device), s2[i].to(device))
            packed2[i], scale2[i] = quantize(d2)
            if os.environ.get("PRISMYRA_INT4_DEBUG") and (i + 1) % 32 == 0:
                print("int4_marlin", i + 1, "/", e, round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)

        self.w1 = gptq_marlin_moe_repack(packed1, hidden, gate_up_n, 4)
        self.w2 = gptq_marlin_moe_repack(packed2, intermediate, down_n, 4)
        self.w1_scale = marlin_moe_permute_scales(scale1, hidden, gate_up_n, GROUP_SIZE)
        self.w2_scale = marlin_moe_permute_scales(scale2, intermediate, down_n, GROUP_SIZE)
        if hasattr(block, "experts"):
            del block.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_topk
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe

        from .qwen3_moe import _ROUTER_LINEAR

        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        if not x.is_contiguous():
            x = x.contiguous()
        # Same reasoning and the same fix as `FusedExpertsFp4.forward`'s own comment: calling `self.gate(x)` whole
        # would recompute the softmax/top-k `fused_topk` already does, and would run the router's plain `F.linear`
        # outside the row-count-invariant kernel this project's companion/qn determinism guarantee needs there.
        linear = _ROUTER_LINEAR or torch.nn.functional.linear
        logits = linear(x, self.gate.weight)
        weights, ids = fused_topk(x, logits, self.top_k, renormalize=True)[:2]
        out = fused_marlin_moe(
            x,
            self.w1,
            self.w2,
            None,
            None,
            self.w1_scale,
            self.w2_scale,
            weights.float(),
            ids,
            self.quant_type_id,
        )
        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return (out + shared).reshape(shape)


@torch.no_grad()
def convert(model: nn.Module, top_k: int, device) -> int:
    """Replace every sparse MoE block's routed experts with the W4A16 Marlin path, converting straight from this
    checkpoint's own FP8 expert weights -- no `PRISMYRA_NVFP4_CALIB`-style file: weight-only quantization needs no
    per-layer activation maximum."""
    names = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Qwen3_5MoeSparseMoeBlock"]
    for name, block in names:
        experts = block.experts
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(
            parent,
            name.rsplit(".", 1)[1],
            FusedExpertsMarlinInt4(
                block,
                top_k,
                experts.gate_up_proj,
                experts.gate_up_proj_scale_inv,
                experts.down_proj,
                experts.down_proj_scale_inv,
                device,
            ),
        )
    torch.cuda.empty_cache()
    return len(names)
