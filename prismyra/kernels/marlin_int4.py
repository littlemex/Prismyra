"""Routed experts in W4A16 (4-bit weights, bf16 activations), on vLLM's Marlin MoE kernel. Experimental.

Selected with `PRISMYRA_EXPERTS=int4_marlin` and `model` pointed at a checkpoint directory whose index leaves the
routed-expert weights out (`PRISMYRA_INT4_MARLIN_CHECKPOINT` names the *original* checkpoint `convert()` reads the
real FP8 expert tensors from -- the same split `kernels/nvfp4.py` makes between the stripped checkpoint a caller
loads and the file its own `prepare_layer`/`convert` read the real weights from). Unlike the NVFP4 path, this is a
weight-only quantization -- activations stay bf16, so no calibration file of per-layer input maxima is needed.

The point is a checkpoint small enough for a 24 GiB Ada/Ampere card (an L4), where the ordinary FP8 routed experts
alone are already about 27 GiB -- more than the card has, loaded whole, before any quantizing even starts
(measured directly: `AutoModel.from_pretrained` on the plain checkpoint `torch.OutOfMemoryError`s on an L4 while
still loading, nowhere near this file's own code). `tiny_experts()` (`kernels/nvfp4.py`, reused rather than
duplicated) is what keeps a caller's `AutoModel.from_pretrained` from ever allocating the full-sized routed
experts in the first place; `convert()` below separately reads each expert's *real* FP8 tensors straight off the
original checkpoint's safetensors shards, one expert at a time, through vLLM's own `safe_open` -- lazy, so opening
a shard costs nothing until a tensor is actually read, and nothing from one expert's read is kept once the next
expert's starts. This is the same "one layer at a time from the host" discipline `nvfp4.py`'s own module docstring
describes, taken one level further (one *expert*, not one *layer*) because here the one card both the FP8 source
and the int4 destination have to fit in at once is the card this format exists for.

The quantization recipe: each expert's FP8 blocks are restored to the values served today
(`nvfp4._dequant_fp8`, reused rather than duplicated), then quantized round-to-nearest in groups of 128 along the
input (reduction) dimension with one `uint4b8` scale per group -- vLLM's own name for a symmetric 4-bit code
biased by 8, which needs no stored zero point. Packing into the AutoGPTQ int32 layout and the group-scale layout
is vLLM's own tested code (`quantize_weights`/`pack_rows`), not reimplemented here: the risk this project's own
record (`RUN-q3.md` section 5) calls out explicitly -- "getting the bit order of the packing wrong even once
produces a wrong answer without crashing" -- is the risk of writing that packing by hand, which this file does
not do. `gptq_marlin_moe_repack`/`marlin_moe_permute_scales` turn that AutoGPTQ layout into the Marlin kernel's
own tile layout, again vLLM's own code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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

    Round-to-nearest, symmetric, group_size=128, `uint4b8` (vLLM's bias-8 convention: packable 0..15 with no
    separate zero point). `w` is `nn.Linear`-shaped (out_features, in_features); GPTQ's own convention is
    (K, N) = (in_features, out_features), the transpose, which is what every function called here expects and
    what `pack_rows`/`gptq_marlin_moe_repack` below assume without re-checking.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import pack_rows, quantize_weights
    from vllm.scalar_type import scalar_types

    out_features, in_features = w.shape
    w_kn = w.t().contiguous()  # (K, N) = (in_features, out_features)
    _, w_q, w_s, _ = quantize_weights(w_kn, scalar_types.uint4b8, GROUP_SIZE)
    packed = pack_rows(w_q, 4, in_features, out_features)
    return packed, w_s.to(SCALE_DTYPE)


class ShardReader:
    """Lazy, per-tensor access to one checkpoint's safetensors shards, by the key names its own
    `model.safetensors.index.json` declares -- the on-disk layout (`model.language_model.layers.N.mlp.experts.
    I.{gate_proj,up_proj,down_proj}.weight[_scale_inv]`, one tensor per expert, not the stacked
    `(num_experts, ...)` parameter the loaded model presents) is a fact about this checkpoint format, not about
    the live module tree `tiny_experts()` leaves behind.

    Handles are opened once per shard file and kept for this reader's lifetime (`convert()` makes one `ShardReader`
    per call and lets it go once every layer is converted) rather than once per tensor: a layer's experts mostly
    share one shard file, and `safe_open` itself is lazy -- a tensor's bytes are only read from disk when
    `get_tensor` is called, not when the file is opened.
    """

    def __init__(self, checkpoint_dir: str):
        from safetensors import safe_open

        self._dir = Path(checkpoint_dir)
        self._index = json.loads((self._dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._open = safe_open
        self._handles: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        fname = self._index[key]
        handle = self._handles.get(fname)
        if handle is None:
            handle = self._handles[fname] = self._open(str(self._dir / fname), framework="pt", device="cpu")
        return handle.get_tensor(key)


class FusedExpertsMarlinInt4(nn.Module):
    """The routed path in W4A16; router and shared expert untouched, as in `FusedExperts`/`FusedExpertsFp4`.

    Built directly from a `ShardReader` over the *original* checkpoint rather than from the live module's own
    (never-materialized, `tiny_experts()`-shrunk) expert parameters: see this module's own docstring for why.
    """

    def __init__(
        self,
        block: nn.Module,
        top_k: int,
        layer: str,
        num_experts: int,
        hidden: int,
        intermediate: int,
        reader: ShardReader,
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

        gate_up_n = 2 * intermediate
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        packed1 = torch.empty(num_experts, hidden // 8, gate_up_n, dtype=torch.int32, device=device)
        scale1 = torch.empty(num_experts, hidden // GROUP_SIZE, gate_up_n, dtype=SCALE_DTYPE, device=device)
        packed2 = torch.empty(num_experts, intermediate // 8, hidden, dtype=torch.int32, device=device)
        scale2 = torch.empty(num_experts, intermediate // GROUP_SIZE, hidden, dtype=SCALE_DTYPE, device=device)
        for i in range(num_experts):
            # `_apply_gate`'s own chunk order (transformers' finegrained_fp8.py: `gate, up = gate_up.chunk(2,
            # dim=-1)`) is gate first, up second -- the concatenation below has to match it exactly, because this
            # is the order the live `FusedExpertsFp4`/`FusedExperts` siblings' `gate_up_proj` was already built in
            # by the framework's own loader for every other expert format this project serves.
            gate_w = reader.get(f"{prefix}{i}.gate_proj.weight")
            gate_s = reader.get(f"{prefix}{i}.gate_proj.weight_scale_inv")
            up_w = reader.get(f"{prefix}{i}.up_proj.weight")
            up_s = reader.get(f"{prefix}{i}.up_proj.weight_scale_inv")
            down_w = reader.get(f"{prefix}{i}.down_proj.weight")
            down_s = reader.get(f"{prefix}{i}.down_proj.weight_scale_inv")
            w13 = torch.cat([gate_w, up_w], dim=0).to(device)
            s13 = torch.cat([gate_s, up_s], dim=0).to(device)
            d1 = _dequant_fp8(w13, s13)
            packed1[i], scale1[i] = quantize(d1)
            d2 = _dequant_fp8(down_w.to(device), down_s.to(device))
            packed2[i], scale2[i] = quantize(d2)
            if os.environ.get("PRISMYRA_INT4_DEBUG") and (i + 1) % 32 == 0:
                print(
                    "int4_marlin layer",
                    layer,
                    i + 1,
                    "/",
                    num_experts,
                    round(torch.cuda.memory_allocated() / 2**30, 2),
                    "GiB",
                    flush=True,
                )

        self.w1 = gptq_marlin_moe_repack(packed1, hidden, gate_up_n, 4)
        self.w2 = gptq_marlin_moe_repack(packed2, intermediate, hidden, 4)
        self.w1_scale = marlin_moe_permute_scales(scale1, hidden, gate_up_n, GROUP_SIZE)
        self.w2_scale = marlin_moe_permute_scales(scale2, intermediate, hidden, GROUP_SIZE)
        del packed1, scale1, packed2, scale2
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
def convert(model: nn.Module, top_k: int, device, checkpoint_dir: str) -> int:
    """Replace every sparse MoE block's routed experts with the W4A16 Marlin path, streaming each expert's real
    FP8 tensors from `checkpoint_dir` (the *original*, un-stripped checkpoint -- `model` itself was loaded from a
    stripped copy under `tiny_experts()`, so its own expert parameters are a 1-row placeholder, never the real
    weights) -- see this module's own docstring for why. No `PRISMYRA_NVFP4_CALIB`-style file: weight-only
    quantization needs no per-layer activation maximum."""
    reader = ShardReader(checkpoint_dir)
    names = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Qwen3_5MoeSparseMoeBlock"]
    for name, block in names:
        key = name[name.index("language_model.") :] if "language_model." in name else name
        layer = key.split(".")[2]  # "language_model.layers.N.mlp" -> ["language_model","layers","N","mlp"]
        num_experts = block.gate.weight.shape[0]
        hidden = block.gate.weight.shape[1]
        intermediate = block.shared_expert.down_proj.weight.shape[1]
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(
            parent,
            name.rsplit(".", 1)[1],
            FusedExpertsMarlinInt4(block, top_k, layer, num_experts, hidden, intermediate, reader, device),
        )
        torch.cuda.empty_cache()
    return len(names)
