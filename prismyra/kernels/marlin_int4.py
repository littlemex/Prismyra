"""Routed experts in W4A16 (4-bit weights, bf16 activations). Experimental.

Selected with `PRISMYRA_EXPERTS=int4_marlin` and `model` pointed at a checkpoint directory whose index leaves the
routed-expert weights out (`PRISMYRA_INT4_MARLIN_CHECKPOINT` names the *original* checkpoint `convert()` reads the
real FP8 expert tensors from -- the same split `kernels/nvfp4.py` makes between the stripped checkpoint a caller
loads and the file its own `prepare_layer`/`convert` read the real weights from). Unlike the NVFP4 path, this is a
weight-only quantization -- activations stay bf16, so no calibration file of per-layer input maxima is needed.

**Not vLLM's `fused_marlin_moe` CUDA kernel** (RUN-quant2.md section 1): the first round of this feature
(`RUN-quant1.md` section 3-3) measured that kernel's answer changing on every repeated, identical ask -- not from
`use_atomic_add` (hardcoded `False` at every one of this kernel's own call sites, in both the dev vLLM checkout and
the actual installed 0.27.1 wheel; confirmed by reading both) and not from a workspace reused stale across calls
(every call allocates a fresh, zeroed scratch buffer; confirmed by reading `marlin_make_workspace_new` and by the
fact that *replaying the same captured CUDA graph* -- same kernel, same addresses, same inputs, nothing left to
vary -- still produced a different number each time). What is left is the kernel's own split-K design: with this
few tokens routed to any one of 256 experts, the number of independent output tiles is far smaller than an L4's 58
SMs, so the kernel always cooperatively splits one tile's reduction across many threadblocks and combines the
partial sums through a global-memory lock, whose number of participants (and therefore its floating-point summation
order) this file could only shrink, not remove, through the kernel's own public knobs (`thread_k`/`thread_n`/
`blocks_per_sm`): the smallest config this file could ask for still launches at least one threadblock per SM, which
on an L4 is already more threadblocks than this shape has independent tiles. So this file does not call that kernel
at all. It stores the weights exactly as compact (packed 4-bit integers plus one scale per group of 128, the same
bytes `quantize()` below has always produced) and, every forward, dequantizes *only the experts a reader's own
tokens actually route to* back to bf16 and runs them through a plain `torch.matmul` -- the same reduction every
other bf16 projection in this project already answers through, with no cooperative cross-threadblock step to race.

The point is a checkpoint small enough for a 24 GiB Ada/Ampere card (an L4), where the ordinary FP8 routed experts
alone are already about 27 GiB -- more than the card has, loaded whole, before any quantizing even starts
(measured directly: `AutoModel.from_pretrained` on the plain checkpoint `torch.OutOfMemoryError`s on an L4 while
still loading, nowhere near this file's own code). This file's own `tiny_experts()` is what keeps a caller's
`AutoModel.from_pretrained` from ever allocating the full-sized routed experts in the first place -- not
`nvfp4.py`'s own `tiny_experts()`, which patches `transformers.integrations.finegrained_fp8.FP8Experts.__init__`
and (measured directly, on real hardware, after this file's first version silently allocated the full 256-expert
tensors and `torch.OutOfMemoryError`'d regardless) is not the class this checkpoint's routed experts actually
load as: `Qwen3_5MoeExperts` (`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`), a plain, dtype-agnostic
module that reads `config.num_experts` directly to size `gate_up_proj`/`down_proj` -- the FP8 dtype itself is
applied to its parameters generically by the loader's own quantization machinery, not by a dedicated FP8 subclass
nvfp4.py's `tiny_experts()` could patch instead. `convert()` below separately reads each expert's *real* FP8
tensors straight off the original checkpoint's safetensors shards, one expert at a time, through vLLM's own
`safe_open` -- lazy, so opening a shard costs nothing until a tensor is actually read, and nothing from one
expert's read is kept once the next expert's starts. This is the same "one layer at a time from the host"
discipline `nvfp4.py`'s own module docstring describes, taken one level further (one *expert*, not one *layer*)
because here the one card both the FP8 source and the int4 destination have to fit in at once is the card this
format exists for.

The quantization recipe: each expert's FP8 blocks are restored to the values served today
(`nvfp4._dequant_fp8`, reused rather than duplicated), then quantized round-to-nearest in groups of 128 along the
input (reduction) dimension with one `uint4b8` scale per group -- vLLM's own name for a symmetric 4-bit code
biased by 8, which needs no stored zero point. Packing into the AutoGPTQ int32 layout and the group-scale layout
is vLLM's own tested code (`quantize_weights`/`pack_rows`), not reimplemented here: the risk this project's own
record (`RUN-q3.md` section 5) calls out explicitly -- "getting the bit order of the packing wrong even once
produces a wrong answer without crashing" -- is the risk of writing that packing by hand, which this file does
not do. The forward pass below unpacks the same way, `pack_rows`' own loop run backwards one slot at a time
(`_unpack_rows`): no Marlin tile layout is ever produced, so there is no `gptq_marlin_moe_repack`/
`marlin_moe_permute_scales` step to call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .nvfp4 import _dequant_fp8

#: Along the reduction (input) dimension of each matrix, matching this checkpoint's hidden size (2048) and routed
#: expert intermediate size (512) exactly (16 and 4 groups respectively, no remainder).
GROUP_SIZE = 128

#: W4A16 compute runs the activations through unmodified, so the per-group scale and the dequantized weight only
#: ever meet a bf16 input tensor. Keeping the scale itself in bf16, the model's own compute dtype, is the natural
#: reading of "weight-only": the only quantity this file invents is the weight's int4 code, and the scale that
#: undoes it should round the same way everything else in a bf16 forward pass does.
SCALE_DTYPE = torch.bfloat16


def _unpack_rows(packed: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    """Inverse of vLLM's `pack_rows` (`quant_utils.py`): `(size_k // pack_factor, size_n)` int32 -> `(size_k,
    size_n)` int32 codes in `[0, 2**num_bits)`. vLLM ships `pack_rows`' mirror image for columns (`unpack_cols`)
    but not for rows, so this file writes the row version itself -- the same loop as `unpack_cols`, shifted to
    split along dim 0 instead of dim 1, which is what `pack_rows` itself packs along (`q_res |= q_w[i::pack_factor,
    :] << num_bits * i`, this function's own loop body run in reverse, one slot at a time rather than hand-derived).
    """
    pack_factor = 32 // num_bits
    assert packed.shape == (size_k // pack_factor, size_n)
    mask = (1 << num_bits) - 1
    out = torch.empty((size_k, size_n), dtype=torch.int32, device=packed.device)
    rest = packed
    for i in range(pack_factor):
        out[i::pack_factor, :] = rest & mask
        rest = rest >> num_bits
    return out


def _dequant_int4(packed: torch.Tensor, scale: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    """One expert's `(size_k, size_n)` dense bf16 weight, from this file's own packed `uint4b8` code plus its
    per-group scale -- the plain-matmul-shaped counterpart of `quantize()`, called for one expert at a time so the
    dense result (16x the packed size) only ever exists for however many experts a single forward's own tokens
    actually route to, never for all of them at once.
    """
    codes = _unpack_rows(packed, 4, size_k, size_n)
    # `uint4b8`: the stored code is the signed value plus 8 (vLLM's own bias-8 convention, `quantize()`'s own
    # docstring), so undoing it is `code - 8`, not a lookup table.
    groups = size_k // GROUP_SIZE
    signed = (codes.view(groups, GROUP_SIZE, size_n).to(SCALE_DTYPE) - 8.0) * scale.view(groups, 1, size_n)
    return signed.reshape(size_k, size_n)


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, in) float -> AutoGPTQ-packed int4 qweight (in//8, out) int32 and per-group scales (in//128, out).

    Round-to-nearest, symmetric, group_size=128, `uint4b8` (vLLM's bias-8 convention: packable 0..15 with no
    separate zero point). `w` is `nn.Linear`-shaped (out_features, in_features); GPTQ's own convention is
    (K, N) = (in_features, out_features), the transpose, which is what `pack_rows`/`_unpack_rows` and
    `_dequant_int4` below all assume without re-checking.
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

        self.gate = block.gate
        self.shared_expert = block.shared_expert
        self.shared_expert_gate = block.shared_expert_gate
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate = intermediate
        self.gate_up_n = 2 * intermediate

        gate_up_n = self.gate_up_n
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

        # No `gptq_marlin_moe_repack`/`marlin_moe_permute_scales` step: see the module docstring for why this file
        # keeps the plain AutoGPTQ-packed layout `quantize()` already produces, rather than Marlin's tile layout.
        self.w1 = packed1
        self.w2 = packed2
        self.w1_scale = scale1
        self.w2_scale = scale2
        if hasattr(block, "experts"):
            del block.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_topk

        from .qwen3_moe import _ROUTER_LINEAR

        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        if not x.is_contiguous():
            x = x.contiguous()
        linear = _ROUTER_LINEAR or torch.nn.functional.linear
        logits = linear(x, self.gate.weight)
        weights, ids = fused_topk(x, logits, self.top_k, renormalize=True)[:2]

        # Flatten the (tokens, top_k) grid of (expert, weight) pairs the same way `ids.reshape(-1)`'s own row-major
        # order does, so `flat_rows[p]` is always the token `flat_ids[p]`/`flat_weights[p]` belong to.
        n_tokens = x.size(0)
        flat_ids = ids.reshape(-1)
        flat_weights = weights.reshape(-1).to(x.dtype)
        flat_rows = torch.arange(n_tokens, device=x.device).repeat_interleave(self.top_k)

        out = torch.zeros_like(x)
        # Only the experts this call's own tokens actually route to are ever dequantized -- `torch.unique` is
        # sorted by default, so this loop (and therefore the sequence of writes into `out`) runs in the same order
        # on every call with the same routing, independent of which physical order CUDA happens to schedule
        # anything in (the property `fused_marlin_moe`'s own split-K reduction could not give; see the module
        # docstring).
        for e in flat_ids.unique(sorted=True).tolist():
            mask = flat_ids == e
            rows = flat_rows[mask]
            row_weights = flat_weights[mask].unsqueeze(-1)
            xin = x.index_select(0, rows)
            w1 = _dequant_int4(self.w1[e], self.w1_scale[e], self.hidden, self.gate_up_n)
            w2 = _dequant_int4(self.w2[e], self.w2_scale[e], self.intermediate, self.hidden)
            gate, up = (xin @ w1).chunk(2, dim=-1)
            down = (F.silu(gate) * up) @ w2
            out.index_add_(0, rows, down * row_weights)

        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return (out + shared).reshape(shape)


#: Substring that marks a checkpoint key as one of the routed experts' own tensors (`...mlp.experts.{i}.
#: {gate_proj,up_proj,down_proj}.weight[_scale_inv]`), shared between `tiny_experts`'s file-reading patch below
#: and `convert()`'s own prefix-building -- both have to agree on what "an expert key" looks like.
EXPERT_KEY_MARKER = ".mlp.experts."


class _ExpertFreePointer:
    """Wraps one `safetensors.safe_open` file handle so `.keys()` never reports a routed-expert tensor.

    Exists because stripping `model.safetensors.index.json`'s own `weight_map` is not enough on its own
    (measured on real hardware: still `torch.OutOfMemoryError`'d with the index stripped). `transformers`'
    `_load_pretrained_model` does not trust that index for *which keys a file contains* -- it opens every file
    `checkpoint_files` names and calls `file_pointer.keys()` on it directly, folding every key the file
    physically holds into the merged state dict, regardless of what the index declared. A checkpoint's expert
    and non-expert tensors for one layer share one shard file here, so opening that file for its (small) kept
    keys silently re-admits all 256 of its (large) expert ones -- the index was never wrong, the file was just
    never asked to agree with it. Everything but `keys()` passes straight through to the real handle.
    """

    def __init__(self, inner):
        self._inner = inner

    def keys(self):
        return [k for k in self._inner.keys() if EXPERT_KEY_MARKER not in k]  # noqa: SIM118 - not a dict, a file handle

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


class tiny_experts:
    """While loading, keep a caller's `AutoModel.from_pretrained` from ever allocating or reading the routed
    experts at full size, the way `nvfp4.tiny_experts` does for the checkpoint and the loader version *it* was
    measured against. Not the same mechanism: on real hardware (an L4), with this checkpoint, on this installed
    `transformers`, neither half of `nvfp4.tiny_experts`'s trick held --

    1. The routed experts are a plain `Qwen3_5MoeExperts` (`transformers/models/qwen3_5_moe/
       modeling_qwen3_5_moe.py`), not `transformers.integrations.finegrained_fp8.FP8Experts` -- a dtype-agnostic
       module that reads `config.num_experts` directly in its own `__init__` to size `gate_up_proj`/`down_proj`.
       Patched here the same way nvfp4.tiny_experts patches the class it targets: a throwaway config copy with
       `num_experts=1` for the one call to the real `__init__`.
    2. A stripped `model.safetensors.index.json` is not enough to keep this installed `transformers`'
       `_load_pretrained_model` from reading the expert tensors anyway: it opens every checkpoint file through
       `safe_open` and folds in every key the file *physically* contains, not only the ones the index names (see
       `_ExpertFreePointer`'s own docstring for the measured mechanism). Patched here by making
       `transformers.modeling_utils.safe_open` return an `_ExpertFreePointer`, so `.keys()` itself never offers
       an expert key to be read -- a stripped index is then unnecessary (this patch alone is sufficient on an
       *un*-stripped checkpoint directory, confirmed on real hardware), though harmless to keep.
    """

    def __enter__(self):
        import copy

        import transformers.modeling_utils as mu
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

        self._m, self._init = m, m.Qwen3_5MoeExperts.__init__
        original = self._init

        def init(module, config, *args, **kwargs):
            small = copy.copy(config)
            small.num_experts = 1
            original(module, small, *args, **kwargs)

        m.Qwen3_5MoeExperts.__init__ = init

        self._mu = mu
        self._safe_open = mu.safe_open

        def filtered_safe_open(*args, **kwargs):
            return _ExpertFreePointer(self._safe_open(*args, **kwargs))

        mu.safe_open = filtered_safe_open
        return self

    def __exit__(self, *exc):
        self._m.Qwen3_5MoeExperts.__init__ = self._init
        self._mu.safe_open = self._safe_open


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
