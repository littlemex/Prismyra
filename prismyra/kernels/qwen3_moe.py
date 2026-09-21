"""Replacements for Qwen3.6-style mixture-of-experts backbones with a gated linear-attention recurrence.

Six replacements, each verified against the implementation it replaces before being kept. What each is worth, and what
was measured and rejected on the way, is in docs/KERNELS.md; the counts below are the only measured facts that belong in
code, because they are what tells an unfamiliar model from a familiar one.

Every replacement falls back rather than failing when the kernel it borrows is absent, so the model answers without
vLLM installed -- slower, and the adapter says so.
"""

from __future__ import annotations

import torch
from torch import nn

from . import Applied, Swap, register
from .conv import available as triton_available
from .conv import causal_depthwise_conv1d, starts_from_boundaries

BLOCK = (128, 128)

#: Submodules the replacements import inside their forward pass. Each is checked before anything is replaced.
REQUIRED_VLLM = (
    "vllm.vllm_flash_attn",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.quantization.utils.fp8_utils",
)

#: The configuration these replacements were measured against, read from the checkpoint. Checked before anything is
#: touched, because it is what makes the module counts below correct. Note the two key-value heads: this family's
#: linear-attention layers carry 32 value heads and there are 16 attention heads, and neither is this number.
MEASURED_CONFIG = {
    "num_hidden_layers": 40,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
}

# How many modules of each kind this architecture has. A different number means a different model.
EXPECTED = {
    "routed_experts": 40,
    "dense_matmul": 250,
    "norm": 80,
    "head_duplication": 30,
    "attention": 10,
    "convolution": 30,
}


# --------------------------------------------------------------------------- routed experts
class FusedExperts(nn.Module):
    """The routed path through one fused kernel, keeping the router and shared expert untouched.

    The framework groups tokens by expert by materialising a permuted copy and scattering the result back. The fused
    kernel reads each row through the routing index instead, so the copy never exists. The checkpoint's expert weights
    are already in the layout it wants.
    """

    def __init__(self, block: nn.Module, top_k: int):
        super().__init__()
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

        self.gate = block.gate
        self.shared_expert = block.shared_expert
        self.shared_expert_gate = block.shared_expert_gate
        self.top_k = top_k
        experts = block.experts
        self.w1, self.w2 = experts.gate_up_proj, experts.down_proj
        # Two conversions, both checked by tests: the scales widen to float32 because the kernel indexes them so, and
        # `_scale_inv` is the multiplier that restores a stored value, not its reciprocal.
        self.quant = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=list(BLOCK),
            w1_scale=experts.gate_up_proj_scale_inv.to(torch.float32),
            w2_scale=experts.down_proj_scale_inv.to(torch.float32),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk

        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        logits, _, _ = self.gate(x)  # the original router, so the same experts are chosen
        weights, ids = fused_topk(x, logits, self.top_k, renormalize=True)[:2]
        routed = fused_experts(x, self.w1, self.w2, weights, ids, quant_config=self.quant)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return (routed + shared).reshape(shape)


# --------------------------------------------------------------------------- dense projections
class Fp8Linear(nn.Module):
    """A block-quantised projection on a faster kernel, wrapping the original module's weights."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.weight = inner.weight  # stays (out, in): the kernel asserts A and B share their last dimension
        self.register_buffer("scale", inner.weight_scale_inv.to(torch.float32), persistent=False)
        self.out_features, self.in_features = inner.weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
            w8a8_triton_block_scaled_mm,
        )

        shape = x.shape
        xq, xs = per_token_group_quant_fp8(x.reshape(-1, shape[-1]), BLOCK[1])
        out = w8a8_triton_block_scaled_mm(xq, self.weight, xs, self.scale, list(BLOCK), output_dtype=x.dtype)
        return out.reshape(*shape[:-1], self.out_features)


# --------------------------------------------------------------------------- normalisation
class FastRMSNorm(nn.Module):
    """Normalisation on a faster kernel, with the model's `1.0 + weight` offset folded in once.

    The offset is the trap: this model computes `normalised * (1.0 + weight)` and the kernel computes
    `normalised * weight`. Folding it at construction keeps the forward free of an addition over the full width.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.eps = float(getattr(inner, "eps", 1e-6))
        self.register_buffer("weight_plus_one", (1.0 + inner.weight.float()).to(inner.weight.dtype), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm import _custom_ops as ops

        y = x if x.is_contiguous() else x.contiguous()
        out = torch.empty_like(y)
        ops.rms_norm(out, y, self.weight_plus_one, self.eps)
        return out


# --------------------------------------------------------------------------- attention
class FlashAttention(nn.Module):
    """Attention on a variable-length kernel, in both the context pass and the branch pass.

    The branch pass is where this matters. With a cache present the framework materialises an additive mask, which the
    fast kernel cannot take, so it falls back to a memory-efficient kernel built for an older architecture. No mask is
    needed: every branch's queries are the last positions of its own sequence and attend to the context's keys plus its
    own, which is the right-aligned causal case. Padding branches to a common width stays safe because a pad position's
    answer is never read.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.layer_idx = getattr(inner, "layer_idx", None)

    def _project(self, hidden_states, position_embeddings):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            apply_rotary_pos_emb,
        )

        a = self.inner
        shape = hidden_states.shape[:-1]
        heads = (*shape, -1, a.head_dim)
        # The query projection carries the output gate beside the queries, which is why it is twice as wide as the head
        # count suggests.
        query, gate = torch.chunk(a.q_proj(hidden_states).view(*shape, -1, a.head_dim * 2), 2, dim=-1)
        query = a.q_norm(query.view(heads)).transpose(1, 2)
        key = a.k_norm(a.k_proj(hidden_states).view(heads)).transpose(1, 2)
        value = a.v_proj(hidden_states).view(heads).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value, gate.reshape(*shape, -1), shape

    def _finish(self, out, gate, shape):
        a = self.inner
        attn = out.reshape(*shape, -1).contiguous() * torch.sigmoid(gate)
        return a.o_proj(attn), None

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func
        except ImportError:
            return self.inner.forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

        a = self.inner
        cu = kwargs.get("cu_seq_lens_q")
        if past_key_values is None and cu is None:
            return self.inner.forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

        query, key, value, gate, shape = self._project(hidden_states, position_embeddings)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, a.layer_idx)
            rows, _, q_len, _ = query.shape
            k_len = key.shape[-2]
            q = query.transpose(1, 2).reshape(rows * q_len, -1, a.head_dim)
            k = key.transpose(1, 2).reshape(rows * k_len, -1, a.head_dim)
            v = value.transpose(1, 2).reshape(rows * k_len, -1, a.head_dim)
            cu_q = torch.arange(0, rows * q_len + 1, q_len, device=q.device, dtype=torch.int32)
            cu_k = torch.arange(0, rows * k_len + 1, k_len, device=q.device, dtype=torch.int32)
            max_q, max_k = q_len, k_len
        else:
            # Packed contexts: one row by construction, the boundaries say where each one ends.
            q = query.squeeze(0).transpose(0, 1).contiguous()
            k = key.squeeze(0).transpose(0, 1).contiguous()
            v = value.squeeze(0).transpose(0, 1).contiguous()
            cu_q = cu_k = cu
            max_q = max_k = int(kwargs.get("max_length_q") or (cu[1:] - cu[:-1]).max().item())

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=a.scaling,
            causal=True,
        )
        return self._finish(out[0] if isinstance(out, tuple) else out, gate, shape)


# --------------------------------------------------------------------------- convolution
#: Set on the convolution weights of the layers this adapter owns. The replacement is installed on a framework-wide
#: name, so it needs a way to tell a tensor it was measured on from one that merely has the same shape.
MEASURED_MARK = "_prismyra_measured"


def _tag_conv_weights(model: nn.Module) -> int:
    """Mark the convolution weight of every recurrent layer.

    Found by type rather than by attribute name: a one-dimensional convolution inside a gated linear-attention layer is
    the thing, and depending on an attribute's spelling would silently tag nothing if the framework renamed it -- which
    the adapter would then report as a model it does not recognise.
    """
    tagged = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3_5MoeGatedDeltaNet":
            continue
        for child in module.modules():
            weight = getattr(child, "weight", None)
            if isinstance(child, nn.Conv1d) and weight is not None:
                setattr(weight, MEASURED_MARK, True)
                tagged += 1
    return tagged


def _install_conv() -> bool:
    """Route the layer's convolution through the Triton kernel, by replacing a name in the framework's own module.

    Module-scope, which covers every layer at once and is also the cost: the replacement is visible to every model of
    this family in the process, not only to this engine's. The wrapper therefore checks the tensor it was handed and
    defers to the original for anything it was not measured on, rather than assuming its caller.

    The layer transposes its tensor before calling, and that transpose is a view, so transposing it back is free and
    recovers the token-major tensor this kernel wants. The old kernel's copy came from its contiguity demand, not from
    the transpose.
    """
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

    if getattr(m.causal_conv1d_fn, "_prismyra", False):
        return True
    if not triton_available():
        return False
    original = m.causal_conv1d_fn

    def patched(x, weight, bias=None, activation=None, **kwargs):
        # Two guards, and the first is the important one. Replacing a name in the framework's module is process-wide, so
        # another model of this family in the same process calls this function too -- and it was not measured on that
        # model. The weights this adapter tagged are the only ones it will act on; everything else goes back to the
        # original, which is the implementation those cases were written for.
        if not getattr(weight, MEASURED_MARK, False):
            return original(x, weight, bias, activation=activation, **kwargs)
        if bias is not None or x.dim() != 3 or x.shape[0] != 1 or not x.is_cuda:
            return original(x, weight, bias, activation=activation, **kwargs)
        tokens_major = x.squeeze(0).t()
        if not tokens_major.is_contiguous():
            tokens_major = tokens_major.contiguous()
        starts = kwargs.get("prismyra_seq_starts")
        if starts is None:
            cu = kwargs.get("cu_seq_lens_q")
            if cu is not None and cu.numel() > 2:
                starts = starts_from_boundaries(cu, tokens_major.shape[0])
        out = causal_depthwise_conv1d(
            tokens_major, weight, seq_starts=starts, activation=activation if activation is not None else "silu"
        )
        return out.t().unsqueeze(0)

    patched._prismyra = True
    m.causal_conv1d_fn = patched
    return True


# --------------------------------------------------------------------------- the adapter
class Qwen3MoeAdapter:
    """Replacements measured on Qwen3.6-35B-A3B-FP8."""

    name = "qwen3-moe"

    def supports(self, config) -> bool:
        """The family, and the configuration within it that these replacements were measured against.

        The name alone is not enough. Another checkpoint in this family has a different number of layers and experts,
        and an adapter that claimed it would mutate the model and only then refuse it -- leaving a half-replaced model
        behind, for a caller who never asked for the kernels in the first place.
        """
        names = getattr(config, "architectures", None) or []
        if not any("Qwen3_5Moe" in n for n in names):
            return False
        decoder = getattr(config, "text_config", config)
        return all(getattr(decoder, name, None) == value for name, value in MEASURED_CONFIG.items())

    def replace(self, model: nn.Module, config) -> Applied:
        decoder = getattr(config, "text_config", config)
        applied = Applied(adapter=self.name)
        have_vllm = _have("vllm")
        # Probed here rather than discovered in the forward pass. A replacement that falls back on its first call has
        # already been counted as applied, so `stats()` would report a kernel that never runs.
        missing = [name for name in REQUIRED_VLLM if have_vllm and not _have(name)]
        if missing:
            have_vllm = False
            applied.skipped.append("vllm is installed but missing " + ", ".join(missing))

        if have_vllm:
            applied.swaps.append(
                Swap(
                    "routed_experts",
                    _swap_children(
                        model, "Qwen3_5MoeSparseMoeBlock", lambda m: FusedExperts(m, decoder.num_experts_per_tok)
                    ),
                    EXPECTED["routed_experts"],
                )
            )
            applied.swaps.append(Swap("dense_matmul", _swap_block_quantised(model), EXPECTED["dense_matmul"]))
            applied.swaps.append(
                Swap("norm", _swap_children(model, "Qwen3_5MoeRMSNorm", FastRMSNorm), EXPECTED["norm"])
            )
            applied.swaps.append(
                Swap("attention", _swap_children(model, "Qwen3_5MoeAttention", FlashAttention), EXPECTED["attention"])
            )
        else:
            applied.skipped.append("vllm is not installed: the borrowed kernels are unavailable")

        applied.swaps.append(Swap("head_duplication", _drop_head_duplication(model), EXPECTED["head_duplication"]))
        if not _install_conv():
            applied.skipped.append("triton is not available: the convolution keeps the framework's path")
        else:
            applied.swaps.append(Swap("convolution", _tag_conv_weights(model), EXPECTED["convolution"]))
            applied.notes.append(
                "the convolution kernel covers the context pass; a branch pass arrives as many rows and keeps the "
                "framework's path"
            )
        return applied


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _swap_children(model: nn.Module, class_name: str, make) -> int:
    """Replace every child whose class has this name. Collected first so the walk is not mutated under itself."""
    targets = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if type(child).__name__ == class_name
    ]
    for parent, name, child in targets:
        setattr(parent, name, make(child))
    return len(targets)


def _swap_block_quantised(model: nn.Module) -> int:
    """Replace every two-dimensional fp8 projection that carries a two-dimensional block scale.

    Identified by structure, which excludes the expert stacks: they hold a leading expert dimension and no `weight`, and
    belong to the routed path.
    """

    def is_target(module: nn.Module) -> bool:
        w = getattr(module, "weight", None)
        if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
            return False
        s = getattr(module, "weight_scale_inv", None)
        return s is not None and getattr(s, "dim", lambda: 0)() == 2

    targets = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if is_target(child)
    ]
    for parent, name, child in targets:
        setattr(parent, name, Fp8Linear(child))
    return len(targets)


def _drop_head_duplication(model: nn.Module, verify: bool = True) -> int:
    """Stop the linear-attention layers duplicating query and key, which the recurrence handles itself.

    The layer reads `num_k_heads` in its forward pass only to decide whether to duplicate, so setting it equal to
    `num_v_heads` disables that and changes nothing else. That is a flag being set through an attribute whose name no
    longer describes its value, so one layer is run both ways and the outputs are required to match exactly before the
    change is applied to any of them.
    """
    layers = [
        m
        for m in model.modules()
        if type(m).__name__ == "Qwen3_5MoeGatedDeltaNet" and getattr(m, "num_v_heads", 0) > getattr(m, "num_k_heads", 1)
    ]
    if not layers:
        return 0
    if verify:
        probe = layers[0]
        p = next(probe.parameters())
        x = torch.randn(1, 64, probe.hidden_size, device=p.device, dtype=p.dtype) * 0.1
        original = probe.num_k_heads
        with torch.inference_mode():
            before = probe(x, cache_params=None)
            probe.num_k_heads = probe.num_v_heads
            after = probe(x, cache_params=None)
        probe.num_k_heads = original
        before = before[0] if isinstance(before, tuple) else before
        after = after[0] if isinstance(after, tuple) else after
        moved = (before.float() - after.float()).abs().max().item()
        if moved != 0.0:
            raise RuntimeError(
                f"skipping the head duplication moved a layer's output by {moved:.3e}; it was bit-identical when "
                f"measured, so this layer's code has changed and the optimisation is no longer safe"
            )
    for m in layers:
        m.num_k_heads = m.num_v_heads
    return len(layers)


register(Qwen3MoeAdapter())
