"""Replacements for Qwen3.6-style mixture-of-experts backbones with a gated linear-attention recurrence.

Six replacements, each verified against the implementation it replaces before being kept. What each is worth, and what
was measured and rejected on the way, is in docs/KERNELS.md; the counts below are the only measured facts that belong
in code, because they are what tells an unfamiliar model from a familiar one.

Every replacement falls back rather than failing when the kernel it borrows is absent, so the model answers without
vLLM installed -- slower, and the adapter says so.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import varlen
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


def expected_counts(decoder) -> dict[str, int]:
    """How many modules of each kind this configuration implies, derived rather than remembered.

    Every count here follows from the config: one routed block per layer, one replacement per attention layer, one per
    recurrent layer. Deriving them means the check keeps working when the framework reorganises its module tree, and
    it keeps failing when the model is genuinely a different one -- which is what the check is for.

    Two replacements are absent on purpose. The normalisation and the dense projections are found by structure, and
    how many of those a model contains is a fact about one revision of somebody else's tree: 40 layers give 80 layer
    norms on one version and 101 modules on another, once per-head query and key norms and the final norm share a
    class. A number there would break on an upgrade while proving nothing, so those two are verified against the
    implementation they replace instead. See `_verify`.
    """
    layer_types = list(getattr(decoder, "layer_types", []) or [])
    attention_layers = layer_types.count("full_attention")
    recurrent_layers = layer_types.count("linear_attention")
    return {
        "routed_experts": decoder.num_hidden_layers,
        "attention": attention_layers,
        "head_duplication": recurrent_layers,
        "convolution": recurrent_layers,
        # Two, and not per layer: the replacement is a module-level name that every layer's call site reads.
        "gated_delta_rule": len(DELTA_PATHS),
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
    """Normalisation on a faster kernel, with the module's scale folded in once.

    The trap is which scale. Some revisions of this model compute `normalised * (1.0 + weight)` and others `normalised
    * weight`, while the borrowed kernel always computes the latter -- so whether to add one is a fact about the code
    in front of you, not about the architecture. Getting it backwards does not raise; it shifts every activation by a
    factor near one, which is a plausible wrong answer.

    So it is not assumed. `offset` is chosen by running the module both ways and keeping the arrangement that matches
    bit for bit, in `_swap_and_verify`. Folding the choice in at construction keeps the forward free of an addition
    over the full width either way.
    """

    def __init__(self, inner: nn.Module, offset: bool = True):
        super().__init__()
        self.inner = inner
        self.offset = offset
        self.eps = float(getattr(inner, "eps", 1e-6))
        scale = inner.weight.float() + 1.0 if offset else inner.weight.float()
        self.register_buffer("weight_plus_one", scale.to(inner.weight.dtype), persistent=False)

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
    needed: every branch's queries are the last positions of its own sequence and attend to the context's keys plus
    its own, which is the right-aligned causal case. Padding branches to a common width stays safe because a pad
    position's answer is never read.
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
        # The query projection carries the output gate beside the queries, which is why it is twice as wide as the
        # head count suggests.
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

        boundaries = varlen.current()
        if past_key_values is not None:
            layer = past_key_values.layers[a.layer_idx]
            reading_many = boundaries is not None and not getattr(layer, "writing_branches", False)
            key, value = past_key_values.update(key, value, a.layer_idx)
            rows, _, q_len, _ = query.shape
            if reading_many:
                # Several documents in one flat row. Nothing may attend across a boundary, so the boundaries are the
                # cumulative lengths on both sides -- the arrangement a serving engine uses for a batch of prefills,
                # and the reason this is possible at all is that the kernel already takes them.
                assert key is not None and rows == 1, "a batched read arrives as one row of concatenated documents"
                q = query.squeeze(0).transpose(0, 1).contiguous()
                k = key.squeeze(0).transpose(0, 1).contiguous()
                v = value.squeeze(0).transpose(0, 1).contiguous()
                out = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=boundaries.offsets,
                    cu_seqlens_k=boundaries.offsets,
                    max_seqlen_q=boundaries.longest,
                    max_seqlen_k=boundaries.longest,
                    softmax_scale=a.scaling,
                    causal=True,
                )
                return self._finish(out[0] if isinstance(out, tuple) else out, gate, shape)
            q = query.transpose(1, 2).reshape(rows * q_len, -1, a.head_dim)
            cu_q = torch.arange(0, rows * q_len + 1, q_len, device=q.device, dtype=torch.int32)

            if key is None:
                # A paged layer returns nothing contiguous for a branch write, because there is nothing contiguous: the
                # context's pages are named by every row's table and never copied. `seqused_k` replaces `cu_seqlens_k`
                # here and passing both is not allowed, and `max_seqlen_k` is the pool's capacity rather than the real
                # length -- an upper bound is what that argument is for, and reading the real one would mean a
                # device-to-host copy on the request path.
                pool_keys, pool_values, table, seqused, capacity = layer.paged_read(rows)
                out = flash_attn_varlen_func(
                    q,
                    pool_keys,
                    pool_values,
                    cu_seqlens_q=cu_q,
                    max_seqlen_q=q_len,
                    max_seqlen_k=capacity,
                    softmax_scale=a.scaling,
                    causal=True,
                    block_table=table,
                    seqused_k=seqused,
                )
                return self._finish(out[0] if isinstance(out, tuple) else out, gate, shape)

            k_len = key.shape[-2]
            k = key.transpose(1, 2).reshape(rows * k_len, -1, a.head_dim)
            v = value.transpose(1, 2).reshape(rows * k_len, -1, a.head_dim)
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


#: Replacements named here are not installed, and the skip says so. A measurement tool, not a feature: the only honest
#: way to price one replacement is to run the same work with and without it, and `fast_kernels=False` turns off
#: all of them at once. Read from the environment because it has to be set before the weights load.
#:
#:     PRISMYRA_WITHOUT=gated_delta_rule python evals/run.py --task race --methods readout
def _withheld() -> set[str]:
    import os

    return {name.strip() for name in os.environ.get("PRISMYRA_WITHOUT", "").split(",") if name.strip()}


# --------------------------------------------------------------------------- the gated delta rule
#: How far the borrowed recurrence may sit from the one it replaces before it is refused. Loose, and deliberately: the
#: framework's own implementation promotes to float32 and scans in chunks of sixty-four, and the kernel does neither, so
#: the two disagree by more than bfloat16 rounding while computing the same recurrence.
DELTA_TOLERANCE = 5e-2

#: What the framework calls the two paths, and which borrowed kernel replaces each. Long sequences take the chunked
#: scan and a single token takes the recurrent step, and a branch pass uses the first of them.
DELTA_PATHS = (
    ("torch_chunk_gated_delta_rule", "chunk", "chunk_gated_delta_rule"),
    ("torch_recurrent_gated_delta_rule", "fused_recurrent", "fused_recurrent_gated_delta_rule"),
)


def _borrowed_delta(where: str, what: str):
    """One of vLLM's vendored flash-linear-attention kernels, or None with the reason it is unavailable."""
    import importlib

    try:
        module = importlib.import_module(f"vllm.third_party.flash_linear_attention.ops.{where}")
    except ImportError as e:
        return None, f"vllm's flash-linear-attention is not importable: {e}"
    kernel = getattr(module, what, None)
    if kernel is None:
        return None, f"{what} is not in vllm's flash-linear-attention build"
    return kernel, None


def _delta_wrapper(kernel, original):
    """The borrowed kernel behind the framework's own signature.

    Arguments are filtered by the kernel's signature rather than forwarded: the two paths take slightly different sets
    -- the recurrent one has no `output_final_state`, because it always returns the state -- and forwarding everything
    would raise on the first forward rather than at install time. `original` is kept on the wrapper so a caller can see
    what was replaced, and so this can be undone.
    """
    import functools
    import inspect

    takes = set(inspect.signature(kernel).parameters)

    @functools.wraps(original)
    def call(query, key, value, g=None, beta=None, **kwargs):
        passed = {name: value_ for name, value_ in kwargs.items() if name in takes}
        boundaries = varlen.current()
        # `passed.get` rather than `not in passed`: the framework hands this argument over explicitly as None, so asking
        # whether the key is present says yes and the boundaries were dropped -- the recurrence then scanned straight
        # across the boundary and returned one state for two documents, which is a plausible answer and not an error.
        if boundaries is not None and "cu_seqlens" in takes and passed.get("cu_seqlens") is None:
            # A batched read. With the boundaries the kernel scans each document separately and returns **one final
            # state per document**, which is exactly the per-document state a fork needs -- so the thing that makes a
            # batched read possible and the thing that makes it useful are the same argument.
            passed["cu_seqlens"] = boundaries.offsets
        return kernel(query, key, value, g=g, beta=beta, **passed)

    call.replaced = original
    return call


def _delta_inputs(decoder, device, tokens: int = 128):
    """Shapes the real layer produces, from the config rather than from a guess.

    The verification runs before any forward, so there is nothing to observe; the numbers that decide the shapes are
    declared -- sixteen key heads of 128, thirty-two value heads of 128 on the supported model -- and the layer repeats
    the query and key across the value heads before it calls either path, which is why every tensor here has the value
    head count.
    """
    import torch

    heads = decoder.linear_num_value_heads
    key_dim = decoder.linear_key_head_dim
    value_dim = decoder.linear_value_head_dim
    generator = torch.Generator(device=device).manual_seed(0)

    def make(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.bfloat16)

    return {
        "query": make(1, tokens, heads, key_dim),
        "key": make(1, tokens, heads, key_dim),
        "value": make(1, tokens, heads, value_dim),
        # In log space and negative, which is what the layer produces: a decay, not a gain. Positive values here would
        # make the scan diverge and the comparison meaningless.
        "g": -make(1, tokens, heads).abs().float(),
        "beta": make(1, tokens, heads).sigmoid().float(),
    }


def _install_gated_delta_rule(decoder, device, verify: bool = True) -> tuple[int, str | None]:
    """Route the linear-attention recurrence through the borrowed kernel.

    This is where a branch pass spends itself. Thirty of the forty layers are gated delta nets, one of them issues
    **1,021 kernel launches** against a full-attention layer's 114, and a whole pass spends 58% of its wall clock
    waiting for launches rather than computing -- 32,873 of them, 93% from these thirty layers. The framework's
    implementation is a chunked scan written in PyTorch: 218 copies, 191 elementwise kernels, 82 multiplies and 66 sums
    for a single layer.

    It is also not the implementation the framework wants. Both paths are decorated to ask a kernel hub first and fall
    back to this one, so the slow path is what runs when the hub has nothing installed -- and vLLM ships the kernel the
    hub would have provided.

    Installed on the framework's module-level names, which is how both of the layer's call sites pick it up. That is
    process-wide rather than engine-wide, which is the same trade the convolution replacement makes and is recorded
    there for the same reason. Verified against the implementation it replaces before being kept, because a recurrence
    that is subtly wrong produces a plausible answer rather than an error.
    """
    import importlib

    import torch

    try:
        modeling = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    except ImportError as e:
        return 0, f"the framework has no qwen3_5_moe module to patch: {e}"

    installed = 0
    for name, where, what in DELTA_PATHS:
        original = getattr(modeling, name, None)
        if original is None:
            return installed, f"the framework has no {name} to replace"
        if getattr(original, "replaced", None) is not None:
            installed += 1  # already installed in this process, by this engine or another
            continue
        kernel, why = _borrowed_delta(where, what)
        if kernel is None:
            return installed, why
        wrapper = _delta_wrapper(kernel, original)
        if verify and where == "chunk":
            # Only the chunked path is verified. The recurrent one is a single token and this engine never takes it:
            # a branch carries a whole suffix, so the chunked path is what a read-out runs.
            moved = _delta_disagreement(original, wrapper, decoder, device)
            if moved is None:
                return installed, "the framework's own implementation raised, so there is nothing to verify against"
            if moved > DELTA_TOLERANCE:
                return installed, f"the borrowed {what} disagreed by {moved:.3e}, above {DELTA_TOLERANCE:.0e}"
        setattr(modeling, name, wrapper)
        installed += 1
    del torch
    return installed, None


def _delta_disagreement(original, wrapper, decoder, device) -> float | None:
    """How far the two implementations are apart on the same inputs, relative to the output's own scale."""
    import torch

    inputs = _delta_inputs(decoder, device)
    try:
        with torch.inference_mode():
            want, _ = original(**inputs, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
            got, _ = wrapper(**inputs, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    except Exception:  # noqa: BLE001 - either side may refuse these shapes, and that is a decline rather than a fault
        return None
    scale = want.float().abs().amax().clamp(min=1e-6)
    return float((got.float() - want.float()).abs().amax() / scale)


# --------------------------------------------------------------------------- convolution
#: Set on the convolution weights of the layers this adapter owns. The replacement is installed on a framework-wide
#: name, so it needs a way to tell a tensor it was measured on from one that merely has the same shape.
MEASURED_MARK = "_prismyra_measured"


#: Data pointers of the convolution weights this adapter measured. Addresses rather than attributes, and that is the
#: whole point: **the layer does not pass the weight, it passes a view of it.** `nn.Conv1d` holds
#: `(channels, 1, kernel)` and the forward calls the kernel with `weight.squeeze(1)`, a new Python object carrying none
#: of the original's attributes. So a mark set as an attribute was never seen, the wrapper deferred to the framework on
#: every call, and `stats()` reported a kernel that had never run -- while the docs credit it with 28.4 ms of 138.
#:
#: A view shares its storage, so the address is the identity that survives. Module-level for the same reason the wrapper
#: is: the name it is installed on is the framework's, and every model of this family in the process reaches it.
MEASURED_POINTERS: set[int] = set()


def _measured_conv(weight: torch.Tensor) -> bool:
    """Whether this tensor is, or is a view of, a convolution weight this adapter measured."""
    if getattr(weight, MEASURED_MARK, False):
        return True
    try:
        return weight.data_ptr() in MEASURED_POINTERS
    except RuntimeError:  # pragma: no cover - a meta or fake tensor has no address
        return False


def _tag_conv_weights(model: nn.Module) -> int:
    """Mark the convolution weight of every recurrent layer, by address as well as by attribute.

    Found by type rather than by attribute name: a one-dimensional convolution inside a gated linear-attention layer
    is the thing, and depending on an attribute's spelling would silently tag nothing if the framework renamed it --
    which the adapter would then report as a model it does not recognise.
    """
    tagged = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3_5MoeGatedDeltaNet":
            continue
        for child in module.modules():
            weight = getattr(child, "weight", None)
            if isinstance(child, nn.Conv1d) and weight is not None:
                setattr(weight, MEASURED_MARK, True)
                MEASURED_POINTERS.add(weight.data_ptr())
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
        # Two guards, and the first is the important one. Replacing a name in the framework's module is process-wide,
        # so another model of this family in the same process calls this function too -- and it was not measured on
        # that model. The weights this adapter tagged are the only ones it will act on; everything else goes back to
        # the original, which is the implementation those cases were written for.
        if not _measured_conv(weight):
            return original(x, weight, bias, activation=activation, **kwargs)
        if bias is not None or x.dim() != 3 or x.shape[0] != 1 or not x.is_cuda:
            return original(x, weight, bias, activation=activation, **kwargs)
        tokens_major = x.squeeze(0).t()
        if not tokens_major.is_contiguous():
            tokens_major = tokens_major.contiguous()
        boundaries = varlen.current()
        # How many tokens the framework prepended: it hands the convolution the state it was holding, convolves, and
        # drops the prefix. Zero on a cache that has never held anything and `kernel - 1` on one that has, which is what
        # a shelf always is after its first document.
        extra = tokens_major.shape[0] - boundaries.tokens if boundaries is not None else 0
        starts = kwargs.get("prismyra_seq_starts")
        if starts is None:
            cu = kwargs.get("cu_seq_lens_q")
            if cu is None and boundaries is not None and extra >= 0:
                # A batched read: several documents in one flat run, with the boundaries declared by the window the
                # engine opened rather than by the framework, which is not the one packing them here.
                cu = boundaries.with_prefix(extra, tokens_major.device)
            if cu is not None and cu.numel() > 2:
                starts = starts_from_boundaries(cu, tokens_major.shape[0])
        if boundaries is not None and extra >= 0:
            # The per-document tail of this layer's convolution input, for the engine to put back afterwards. The
            # framework will set a one-row state from the end of the whole run, which is the end of the last document.
            boundaries.conv_tails.append(boundaries.tails(tokens_major, weight.shape[-1], extra=extra))
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
        expected = expected_counts(decoder)
        applied = Applied(adapter=self.name)
        # The text decoder only. This checkpoint's class carries a vision tower that nothing here ever runs, and
        # replacing modules inside it would be work with no measurement behind it and no caller to benefit.
        text = getattr(model, "language_model", model)
        if text is not model:
            applied.notes.append("replacements were applied to the text decoder; the vision tower is untouched")
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
                        text, "Qwen3_5MoeSparseMoeBlock", lambda m: FusedExperts(m, decoder.num_experts_per_tok)
                    ),
                    expected["routed_experts"],
                )
            )
            applied.swaps.append(
                Swap(
                    "attention",
                    _swap_children(text, "Qwen3_5MoeAttention", FlashAttention),
                    expected["attention"],
                )
            )
            # Counted by neither of these two: found by structure, so verified against what they replace instead. The
            # normalisation is offered both scale conventions and keeps whichever agrees to a rounding step.
            _swap_and_verify(
                applied,
                text,
                "norm",
                "Qwen3_5MoeRMSNorm",
                [
                    ("weight+1", lambda m: FastRMSNorm(m, offset=True)),
                    ("weight", lambda m: FastRMSNorm(m, offset=False)),
                ],
                tolerance=2 * BF16_ULP,
            )
            _swap_and_verify(applied, text, "dense_matmul", None, [("fp8", Fp8Linear)], tolerance=5e-2)
        else:
            applied.skipped.append("vllm is not installed: the borrowed kernels are unavailable")

        dropped, declined = _drop_head_duplication(text)
        if declined is None:
            applied.swaps.append(Swap("head_duplication", dropped, expected["head_duplication"]))
        else:
            applied.skipped.append(f"head duplication left in place: {declined}")
        withheld = _withheld()
        paths, declined = (
            (0, "withheld by PRISMYRA_WITHOUT")
            if "gated_delta_rule" in withheld
            else _install_gated_delta_rule(decoder, next(model.parameters()).device)
        )
        if declined is None:
            applied.swaps.append(Swap("gated_delta_rule", paths, len(DELTA_PATHS)))
            applied.notes.append(
                "the recurrence kernel covers both passes and is where a branch pass spends itself: thirty of forty "
                "layers are gated delta nets and 93% of a pass's kernel launches came from them"
            )
        else:
            applied.skipped.append(f"the recurrence keeps the framework's path: {declined}")
        if not _install_conv():
            applied.skipped.append("triton is not available: the convolution keeps the framework's path")
        else:
            applied.swaps.append(Swap("convolution", _tag_conv_weights(text), expected["convolution"]))
            applied.notes.append(
                "the convolution kernel covers the context pass; a branch pass arrives as many rows and keeps the "
                "framework's path"
            )
        return applied


def _swap_and_verify(applied, root: nn.Module, name: str, class_name: str | None, candidates, tolerance: float) -> None:
    """Replace every module this pattern matches, then check one against the implementation it replaced.

    The check is the point. These two replacements are found by structure rather than by counting, so there is no
    number to compare against -- and a number would be wrong anyway, being a fact about one revision of somebody
    else's module tree. What can be compared is behaviour: run the original and the replacement on the same input and
    require them to agree. If they do not, every replacement of that kind is put back and the reason is recorded,
    because a kernel that changes answers is worth less than the milliseconds it saves.

    `tolerance` is relative to the largest output, because that is the only scale at which "the same answer" means
    anything. Bit-identity is the wrong bar even for a replacement that does identical arithmetic: this model's
    normalisation accumulates in float32 and rounds once at the end, while the borrowed kernel rounds earlier, and the
    two differ by one step of the output's own dtype -- 3.9e-3 relative in bfloat16. Demanding zero rejects a correct
    kernel; demanding one ulp accepts it and still rejects a wrong scale, which is off by a factor rather than a step.
    """
    if class_name is not None:
        targets = _find_children(root, lambda m: type(m).__name__ == class_name)
    else:
        targets = _find_children(root, _is_block_quantised)
    if not targets:
        applied.skipped.append(f"{name}: nothing matched")
        return

    # Each candidate is tried on one module before any of the rest is touched. More than one exists where the
    # arrangement cannot be read off the model -- see `FastRMSNorm` -- and choosing by measurement is the only way
    # that stays true across framework versions.
    parent, attribute, original = targets[0]
    tried = []
    for label, make in candidates:
        moved = _compare(original, make(original))
        tried.append(f"{label} {'raised' if moved is None else format(moved, '.3e')}")
        if moved is not None and moved <= tolerance:
            for p, a, child in targets:
                setattr(p, a, make(child))
            applied.swaps.append(
                Swap(name, len(targets), None, verified=f"{label}, {len(targets)} modules, agreed to {moved:.3e}")
            )
            return

    setattr(parent, attribute, original)
    applied.skipped.append(f"{name} left alone, nothing agreed to within {tolerance:.1e} relative: " + "; ".join(tried))


#: One step of bfloat16 at a given magnitude, relative. Two of these is the bar for a replacement that should be doing
#: the same arithmetic in a different order: it admits a different rounding point and excludes a different scale.
BF16_ULP = 2.0**-8


def _compare(original: nn.Module, replacement: nn.Module) -> float | None:
    """The largest disagreement on one random input, relative to the largest output. None if either side raised.

    Relative, not absolute. An absolute figure says nothing without the magnitude beside it: 6.25e-2 is a rounding
    step where the output reaches 16 and a wrong answer where it reaches 0.1.
    """
    p = next(original.parameters())
    width = getattr(original, "in_features", None) or p.shape[-1]
    x = torch.randn(4, width, device=p.device, dtype=torch.bfloat16) * 0.1
    try:
        with torch.inference_mode():
            want = original(x)
            got = replacement(x)
    except Exception:  # noqa: BLE001 - a replacement that cannot run on this model is a replacement to put back
        return None
    scale = want.float().abs().max().item()
    return (want.float() - got.float()).abs().max().item() / max(scale, 1e-6)


def _is_block_quantised(module: nn.Module) -> bool:
    """A two-dimensional fp8 projection carrying a two-dimensional block scale.

    Identified by structure, which excludes the expert stacks: those hold a leading expert dimension and no
    `weight`, and belong to the routed path.
    """
    w = getattr(module, "weight", None)
    if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
        return False
    s = getattr(module, "weight_scale_inv", None)
    return s is not None and getattr(s, "dim", lambda: 0)() == 2


def _find_children(root: nn.Module, matches) -> list[tuple[nn.Module, str, nn.Module]]:
    """Collected before anything is replaced, so the walk is not mutated under itself."""
    return [
        (parent, name, child) for parent in root.modules() for name, child in parent.named_children() if matches(child)
    ]


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


def _drop_head_duplication(model: nn.Module, verify: bool = True) -> tuple[int, str | None]:
    """Stop the linear-attention layers duplicating query and key, where the recurrence handles it itself.

    This one is a flag set through an attribute whose name stops describing its value. On the framework version it was
    measured against, `num_k_heads` was read in the forward pass only to decide whether to duplicate, so setting it
    equal to `num_v_heads` disabled the duplication and changed nothing else. That is not a property of the model but
    of one revision of somebody else's forward pass, so it is checked every time rather than assumed: one layer is run
    both ways and the outputs must match exactly.

    Returns the number of layers changed and, if none were, why. **Declining is not an error.** A later framework
    version uses the attribute for more than that -- on transformers 5.15 the probe raises a shape mismatch rather
    than returning a different answer -- and the right response is to leave the duplication in place and say so. It is
    worth 8.5 ms of a 138 ms pass; refusing to run at all over it would be a poor trade.
    """
    layers = [
        m
        for m in model.modules()
        if type(m).__name__ == "Qwen3_5MoeGatedDeltaNet" and getattr(m, "num_v_heads", 0) > getattr(m, "num_k_heads", 1)
    ]
    if not layers:
        return 0, "no layer duplicates its heads"
    if verify:
        declined = _probe_head_duplication(layers[0])
        if declined is not None:
            return 0, declined
    for m in layers:
        m.num_k_heads = m.num_v_heads
    return len(layers), None


def _probe_head_duplication(probe: nn.Module) -> str | None:
    """Run one layer both ways. Returns None when they match exactly, or the reason to decline."""
    p = next(probe.parameters())
    x = torch.randn(1, 64, probe.hidden_size, device=p.device, dtype=p.dtype) * 0.1
    original = probe.num_k_heads
    try:
        with torch.inference_mode():
            before = probe(x, cache_params=None)
            probe.num_k_heads = probe.num_v_heads
            after = probe(x, cache_params=None)
    except Exception as e:  # noqa: BLE001 - any failure here means the attribute now does more than gate duplication
        return f"the head-duplication probe raised {type(e).__name__}: {e}"
    finally:
        probe.num_k_heads = original

    before = before[0] if isinstance(before, tuple) else before
    after = after[0] if isinstance(after, tuple) else after
    moved = (before.float() - after.float()).abs().max().item()
    if moved != 0.0:
        return (
            f"skipping the head duplication moved a layer's output by {moved:.3e}; it was bit-identical when measured, "
            f"so this framework version uses the attribute for more than gating the duplication"
        )
    return None


register(Qwen3MoeAdapter())
