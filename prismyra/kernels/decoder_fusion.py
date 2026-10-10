"""Fuse a residual add with the RMSNorm that reads its result, across the decoder-layer boundary that separates them.

Why this crosses a module boundary other replacements in this package do not. The framework's own decoder layer
computes, after its MLP/MoE block: `hidden_states = residual + hidden_states`, a bare Python operator, and the next
thing to touch that tensor is the *next* layer's own `input_layernorm` -- a different module, holding a different
weight. `kernels.qwen3_moe.FastRMSNorm` already borrows vLLM's RMSNorm kernel for every norm in this model, but it only
ever sees one tensor (`x`), never the pair (`x`, `residual`) that `vllm._custom_ops.fused_add_rms_norm` fuses the add
into. Reaching this fusion means both decoder layers that share the boundary cooperate, which means the layer loop
that calls them has to carry the pending, not-yet-added residual between calls instead of each layer re-deriving it --
the pattern vLLM's own model implementations use throughout, and the one this framework's model does not.

What this does. Replaces `Qwen3_5MoeDecoderLayer.forward` (bound per instance, the framework's children -- attention,
recurrence, MLP, both norms -- stay exactly where they are) and `Qwen3_5MoeTextModel.forward` (same, bound per
instance) with versions that thread a `residual` tensor between layers instead of adding it in and reading it back out
each time, fusing every add this model does with the norm that reads its result: an decoder layer's post-attention
norm with the attention output's add, its *next* layer's pre-attention norm with the MLP output's add, and the
model's own final norm with the last layer's MLP output's add. Deliberately not counted as a `Swap` the way other
replacements are: there is no module to compare a count against (`forward` is overridden, not swapped for another
module), so correctness here rests entirely on the end-to-end probability check in `tests/test_gpu.py`,
not on a swap count.

Must run after `kernels.qwen3_moe`'s own norm replacement (`_swap_and_verify(..., "norm", "Qwen3_5MoeRMSNorm", ...)`):
this reuses that pass's already-verified `FastRMSNorm` instances (and the `weight+1` vs `weight` choice it measured)
rather than re-deriving the same choice a second time.
"""

from __future__ import annotations

import types

import torch
from torch import nn


class _FusedNorm(nn.Module):
    """One `FastRMSNorm`, offering the extra (x, residual) call `fused_add_rms_norm` needs. Not a replacement module
    in its own right -- it wraps an already-installed, already-verified `FastRMSNorm` and reuses its `weight_plus_one`
    and `eps` rather than re-deriving the offset choice."""

    def __init__(self, fast_rms_norm: nn.Module):
        super().__init__()
        self.inner = fast_rms_norm  # keeps the already-verified module alive and counted where it was counted

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        from vllm import _custom_ops as ops

        w = self.inner.weight_plus_one
        eps = self.inner.eps
        if residual is None:
            y = x if x.is_contiguous() else x.contiguous()
            out = torch.empty_like(y)
            ops.rms_norm(out, y, w, eps)
            return out, y
        x = x if x.is_contiguous() else x.contiguous()
        residual = residual if residual.is_contiguous() else residual.contiguous()
        ops.fused_add_rms_norm(x, residual, w, eps)  # in place: residual += x; x = rms_norm(residual) * w
        return x, residual


def _layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    residual=None,
    **kwargs,
):
    """Same arithmetic as the framework's `Qwen3_5MoeDecoderLayer.forward`, with both of its adds deferred into the
    norm that reads their result. Returns `(mlp_output, pending_residual)` instead of one added-up tensor; the caller
    (`_text_model_forward` below, or the next layer's own `residual=` argument) is what finally adds `pending_residual`
    in, inside that norm's fused kernel -- or, for the very last layer, inside the model's own final norm."""
    normed, residual = self._fused_input_layernorm(hidden_states, residual)

    if self.block_type == "linear_attention":
        attn_out = self.linear_attn(
            hidden_states=normed, cache_params=past_key_values, attention_mask=attention_mask, **kwargs
        )
    else:
        attn_out, _ = self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    normed2, residual = self._fused_post_attention_layernorm(attn_out, residual)
    mlp_out = self.mlp(normed2)
    if isinstance(mlp_out, tuple):
        mlp_out, _ = mlp_out
    return mlp_out, residual  # the caller still owes `residual += mlp_out` -- done by whoever norms it next


def _text_model_forward(
    self,
    hidden_states,
    position_embeddings,
    causal_mask_mapping,
    position_ids,
    past_key_values,
    use_cache,
    layer_types,
    num_hidden_layers,
    **kwargs,
):
    """Same loop as `Qwen3_5MoeTextModel.forward`'s layer loop and final norm, with the residual thread this module
    adds. Takes the already-prepared pieces (embeddings, masks, rope) so it does not have to re-derive anything the
    framework's own `forward` (kept as the entry point; see `install`) already computed and verified."""
    residual = None
    for i, decoder_layer in enumerate(self.layers[:num_hidden_layers]):
        hidden_states, residual = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask_mapping[layer_types[i]],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            residual=residual,
            **kwargs,
        )
    hidden_states, _ = self._fused_final_norm(hidden_states, residual)
    return hidden_states


def install(applied, text: nn.Module) -> None:
    """Bind the fused forwards onto every decoder layer and the text model itself, reusing the norm modules
    `kernels.qwen3_moe`'s own pass already replaced and verified. No-ops (and records why) if that pass left the
    plain framework norm in place, or if vLLM's fused kernel is unavailable."""
    from vllm import _custom_ops as ops

    if not hasattr(ops, "fused_add_rms_norm"):
        applied.skipped.append("decoder_fusion: vllm._custom_ops.fused_add_rms_norm is unavailable")
        return
    layers = getattr(text, "layers", None)
    final_norm = getattr(text, "norm", None)
    if layers is None or final_norm is None or type(final_norm).__name__ != "FastRMSNorm":
        applied.skipped.append("decoder_fusion: norm was not already replaced with FastRMSNorm (or no layers found)")
        return
    for layer in layers:
        in_norm, post_norm = getattr(layer, "input_layernorm", None), getattr(layer, "post_attention_layernorm", None)
        if type(in_norm).__name__ != "FastRMSNorm" or type(post_norm).__name__ != "FastRMSNorm":
            applied.skipped.append("decoder_fusion: a layer's own norms were not FastRMSNorm, left untouched")
            return
        layer._fused_input_layernorm = _FusedNorm(in_norm)
        layer._fused_post_attention_layernorm = _FusedNorm(post_norm)
        layer.forward = types.MethodType(_layer_forward, layer)
    text._fused_final_norm = _FusedNorm(final_norm)

    def wrapped_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        # Reruns the framework's own preamble (embeddings, rope, masks) unchanged, then hands the loop to the fused
        # version above instead of the framework's own layer-by-layer add-then-norm. Keeping the preamble as the one
        # true copy (calling it, not re-deriving it) means a framework upgrade that changes mask construction or rope
        # still gets picked up here.
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = _text_model_forward(
            self,
            hidden_states,
            position_embeddings,
            causal_mask_mapping,
            text_position_ids,
            past_key_values,
            use_cache,
            self.config.layer_types,
            self.config.num_hidden_layers,
            **kwargs,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    text.forward = types.MethodType(wrapped_forward, text)
    applied.notes.append(f"decoder_fusion: {len(layers)} layers' residual-add+RMSNorm fused across the layer boundary")
