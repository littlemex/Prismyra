"""S4c: layer-interleaved fork (SYNTHESIS.md P5, out/p1_speed_opus.md P5, RUN-fp8spd.md).

The two-pass design reads a context through all 36 layers (`engine._read`), then answers through all 36 layers
again (`engine._branch`) -- which re-streams every layer's dense and routed-expert weights for a branch pass that
may carry as few as one row. A microbenchmark on a real `FusedExperts` module with its own weights (RUN-fp8spd.md,
"P4(a) 射影結合の効果を単体カーネルで実測") found that restreaming costs 51.3% of a 16-row pass's time against one
call that already held the row count the context pass used; a second microbenchmark on the same module
(RUN-fp8spd.md, "S4c: 4層試作、実モデルの重みで検証") found concatenating a context's 5,000 rows with a branch's
16 or 64 rows into one call, instead of two, saved 29-30% of that layer's time and both halves of the combined
output were `torch.equal` to running them alone -- the expert computation does not depend on which other rows
share the call. Scaled from a 4-layer measurement to this model's 36, that is about 39 ms at both 16 and 64
questions, which is the saving this module exists to realise end to end rather than in a microbenchmark.

**What is fused and what is not.** Only the dense projections and the routed/shared-expert block (`decoder_layer.
mlp`, called on the context's and the branch's rows concatenated) are shared across one call per layer -- these
are the ops a microbenchmark showed to be free of any dependence on which rows share them (`torch.equal`,
RUN-fp8spd.md). The gated-delta-net recurrence and the attention read are the two things that genuinely cannot
share a call, because each carries state the other side must not see or must see *only* for one layer:

* the GDN layer's recurrent and convolution state is **this layer's own**, nothing a 36-layer snapshot would
  average or confuse with another layer's -- so the branch's own GDN call needs only what the context's own GDN
  call, run moments earlier at this same layer, just left behind. `fork.widen_for_branch` hands it over for
  exactly that one call and takes it back immediately after, so the next layer's context forward (a different
  layer object, with its own state) is unaffected and so is a second branch group forking later from this
  layer's snapshot (`fork.snapshot_layer`, taken between the two calls, before the widen);
* the attention layer (`cache.ForkLayer`) already keeps the context in one row and joins it with each branch
  row on read -- `begin_branches()` is the one call that flips a layer from writing the context to writing
  branches, and calling it per layer, immediately after that layer's own context write, is the only change this
  module makes there. Nothing about the join itself changes.

**Scope (first cut).** One document, one branch group of up to `group` rows, text only, the borrowed kernels
installed (`engine._borrowed_kernel`), the joined (non-paged) cache. A request needing a second group of
questions still gets one: `read_and_branch` returns a `Prefill` whose `snapshot` is the pure context state,
assembled one layer at a time as the loop goes rather than taken in a separate pass -- so `engine._branch` and
`fork.restore_and_fork`, both unmodified, serve every group after the first exactly as they do today. Extending
this to several documents at once (`open_batch`, `Shelf`) and to CUDA graphs is out of scope for this cut; see
RUN-fp8spd.md's design note for why and what each would need.
"""

from __future__ import annotations

import torch

from . import varlen
from .fork import Prefill, build_suffixes, snapshot_layer, widen_for_branch


def read_and_branch(engine, encoded, texts: list[str], width: int, group: int, padded_rows: int):
    """Read `encoded`'s context and answer `texts` (one branch group) in one layer-interleaved pass.

    `width` and `padded_rows` are the caller's own (`engine._packed_groups` / `engine._round_rows`): this
    function does not repeat that decision, so a document answered partly through this path and partly through
    `engine._branch` (a second group, or a caller that falls back) sees the identical padding both times --
    `_round_rows`'s own docstring is why a mismatch there would move an answer rather than raise.

    Returns `(hidden, prefill)`. `hidden` is `(len(texts), hidden_size)`, read at each row's own last real token
    -- the same shape and, `torch.equal`-gated (RUN-fp8spd.md), the same values `engine._branch` gives for this
    one group. `prefill` is this document's usual `Prefill`, snapshot included, built one layer at a time as the
    loop below runs rather than by a separate `fork.snapshot(cache)` call afterwards.
    """
    text_model = engine.backbone.language_model if hasattr(engine.backbone, "language_model") else engine.backbone
    layers = text_model.layers
    device = engine.device
    hidden_size = engine.hidden_size
    decoder = getattr(engine.config, "text_config", engine.config)
    layer_types = list(decoder.layer_types)

    ctx_tokens = encoded.tokens
    cache, room = engine._claim_cache(ctx_tokens, group=group)

    branch_ids, read_at, _ = build_suffixes(texts, engine.tokenizer, device, padded_rows, width)
    branch_width = branch_ids.shape[1]

    ctx_positions = torch.arange(ctx_tokens, device=device).unsqueeze(0)
    branch_positions = torch.arange(
        ctx_tokens, ctx_tokens + branch_width, device=device
    ).unsqueeze(0).expand(padded_rows, -1)

    hidden_ctx = text_model.embed_tokens(encoded.input_ids)
    hidden_branch = text_model.embed_tokens(branch_ids)
    ctx_rope = text_model.rotary_emb(hidden_ctx, ctx_positions)
    branch_rope = text_model.rotary_emb(hidden_branch, branch_positions)

    snap: dict[int, dict] = {}
    for i, decoder_layer in enumerate(layers):
        layer_cache = cache.layers[i]
        residual_ctx, residual_branch = hidden_ctx, hidden_branch
        normed_ctx = decoder_layer.input_layernorm(hidden_ctx)
        normed_branch = decoder_layer.input_layernorm(hidden_branch)

        if layer_types[i] == "linear_attention":
            # The context's own GDN+conv, in the same shape as `engine._read`'s own pass through this layer:
            # one row, `output_final_state=True` (no `varlen.branching()` here -- that flag is what tells the
            # borrowed kernel to *skip* writing the state back, and this call is the one call that must write
            # it).
            out_ctx = decoder_layer.linear_attn(normed_ctx, cache_params=cache, attention_mask=None)
            # This layer's pure context state, captured now -- before the widen below replaces the dict's
            # tensors with a wider view -- so a second branch group's fork later sees the one-row context
            # tensors, not this group's `padded_rows`-wide ones.
            snap[i] = snapshot_layer(layer_cache)
            with widen_for_branch(layer_cache, padded_rows, group), varlen.branching():
                out_branch = decoder_layer.linear_attn(normed_branch, cache_params=cache, attention_mask=None)
        else:
            out_ctx, _ = decoder_layer.self_attn(
                normed_ctx, position_embeddings=ctx_rope, attention_mask=None, past_key_values=cache
            )
            # `ForkLayer`'s own snapshot is nothing but its length -- see `fork.restore_and_fork`'s own
            # docstring, "there is nothing to restore beyond the length". Taken here, before `begin_branches()`
            # advances it further, for exactly the same reason as the GDN branch above.
            snap[i] = snapshot_layer(layer_cache)
            begin = getattr(layer_cache, "begin_branches", None)
            if begin is not None:
                begin()
            with varlen.branching():
                out_branch, _ = decoder_layer.self_attn(
                    normed_branch, position_embeddings=branch_rope, attention_mask=None, past_key_values=cache
                )

        hidden_ctx = residual_ctx + out_ctx
        hidden_branch = residual_branch + out_branch

        # The one call this module exists for: the dense projections inside `mlp` (the shared expert) and the
        # routed experts, on the context's and the branch's rows concatenated -- one weight stream instead of
        # two. `torch.equal`-verified against running the two halves separately (RUN-fp8spd.md).
        residual_ctx, residual_branch = hidden_ctx, hidden_branch
        normed2_ctx = decoder_layer.post_attention_layernorm(hidden_ctx)
        normed2_branch = decoder_layer.post_attention_layernorm(hidden_branch)
        combined = torch.cat(
            (normed2_ctx.reshape(-1, hidden_size), normed2_branch.reshape(-1, hidden_size)), dim=0
        )
        moe_out = decoder_layer.mlp(combined)
        if isinstance(moe_out, tuple):  # the unpatched block returns (output, router_logits); FusedExperts does not
            moe_out = moe_out[0]
        moe_ctx, moe_branch = moe_out.split([ctx_tokens, padded_rows * branch_width], dim=0)
        hidden_ctx = residual_ctx + moe_ctx.reshape(1, ctx_tokens, hidden_size)
        hidden_branch = residual_branch + moe_branch.reshape(padded_rows, branch_width, hidden_size)

    # The context's own final norm is never read -- nothing downstream of `engine._read` reads its hidden state
    # either, only the cache it leaves behind, which the loop above has already written layer by layer.
    hidden_branch = text_model.norm(hidden_branch)
    out = hidden_branch[torch.arange(padded_rows, device=device), read_at][: len(texts)]

    prefill = Prefill(
        snapshot=snap,
        cache=cache,
        room=room,
        tokens=ctx_tokens,
        last_position=torch.tensor([ctx_tokens - 1], device=device),
        position_from=ctx_tokens,
        group=group,
    )
    return out, prefill
