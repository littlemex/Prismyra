"""Layer-interleaved fork.

The two-pass design reads a context through all 36 layers (`engine._read`), then answers through all 36 layers
again (`engine._branch`) -- which re-streams every layer's dense and routed-expert weights for a branch pass that
may carry as few as one row. A microbenchmark on a real `FusedExperts` module with its own weights found that
restreaming costs 51.3% of a 16-row pass's time against one call that already held the row count the context
pass used; a second microbenchmark on the same module found concatenating a context's 5,000 rows with a branch's
16 or 64 rows into one call, instead of two, saved 29-30% of that layer's time and both halves of the combined
output were `torch.equal` to running them alone -- the expert computation does not depend on which other rows
share the call. Scaled from a 4-layer measurement to this model's 36, that is about 39 ms at both 16 and 64
questions, which is the saving this module exists to realise end to end rather than in a microbenchmark.

**What is fused and what is not.** Only the dense projections and the routed/shared-expert block (`decoder_layer.
mlp`, called on the context's and the branch's rows concatenated) are shared across one call per layer -- these
are the ops a microbenchmark showed to be free of any dependence on which rows share them (`torch.equal`-verified).
The gated-delta-net recurrence and the attention read are the two things that genuinely cannot
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
this to several documents at once (`open_batch`, `Shelf`) and to CUDA graphs is out of scope for this cut.
"""

from __future__ import annotations

import torch

from . import varlen
from .fork import Prefill, build_suffixes, pick, snapshot_bytes, snapshot_layer, widen_for_branch, widen_for_branch_many


def read_and_branch(engine, encoded, texts: list[str], width: int, group: int, padded_rows: int):
    """Read `encoded`'s context and answer `texts` (one branch group) in one layer-interleaved pass.

    `width` and `padded_rows` are the caller's own (`engine._packed_groups` / `engine._round_rows`): this
    function does not repeat that decision, so a document answered partly through this path and partly through
    `engine._branch` (a second group, or a caller that falls back) sees the identical padding both times --
    `_round_rows`'s own docstring is why a mismatch there would move an answer rather than raise.

    Returns `(hidden, prefill)`. `hidden` is `(len(texts), hidden_size)`, read at each row's own last real token
    -- the same shape and, `torch.equal`-gated, the same values `engine._branch` gives for this
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
        # two. `torch.equal`-verified against running the two halves separately.
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


def read_and_branch_shelf(engine, shelf, context: str, texts: list[str], width: int, padded_rows: int):
    """`read_and_branch`'s counterpart for `Shelf`/`Batcher`: fuse one *fresh* document's read into the
    shelf with its first (and, in `Batcher`, only -- a request's own questions already fit one group,
    `schedule.Limits.questions`) branch group, through the same per-layer loop.

    **Scope.** Exactly one document, read for the first time this call (an already-resident document, or a pass
    naming more than one document, still goes through `Shelf.put_many`/`Shelf.ask` unchanged -- see
    `schedule.Batcher._answer`, the only caller). Text only, same restriction `read_and_branch` states.

    **Why the paged cache needed nothing new for the GDN layers.** `cache.build_cache` gives every cache the
    *same* recurrent/convolution storage (the framework's own layer class) whether `paged` is on or off -- only
    the attention layers switch between `ForkLayer` and `PagedForkLayer`. So `fork.widen_for_branch` and
    `fork.snapshot_layer`, written against that storage, run here completely unchanged. What *is* new:

    * **admission.** A paged attention layer has to be told which document it is about to write
      (`begin_document`), which the joined `ForkLayer` never needed (a joined cache only ever holds one).
    * **the branch read.** `PagedForkLayer.begin_branches` takes a `rows_for` argument (`fork._takes_rows` is the
      existing dispatch this project already uses in `restore_and_fork_many` to tell it apart from the joined
      layer's own no-argument version) -- naming which rows answer about this document, needed because a page
      table has more than one document's pages to choose from in general, even though this cut only ever puts
      one document's rows in `rows_for`.
    * **the padding segment.** `Shelf.put_many` always reads a document (even alone) alongside a padding
      segment, through `engine._pad_context_lengths`, so that the borrowed chunked recurrent kernel -- whose
      configuration is chosen by a read's *total* length, not by what it contains (`engine._read`'s own
      comment) -- picks the same configuration whatever else later shares this shelf's reads. That makes the
      context forward a *two-document* flat run (the real document, then padding), which this function keeps as
      one flat `(1, tokens, hidden)` tensor exactly as `read_and_branch`'s own single-document `hidden_ctx` is --
      a flat run does not care how many logical documents it holds. Two things do: each GDN layer's own
      `recurrent_states`/`conv_states` come back from the framework **one row per document** once
      `varlen.reading()` is active (the same reason `engine._check_batched_read` exists), and this project's own
      `_put_back_conv_states` exists because the framework's convolution state is otherwise sliced from the
      *end* of the flat run -- the padding's end, not the real document's. Both are narrowed to the real
      document's own row, right here, layer by layer, rather than in the bulk pass `Shelf.put_many` runs after
      its own single whole-model call returns: the tail each convolution needs is appended to
      `varlen.current().conv_tails`, in layer order, the moment that layer's own borrowed kernel runs
      (`kernels/qwen3_moe.py`'s patched `causal_conv1d_fn`) -- already correct and already available by the time
      this loop reaches that layer, which is what makes computing it layer-by-layer rather than in one pass
      afterward exactly as correct as `Shelf.put_many`'s own bulk version, not an approximation of it.

    Returns `(hidden, handle, shelved)`: `hidden` is this group's answer, same shape and `torch.equal`-gated
    contract as `read_and_branch`'s own return; `handle` is the shelf handle the real document was given;
    `shelved` is the `engine.Shelved` the caller stores at `shelf.documents[handle]`, built the same way
    `Shelf.put_many` builds one.
    """
    from .engine import Shelved, _forget_recurrent_state

    text_model = engine.backbone.language_model if hasattr(engine.backbone, "language_model") else engine.backbone
    layers = text_model.layers
    device = engine.device
    hidden_size = engine.hidden_size
    decoder = getattr(engine.config, "text_config", engine.config)
    layer_types = list(decoder.layer_types)

    cache = shelf._cache
    encoded = engine.encode_context(context)
    if encoded.has_media:
        raise ValueError("a shelf is text only for now, for the same reason a batch is: media move the positions")

    _forget_recurrent_state(cache)

    handle = shelf._next_handle
    shelf._next_handle += 1
    pad_ids, lengths = engine._pad_context_lengths([encoded.tokens], [encoded.input_ids])
    has_pad = pad_ids.shape[1] > 0
    pad_handle = None
    if has_pad:
        pad_handle = shelf._next_handle
        shelf._next_handle += 1
    doc_handles = [handle, pad_handle] if has_pad else [handle]
    ctx_ids = torch.cat([encoded.input_ids, pad_ids], dim=1) if has_pad else encoded.input_ids
    ctx_tokens = ctx_ids.shape[1]

    branch_ids, read_at, _ = build_suffixes(texts, engine.tokenizer, device, padded_rows, width)
    branch_width = branch_ids.shape[1]
    hidden_branch = text_model.embed_tokens(branch_ids)
    branch_positions = torch.arange(
        encoded.tokens, encoded.tokens + branch_width, device=device
    ).unsqueeze(0).expand(padded_rows, -1)
    branch_rope = text_model.rotary_emb(hidden_branch, branch_positions)

    for layer in cache.layers:
        begin = getattr(layer, "begin_documents", None)
        if begin is not None:
            begin(doc_handles)

    hidden_ctx = text_model.embed_tokens(ctx_ids)
    # Pure arithmetic on `lengths` -- the same thing `Boundaries.positions` computes -- so this needs no
    # `varlen.reading()` entered yet: positions restart at zero for the padding document, same as any other.
    ctx_positions = (
        torch.cat([torch.arange(n, device=device) for n in lengths]).unsqueeze(0)
        if has_pad
        else torch.arange(ctx_tokens, device=device).unsqueeze(0)
    )
    ctx_rope = text_model.rotary_emb(hidden_ctx, ctx_positions)

    snap: dict[int, dict] = {}
    for i, decoder_layer in enumerate(layers):
        layer_cache = cache.layers[i]
        residual_ctx, residual_branch = hidden_ctx, hidden_branch
        normed_ctx = decoder_layer.input_layernorm(hidden_ctx)
        normed_branch = decoder_layer.input_layernorm(hidden_branch)

        if layer_types[i] == "linear_attention":
            # `varlen.reading()` scoped to just this call, not the branch call below: the borrowed kernel reads
            # `varlen.current()` on *every* call regardless of which flag set it, so leaving the two-document
            # boundaries active for the branch call too (entering once for the whole function, as
            # `read_and_branch`'s single-document version has no reason to distinguish) made the branch's
            # `padded_rows` rows look like a mismatched *document* count to the framework ("expected 2 initial
            # states... rather than 1").
            if has_pad:
                with varlen.reading(lengths, device) as boundaries:
                    out_ctx = decoder_layer.linear_attn(normed_ctx, cache_params=cache, attention_mask=None)
                    # The framework's convolution state is sliced from the end of the flat run (the padding's
                    # end); the tail this layer's own borrowed kernel already recorded, per document, is correct
                    # (see this function's own docstring) and is still in scope here, before `reading()` exits
                    # and clears it. Applied now, one layer early relative to `engine._put_back_conv_states`'s
                    # own bulk pass, because `widen_for_branch` below needs the real document's row of the
                    # *corrected* one and nothing later in this loop needs the uncorrected value.
                    conv = getattr(layer_cache, "conv_states", None)
                    if isinstance(conv, dict) and boundaries.conv_tails:
                        tail = boundaries.conv_tails[-1]
                        if tail.shape[0] != len(doc_handles):
                            raise ValueError(
                                f"this layer's convolution recorded a tail for {tail.shape[0]} documents and "
                                f"{len(doc_handles)} were read -- a document's row would be the wrong one"
                            )
                        for key, held in list(conv.items()):
                            conv[key] = tail[:1].to(dtype=held.dtype) if held is not None else tail[:1]
                    # The recurrence itself already returns one row per document once `varlen.reading()` is
                    # active (the same guarantee `engine._check_batched_read` checks for the ordinary
                    # batched-read path; checked by hand here, inline, because `_check_batched_read` wants the
                    # *unnarrowed* row count and this narrows it immediately).
                    rec = getattr(layer_cache, "recurrent_states", None)
                    if isinstance(rec, dict):
                        for key, held in list(rec.items()):
                            if held is None:
                                continue
                            if held.shape[0] != len(doc_handles):
                                raise ValueError(
                                    f"this layer's recurrent state has {held.shape[0]} rows and "
                                    f"{len(doc_handles)} documents were read -- a document's row would be the "
                                    f"wrong one"
                                )
                            rec[key] = held[:1].clone()
            else:
                out_ctx = decoder_layer.linear_attn(normed_ctx, cache_params=cache, attention_mask=None)
            snap[i] = snapshot_layer(layer_cache)
            with widen_for_branch(layer_cache, padded_rows, padded_rows), varlen.branching():
                out_branch = decoder_layer.linear_attn(normed_branch, cache_params=cache, attention_mask=None)
        else:
            if has_pad:
                with varlen.reading(lengths, device):
                    out_ctx, _ = decoder_layer.self_attn(
                        normed_ctx, position_embeddings=ctx_rope, attention_mask=None, past_key_values=cache
                    )
            else:
                out_ctx, _ = decoder_layer.self_attn(
                    normed_ctx, position_embeddings=ctx_rope, attention_mask=None, past_key_values=cache
                )
            snap[i] = snapshot_layer(layer_cache)
            begin_branches = getattr(layer_cache, "begin_branches", None)
            if begin_branches is not None:
                begin_branches([handle] * padded_rows)
            with varlen.branching():
                out_branch, _ = decoder_layer.self_attn(
                    normed_branch, position_embeddings=branch_rope, attention_mask=None, past_key_values=cache
                )

        hidden_ctx = residual_ctx + out_ctx
        hidden_branch = residual_branch + out_branch

        residual_ctx, residual_branch = hidden_ctx, hidden_branch
        normed2_ctx = decoder_layer.post_attention_layernorm(hidden_ctx)
        normed2_branch = decoder_layer.post_attention_layernorm(hidden_branch)
        combined = torch.cat(
            (normed2_ctx.reshape(-1, hidden_size), normed2_branch.reshape(-1, hidden_size)), dim=0
        )
        moe_out = decoder_layer.mlp(combined)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        moe_ctx, moe_branch = moe_out.split([ctx_tokens, padded_rows * branch_width], dim=0)
        hidden_ctx = residual_ctx + moe_ctx.reshape(1, ctx_tokens, hidden_size)
        hidden_branch = residual_branch + moe_branch.reshape(padded_rows, branch_width, hidden_size)

    hidden_branch = text_model.norm(hidden_branch)
    out = hidden_branch[torch.arange(padded_rows, device=device), read_at][: len(texts)]

    for layer in cache.layers:
        finish = getattr(layer, "finish_branches", None)
        if finish is not None:
            finish()
    if has_pad:
        for layer in cache.layers:
            release = getattr(layer, "release_document", None)
            if release is not None:
                release(pad_handle)

    shelved = Shelved(
        handle=handle,
        tokens=encoded.tokens,
        snapshot=snap,
        position_from=encoded.tokens,
        snapshot_bytes=snapshot_bytes(snap),
    )
    return out, handle, shelved


def read_and_branch_shelf_many(
    engine, shelf, contexts: list[str], texts_per_doc: list[list[str]], width: int, padded_rows_per_doc: list[int]
):
    """`read_and_branch_shelf`'s own job for several *fresh* documents at once: fuse `N` documents' reads into
    the shelf with each one's own (first, and in `Batcher`, only -- see that function's own docstring) branch
    group, through one per-layer loop that carries every document's rows at once instead of one.

    **Scope.** `N >= 2` documents, every one read for the first time this call -- a pass naming an already
    resident document, or exactly one fresh document, still goes through `Shelf.put_many`/`Shelf.ask` or
    `read_and_branch_shelf` respectively (see `schedule.Batcher._answer`, the only caller, for which path a
    pass takes). Text only, same restriction `read_and_branch`/`read_and_branch_shelf` state.

    **Why each document's own branch rows are rounded independently.** `_branch_across` (the existing,
    non-interleaved multi-document branch pass) rounds the *combined* row count once and pads only the last
    document -- which is exactly the design that creates a residual: a document's own answer
    moving when a companion's question count changes the bucket the *pair* rounds to, not the bucket either
    one would round to alone. This function rounds each
    document's `padded_rows_per_doc[i]` on its own count alone (the caller already did this, in
    `schedule.Batcher._answer`'s own `_round_rows` call per job) and never recombines it with anyone else's --
    nothing here depends on how many rows a companion asked for, which is this module's own invariance
    guarantee by construction rather than by a later patch.

    **The one new primitive this needed.** `fork.widen_for_branch_many`:
    `widen_for_branch`'s one-document broadcast, generalised to broadcast document `i`'s own just-written row
    to its own `padded_rows_per_doc[i]` branch rows and nobody else's, for every document in one call -- the
    same guarantee `widen_for_branch` gives a single document, extended rather than approximated.

    **Everything else is `open_batch`'s own admission (`begin_documents`, one shared padding segment sized by
    the *combined* total exactly as `Prismyra._pad_context_lengths` already does for any batched read) plus
    `read_and_branch_shelf`'s own per-layer narrowing of the framework's end-of-run convolution tail and
    its one-document-per-row recurrent state** (see that function's own docstring, which explains both), run in
    a loop over `N` documents instead of written out for one.

    Returns `(hidden, handles, shelved)`: `hidden` is every document's own answer rows concatenated in the
    order `contexts` was given, each trimmed to its own real question count -- the same shape and
    `torch.equal`-gated contract `read_and_branch`/`read_and_branch_shelf` give for one document, extended to
    `N`; `handles` and `shelved` are what the caller stores on the shelf, one per document, in that same order.
    """
    from .engine import Shelved, _forget_recurrent_state

    text_model = engine.backbone.language_model if hasattr(engine.backbone, "language_model") else engine.backbone
    layers = text_model.layers
    device = engine.device
    hidden_size = engine.hidden_size
    decoder = getattr(engine.config, "text_config", engine.config)
    layer_types = list(decoder.layer_types)

    n = len(contexts)
    if n < 2:
        raise ValueError(
            "read_and_branch_shelf_many needs at least two fresh documents; use read_and_branch_shelf for one"
        )
    if len(texts_per_doc) != n or len(padded_rows_per_doc) != n:
        raise ValueError(f"{n} documents need {n} question lists and {n} padded-row counts, not "
                          f"{len(texts_per_doc)} and {len(padded_rows_per_doc)}")

    cache = shelf._cache
    encoded = [engine.encode_context(c) for c in contexts]
    if any(e.has_media for e in encoded):
        raise ValueError("a shelf is text only for now, for the same reason a batch is: media move the positions")

    _forget_recurrent_state(cache)

    handles = list(range(shelf._next_handle, shelf._next_handle + n))
    shelf._next_handle += n
    lengths = [e.tokens for e in encoded]
    pad_ids, lengths = engine._pad_context_lengths(lengths, [e.input_ids for e in encoded])
    has_pad = pad_ids.shape[1] > 0
    pad_handle = None
    if has_pad:
        pad_handle = shelf._next_handle
        shelf._next_handle += 1
    doc_handles = [*handles, pad_handle] if has_pad else list(handles)
    ctx_ids = torch.cat([*(e.input_ids for e in encoded), pad_ids], dim=1) if has_pad else torch.cat(
        [e.input_ids for e in encoded], dim=1
    )
    ctx_tokens = ctx_ids.shape[1]

    branch_blocks, read_ats = [], []
    for i in range(n):
        ids_i, read_at_i, _ = build_suffixes(texts_per_doc[i], engine.tokenizer, device, padded_rows_per_doc[i], width)
        branch_blocks.append(ids_i)
        read_ats.append(read_at_i)
    branch_ids_cat = torch.cat(branch_blocks, dim=0)
    branch_width = branch_ids_cat.shape[1]
    total_branch_rows = sum(padded_rows_per_doc)

    for layer in cache.layers:
        begin = getattr(layer, "begin_documents", None)
        if begin is not None:
            begin(doc_handles)

    hidden_ctx = text_model.embed_tokens(ctx_ids)
    hidden_branch = text_model.embed_tokens(branch_ids_cat)
    # Pure arithmetic on `lengths` (`Boundaries.positions` would compute the same thing once `varlen.reading()`
    # is entered, but rope needs this before the layer loop starts): each real document's own tokens start at
    # zero, same as `open_batch`'s own positions for a batched read, and so does the shared padding segment's.
    ctx_positions = torch.cat([torch.arange(ln, device=device) for ln in lengths]).unsqueeze(0)
    ctx_rope = text_model.rotary_emb(hidden_ctx, ctx_positions)
    # Each document's own branch rows start at that document's own end -- `encoded[i].tokens`, not the shared
    # padding's end and not another document's -- same per-row-start-position reasoning `_branch_across` uses
    # for a mixed batch.
    starts = torch.cat(
        [torch.full((padded_rows_per_doc[i],), encoded[i].tokens, device=device, dtype=torch.long) for i in range(n)]
    )
    branch_positions = starts.unsqueeze(1) + torch.arange(branch_width, device=device).unsqueeze(0)
    branch_rope = text_model.rotary_emb(hidden_branch, branch_positions)

    # Which document each branch row answers about, in row order -- `restore_and_fork_many`'s own `rows_for`
    # contract, handed to the paged attention layer's `begin_branches` below.
    rows_for = [h for i, h in enumerate(handles) for _ in range(padded_rows_per_doc[i])]

    snap: dict[int, dict] = {}
    for i, decoder_layer in enumerate(layers):
        layer_cache = cache.layers[i]
        residual_ctx, residual_branch = hidden_ctx, hidden_branch
        normed_ctx = decoder_layer.input_layernorm(hidden_ctx)
        normed_branch = decoder_layer.input_layernorm(hidden_branch)

        if layer_types[i] == "linear_attention":
            with varlen.reading(lengths, device) as boundaries:
                out_ctx = decoder_layer.linear_attn(normed_ctx, cache_params=cache, attention_mask=None)
                # The framework's convolution state is sliced from the end of the flat run (the padding's end,
                # when there is one); the tail this layer's own borrowed kernel already recorded, one row per
                # document in `doc_handles` order, is correct and still in scope here, before `reading()`
                # exits and clears it (`read_and_branch_shelf`'s own docstring explains the same narrowing for
                # one document; this keeps the first `n` real documents' rows and drops the padding's).
                conv = getattr(layer_cache, "conv_states", None)
                if isinstance(conv, dict) and boundaries.conv_tails:
                    tail = boundaries.conv_tails[-1]
                    if tail.shape[0] != len(doc_handles):
                        raise ValueError(
                            f"this layer's convolution recorded a tail for {tail.shape[0]} documents and "
                            f"{len(doc_handles)} were read -- a document's row would be the wrong one"
                        )
                    for key, held in list(conv.items()):
                        conv[key] = tail[:n].to(dtype=held.dtype) if held is not None else tail[:n]
                rec = getattr(layer_cache, "recurrent_states", None)
                if isinstance(rec, dict):
                    for key, held in list(rec.items()):
                        if held is None:
                            continue
                        if held.shape[0] != len(doc_handles):
                            raise ValueError(
                                f"this layer's recurrent state has {held.shape[0]} rows and "
                                f"{len(doc_handles)} documents were read -- a document's row would be the "
                                f"wrong one"
                            )
                        rec[key] = held[:n].clone()
            snap[i] = snapshot_layer(layer_cache)
            parts = [(doc_row, padded_rows_per_doc[doc_row]) for doc_row in range(n)]
            with widen_for_branch_many(layer_cache, parts), varlen.branching():
                out_branch = decoder_layer.linear_attn(normed_branch, cache_params=cache, attention_mask=None)
        else:
            with varlen.reading(lengths, device):
                out_ctx, _ = decoder_layer.self_attn(
                    normed_ctx, position_embeddings=ctx_rope, attention_mask=None, past_key_values=cache
                )
            snap[i] = snapshot_layer(layer_cache)
            begin_branches = getattr(layer_cache, "begin_branches", None)
            if begin_branches is not None:
                begin_branches(rows_for)
            with varlen.branching():
                out_branch, _ = decoder_layer.self_attn(
                    normed_branch, position_embeddings=branch_rope, attention_mask=None, past_key_values=cache
                )

        hidden_ctx = residual_ctx + out_ctx
        hidden_branch = residual_branch + out_branch

        residual_ctx, residual_branch = hidden_ctx, hidden_branch
        normed2_ctx = decoder_layer.post_attention_layernorm(hidden_ctx)
        normed2_branch = decoder_layer.post_attention_layernorm(hidden_branch)
        combined = torch.cat(
            (normed2_ctx.reshape(-1, hidden_size), normed2_branch.reshape(-1, hidden_size)), dim=0
        )
        moe_out = decoder_layer.mlp(combined)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        moe_ctx, moe_branch = moe_out.split([ctx_tokens, total_branch_rows * branch_width], dim=0)
        hidden_ctx = residual_ctx + moe_ctx.reshape(1, ctx_tokens, hidden_size)
        hidden_branch = residual_branch + moe_branch.reshape(total_branch_rows, branch_width, hidden_size)

    hidden_branch = text_model.norm(hidden_branch)

    outs, at = [], 0
    for i in range(n):
        pr = padded_rows_per_doc[i]
        block = hidden_branch[at : at + pr]
        outs.append(block[torch.arange(pr, device=device), read_ats[i]][: len(texts_per_doc[i])])
        at += pr
    out = torch.cat(outs, dim=0)

    for layer in cache.layers:
        finish = getattr(layer, "finish_branches", None)
        if finish is not None:
            finish()
    if has_pad:
        for layer in cache.layers:
            release = getattr(layer, "release_document", None)
            if release is not None:
                release(pad_handle)

    shelved = []
    for i in range(n):
        piece = pick(snap, i)
        shelved.append(
            Shelved(
                handle=handles[i],
                tokens=encoded[i].tokens,
                snapshot=piece,
                position_from=encoded[i].tokens,
                snapshot_bytes=snapshot_bytes(piece),
            )
        )
    return out, handles, shelved
