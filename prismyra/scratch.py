"""An external, question-local scratchpad, updated by looping one existing layer: an experiment, not a feature.

This is the first mechanism both independent reviews (Sol and Astra, `air/loop/out_p2_which_wins_*.md`) ranked
highest among the brainstormed options in `air/loop/BRIEF.md`: give each question `K` extra, writable positions --
a work area distinct from the document's own token positions and from the read-out position `loop.py` already
loops -- and update them by replaying an existing full-attention layer `r` times, reading the document's keys and
values but never writing them.

The technique is `loop.py`'s own, generalised from one self-referential position to `K` scratch positions:

* `loop.py` replaces **one** position's input to layer `k` with a blend of that position's own first-pass input and
  the *same* position's final hidden state from the previous loop, then re-runs the whole one-pass read from an
  empty cache (the only way to re-enter the middle of a fixed HF decoder stack without hand-rolling every layer's
  own cache/position-id plumbing, including the 27 GDN layers' recurrent and convolution state). This project's own
  two closed determinism bugs were in exactly that kind of hand-rolled machinery, so re-using the framework's own
  forward call for every loop, and letting a forward hook do the one substitution, was the deliberate choice there,
  and is kept here.
* This module does the same thing to `K` positions that are not the read-out position: they are appended, as
  **embeddings with no vocabulary identity**, after the document and the question's own tokens, and the read-out
  position is simply the last of them. Causal attention lets it (and every scratch position before it) read every
  earlier scratch position, the question, and the document -- never the other way around, so the document and the
  question's own tokens are read-only in the sense the brief asks for: nothing about their *input* changes between
  loops, even though re-running the whole forward call recomputes their output every time (wasted, not wrong).

Initial value (BRIEF rule 2: must depend on the question's identity, not on fork packing order or wall-clock time):
slot `i` of `K` starts at `q_repr + slot_id(i)`, where `q_repr` is the mean of the question's own token embeddings
(so it depends only on the question's rendered text) and `slot_id(i)` is a fixed sinusoidal vector of the slot's
*index* (the standard Transformer positional-encoding formula, with no learned or random part). Neither term depends
on which other questions share a pass or on anything about the serving process.

GDN layers are never looped: `layer_k` must be a `full_attention` layer (checked against `decoder.layer_types`), and
only that one layer's own input is replaced; every other layer, including the 27 GDN layers before and after it,
runs in its normal, single-pass order on every one of the `r` re-runs.

Scope kept narrow on purpose: like `loop.loop_one_pass`, this reads a single question through the one-pass path
(`engine.backbone(inputs_embeds=..., ...)`, no snapshot, no `fork` row-packing), not the production `fork`-batched
path. `RUN-loop1.md` section 4 found a ~0.16-point gap between the two paths on this checkpoint; that is noise this
module inherits, not something it introduces. A full production integration (batched fork rows of mixed widths, the
paged attention cache, the `Shelf`-resident path) is out of scope for this probe and is called out as unaudited in
`RUN-ws1.md` rather than silently assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .cache import build_cache
from .fork import WIDTHS, branch_ids
from .loop import decoder_parts
from .media import encode
from .readout import plan, score
from .schema import Question


def full_attention_layers(decoder) -> list[int]:
    """1-based layer numbers whose `layer_types` entry is `full_attention`, in this checkpoint's own layer order."""
    layer_types = list(getattr(decoder, "layer_types", []) or [])
    if not layer_types:
        raise RuntimeError("decoder has no layer_types; cannot tell full-attention layers from GDN layers")
    return [i + 1 for i, kind in enumerate(layer_types) if kind == "full_attention"]


def slot_init(k_positions: int, width: int, device, dtype: torch.dtype) -> torch.Tensor:
    """`K` deterministic slot-identifier vectors (standard sinusoidal position encoding of the slot's own index).

    Depends only on `k_positions` and `width`: no RNG, no question content, no batch state. This is the "fixed slot
    identifier" half of a scratch position's initial value; the other half is the question's own representation,
    added by the caller.
    """
    position = torch.arange(k_positions, dtype=torch.float32, device=device).unsqueeze(1)
    dim = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0)
    angle = position / torch.pow(10000.0, (2 * torch.div(dim, 2, rounding_mode="floor")) / width)
    pe = torch.zeros((k_positions, width), dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe.to(dtype)


@dataclass
class ScratchResult:
    """Probabilities per (layer_k, k_positions, pass). Pass 0 is `base`: the plain read with an untouched, r=0
    scratchpad (slots present and attended to, but layer `layer_k` never re-run on them)."""

    options: list[str]
    base: list[float]
    passes: dict = field(default_factory=dict)
    base_ms: float = 0.0
    pass_ms: dict = field(default_factory=dict)
    scratch_final: dict = field(default_factory=dict)  # (layer_k, k_positions, pass) -> cloned float32 tensor


class _ScratchHooks:
    """Capture and optionally replace `count` consecutive positions' input to one layer, and capture the same
    positions' input to the final norm. The block-of-K generalisation of `loop._Hooks`."""

    def __init__(self, layers, norm, layer_k: int):
        self.layer_k = layer_k
        self.start = 0
        self.count = 0
        self.input_at_k: torch.Tensor | None = None
        self.final: torch.Tensor | None = None
        self.replace: torch.Tensor | None = None
        self.active = False
        self.handles = [
            layers[layer_k - 1].register_forward_pre_hook(self._pre, with_kwargs=True),
            norm.register_forward_pre_hook(self._norm, with_kwargs=True),
        ]

    def _slice(self) -> slice:
        return slice(self.start, self.start + self.count)

    def _pre(self, module, args, kwargs):
        if not self.active:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        sl = self._slice()
        if self.replace is not None:
            hs = hs.clone()
            hs[0, sl] = self.replace.to(hs.dtype)
            if args:
                args = (hs, *args[1:])
            else:
                kwargs = {**kwargs, "hidden_states": hs}
        self.input_at_k = hs[0, sl].detach().float().clone()
        return args, kwargs

    def _norm(self, module, args, kwargs):
        if not self.active:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        self.final = hs[0, self._slice()].detach().float().clone()
        return None

    def close(self):
        for h in self.handles:
            h.remove()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def scratch_loop(
    engine,
    context: str,
    question: Question,
    layer_k: int,
    k_positions: int,
    r: int,
    alpha: float = 1.0,
) -> ScratchResult:
    """Read `question` about `context` with `k_positions` extra scratch positions appended after its own text, then
    re-run layer `layer_k..L` at those positions `r` times. Returns the plain (r=0) read and, for each pass
    `1..r`, the probabilities and the scratch's own final hidden state (for the clear/swap check in stage C).

    One question, one GPU call per pass (the one-pass path, not `fork`'s row-packed path -- see module docstring).
    `layer_k` must be 1-based and name a `full_attention` layer; the caller is expected to have checked that against
    `full_attention_layers(decoder)` once per process, not per question.
    """
    decoder = getattr(engine.config, "text_config", engine.config)
    layers, norm = decoder_parts(engine.backbone, decoder.num_hidden_layers)
    layer_types = list(getattr(decoder, "layer_types", []) or [])
    if layer_types and layer_types[layer_k - 1] != "full_attention":
        raise ValueError(f"layer {layer_k} is {layer_types[layer_k - 1]!r}, not full_attention")

    planned = plan(question, engine.tokenizer)
    suffix = branch_ids(planned.text, engine.tokenizer)
    device = engine.torch_device
    width = decoder.hidden_size

    def read(hidden_normed) -> list[float]:
        (p,) = engine.heads.apply(
            hidden_normed, [question.options], score(hidden_normed, engine.unembedding, [planned.token_ids], None)
        )
        return p.float().tolist()

    hooks = _ScratchHooks(layers, norm, layer_k)
    try:
        with engine._lock, torch.inference_mode():
            encoded = encode(context, None, None, engine.processor, engine.tokenizer, engine.device)
            embed_layer = engine.backbone.get_input_embeddings()
            suffix_ids = torch.tensor([suffix], device=engine.device)
            ids = torch.cat([encoded.input_ids, suffix_ids], dim=1)
            text_embeds = embed_layer(ids).to(engine.dtype)
            q_repr = embed_layer(suffix_ids).to(torch.float32)[0].mean(dim=0)

            slots = slot_init(k_positions, width, device, torch.float32)
            text_norm = text_embeds[0].float().norm(dim=-1).mean()
            slot_norm = slots.norm(dim=-1).mean().clamp_min(1e-6)
            s0 = (q_repr.unsqueeze(0) + slots * (text_norm / slot_norm)).to(engine.dtype)

            full_embeds = torch.cat([text_embeds, s0.unsqueeze(0)], dim=1)
            total_len = full_embeds.shape[1]
            hooks.start = total_len - k_positions
            hooks.count = k_positions
            hooks.active = True

            def forward_once():
                cache = build_cache(
                    engine.config, engine.room_for(total_len) + WIDTHS[-1], 1, engine.dtype, engine.device, WIDTHS[-1]
                )
                _sync(device)
                t = time.perf_counter()
                out = engine.backbone(inputs_embeds=full_embeds, use_cache=True, past_key_values=cache)
                hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[0, -1:]
                _sync(device)
                ms = (time.perf_counter() - t) * 1000
                del out, cache
                return hidden, ms

            hidden, base_ms = forward_once()
            result = ScratchResult(options=list(question.options), base=read(hidden), base_ms=base_ms)
            first_in = hooks.input_at_k.clone()
            h_final = hooks.final.clone()
            for n in range(1, r + 1):
                hooks.replace = first_in + alpha * (h_final - first_in)
                hidden, ms = forward_once()
                hooks.replace = None
                key = (layer_k, k_positions, n)
                result.passes[key] = read(hidden)
                result.pass_ms[key] = ms
                result.scratch_final[key] = hooks.final.clone()
                h_final = hooks.final.clone()
    finally:
        hooks.close()
    return result


def scratch_loop_clear_readout(
    engine,
    context: str,
    question: Question,
    layer_k: int,
    k_positions: int,
    r: int,
    alpha: float = 1.0,
) -> list[float]:
    """Stage C: same as `scratch_loop`'s final pass, but the last scratch position (the one the read-out scores) is
    zeroed just before the final norm, on that last pass only. If clearing it does not change the answer, the
    read-out was never actually using the scratchpad -- see RUN-ws1.md stage C.
    """
    decoder = getattr(engine.config, "text_config", engine.config)
    layers, norm = decoder_parts(engine.backbone, decoder.num_hidden_layers)
    planned = plan(question, engine.tokenizer)
    suffix = branch_ids(planned.text, engine.tokenizer)
    device = engine.torch_device
    width = decoder.hidden_size

    def read(hidden_normed) -> list[float]:
        (p,) = engine.heads.apply(
            hidden_normed, [question.options], score(hidden_normed, engine.unembedding, [planned.token_ids], None)
        )
        return p.float().tolist()

    hooks = _ScratchHooks(layers, norm, layer_k)
    clear = {"on": False}

    def _zero_last(module, args, kwargs):
        if not clear["on"]:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        hs = hs.clone()
        hs[0, -1:] = 0.0
        if args:
            return (hs, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": hs}

    extra_handle = norm.register_forward_pre_hook(_zero_last, with_kwargs=True)
    try:
        with engine._lock, torch.inference_mode():
            encoded = encode(context, None, None, engine.processor, engine.tokenizer, engine.device)
            embed_layer = engine.backbone.get_input_embeddings()
            suffix_ids = torch.tensor([suffix], device=engine.device)
            ids = torch.cat([encoded.input_ids, suffix_ids], dim=1)
            text_embeds = embed_layer(ids).to(engine.dtype)
            q_repr = embed_layer(suffix_ids).to(torch.float32)[0].mean(dim=0)
            slots = slot_init(k_positions, width, device, torch.float32)
            text_norm = text_embeds[0].float().norm(dim=-1).mean()
            slot_norm = slots.norm(dim=-1).mean().clamp_min(1e-6)
            s0 = (q_repr.unsqueeze(0) + slots * (text_norm / slot_norm)).to(engine.dtype)
            full_embeds = torch.cat([text_embeds, s0.unsqueeze(0)], dim=1)
            total_len = full_embeds.shape[1]
            hooks.start = total_len - k_positions
            hooks.count = k_positions
            hooks.active = True

            def forward_once():
                cache = build_cache(
                    engine.config, engine.room_for(total_len) + WIDTHS[-1], 1, engine.dtype, engine.device, WIDTHS[-1]
                )
                out = engine.backbone(inputs_embeds=full_embeds, use_cache=True, past_key_values=cache)
                hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[0, -1:]
                del out, cache
                return hidden

            hidden = forward_once()
            first_in = hooks.input_at_k.clone()
            h_final = hooks.final.clone()
            for n in range(1, r + 1):
                hooks.replace = first_in + alpha * (h_final - first_in)
                clear["on"] = n == r  # only the pass whose read-out we report
                hidden = forward_once()
                hooks.replace = None
                h_final = hooks.final.clone()
            clear["on"] = False
            return read(hidden)
    finally:
        hooks.close()
        extra_handle.remove()


# -------------------------------------------------------------------- ws3: placement fix (RUN-ws3.md section 0)
#
# `scratch_loop` (above) appends its `K` scratch positions *after* the whole rendered question, which already
# ends in the literal tail `readout.plan` and `loop.py`'s read-out position are built around: "Answer:" or
# "Answer: ". That put the read-out position -- redefined here to be the scratchpad's own last slot -- one or
# more positions downstream of the place every other read-out in this codebase reads. ws1's stage A measured the
# cost of that move on its own, before any loop: -17 to -27 points at r=0, far past the ~0.16-point path noise
# `loop.py` documents. `RUN-ws3.md` section 0 diagnoses this as a placement confound, not a property of the
# scratchpad or the loop.
#
# The functions below keep the mechanism identical -- same hook class, same slot_init, same per-pass replace --
# and change exactly one thing: the `K` positions are spliced in *before* the tail, not after it. The read-out
# position is therefore still the tail's own last token, exactly where `plan`/`loop.py` already put it; the
# scratchpad becomes extra context the tail causally attends to; the tail is still the last thing in the
# sequence, so `out.last_hidden_state[0, -1:]` reads the same kind of position it always did.


def tail_text_for(planned) -> str:
    """The literal suffix every rendered question ends in -- the thing `loop.py`'s read-out position already is."""
    return "Answer: " if planned.trailing_space else "Answer:"


def split_body_tail(engine, planned) -> tuple[list[int], list[int]]:
    """Split `planned.text`'s own tokenization into `(body_ids, tail_ids)` at the literal tail.

    Tokenizes the full text exactly as `plan` already implies (via `branch_ids`), then slices off the tail
    rather than re-tokenizing the body on its own -- re-tokenizing separately would be wrong if the tokenizer
    merges a token across the body/tail boundary differently than it does inside the one full string. The tail
    is tokenized alone only to know how many trailing ids to slice off, and the slice is then checked against
    that tail tokenization so a merge there (if the tokenizer ever did one) raises instead of silently reading
    at the wrong offset.
    """
    full_ids = branch_ids(planned.text, engine.tokenizer)
    tail_text = tail_text_for(planned)
    tail_ids = engine.tokenizer(tail_text, add_special_tokens=False)["input_ids"]
    if full_ids[-len(tail_ids) :] != tail_ids:
        raise RuntimeError(
            f"tail {tail_text!r} does not tokenize the same at the end of the full rendered text as it does "
            f"alone ({full_ids[-len(tail_ids):]!r} != {tail_ids!r}); splicing scratch positions in front of it "
            f"is not safe for this question without re-deriving the split"
        )
    return full_ids[: -len(tail_ids)], tail_ids


def scratch_loop_tailfixed(
    engine,
    context: str,
    question: Question,
    layer_k: int,
    k_positions: int,
    r: int,
    alpha: float = 1.0,
    filler: bool = False,
) -> ScratchResult:
    """`scratch_loop`, with the placement fixed: `k_positions` scratch slots go *before* the tail
    ("Answer:"/"Answer: "), not after it. The read-out position is unchanged from the no-scratchpad baseline --
    the tail's own last token, which is still the last position in the sequence.

    `filler=True` replaces the scratch slots' real content (the question's mean embedding plus a per-slot
    sinusoidal identifier, `scratch_loop`'s own init) with `k_positions` *identical* copies of the question's
    mean embedding alone -- no per-slot variation, no loopable structure. This is the "same embedding repeated"
    control RUN-ws3.md's section 2 calls for: it isolates "any extra positions in the corrected place help"
    from "position-coded, loopable scratch content helps". A filler run is always called with `r=0` by the
    caller; the loop machinery below still works on it (nothing stops a filler slot from being looped), it is
    just not part of the pre-registered filler comparison.
    """
    decoder = getattr(engine.config, "text_config", engine.config)
    layers, norm = decoder_parts(engine.backbone, decoder.num_hidden_layers)
    layer_types = list(getattr(decoder, "layer_types", []) or [])
    if layer_types and layer_types[layer_k - 1] != "full_attention":
        raise ValueError(f"layer {layer_k} is {layer_types[layer_k - 1]!r}, not full_attention")

    planned = plan(question, engine.tokenizer)
    body_ids, tail_ids = split_body_tail(engine, planned)
    device = engine.torch_device
    width = decoder.hidden_size

    def read(hidden_normed) -> list[float]:
        (p,) = engine.heads.apply(
            hidden_normed, [question.options], score(hidden_normed, engine.unembedding, [planned.token_ids], None)
        )
        return p.float().tolist()

    hooks = _ScratchHooks(layers, norm, layer_k)
    try:
        with engine._lock, torch.inference_mode():
            encoded = encode(context, None, None, engine.processor, engine.tokenizer, engine.device)
            embed_layer = engine.backbone.get_input_embeddings()
            body_t = torch.tensor([body_ids], device=engine.device)
            tail_t = torch.tensor([tail_ids], device=engine.device)
            body_embeds = embed_layer(torch.cat([encoded.input_ids, body_t], dim=1)).to(engine.dtype)
            tail_embeds = embed_layer(tail_t).to(engine.dtype)
            # q_repr spans body+tail together, matching `scratch_loop`'s own definition (the mean of the
            # question's own rendered-text tokens) -- only *where* the scratchpad sits changes, not what seeds it.
            q_repr = embed_layer(torch.cat([body_t, tail_t], dim=1)).to(torch.float32)[0].mean(dim=0)

            if filler:
                s0 = q_repr.unsqueeze(0).expand(k_positions, -1).to(engine.dtype)
            else:
                slots = slot_init(k_positions, width, device, torch.float32)
                text_norm = body_embeds[0].float().norm(dim=-1).mean()
                slot_norm = slots.norm(dim=-1).mean().clamp_min(1e-6)
                s0 = (q_repr.unsqueeze(0) + slots * (text_norm / slot_norm)).to(engine.dtype)

            full_embeds = torch.cat([body_embeds, s0.unsqueeze(0), tail_embeds], dim=1)
            total_len = full_embeds.shape[1]
            hooks.start = body_embeds.shape[1]  # scratch slots sit right after body, right before the tail
            hooks.count = k_positions
            hooks.active = True

            def forward_once():
                cache = build_cache(
                    engine.config, engine.room_for(total_len) + WIDTHS[-1], 1, engine.dtype, engine.device, WIDTHS[-1]
                )
                _sync(device)
                t = time.perf_counter()
                out = engine.backbone(inputs_embeds=full_embeds, use_cache=True, past_key_values=cache)
                # The tail is still the sequence's last thing, so this is still the tail's own last token --
                # exactly the position `plan`/`loop.py` already read, unlike `scratch_loop`'s last scratch slot.
                hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[0, -1:]
                _sync(device)
                ms = (time.perf_counter() - t) * 1000
                del out, cache
                return hidden, ms

            hidden, base_ms = forward_once()
            result = ScratchResult(options=list(question.options), base=read(hidden), base_ms=base_ms)
            if r == 0:
                return result
            first_in = hooks.input_at_k.clone()
            h_final = hooks.final.clone()
            for n in range(1, r + 1):
                hooks.replace = first_in + alpha * (h_final - first_in)
                hidden, ms = forward_once()
                hooks.replace = None
                key = (layer_k, k_positions, n)
                result.passes[key] = read(hidden)
                result.pass_ms[key] = ms
                result.scratch_final[key] = hooks.final.clone()
                h_final = hooks.final.clone()
    finally:
        hooks.close()
    return result
