"""Keep the input-embedding (`embed_tokens`) matrix off the device, on request.

Companion to `readout.load_unembedding`'s `lazy=True` path, which does the same thing for the *output* embedding
(the matrix a read-out scores against). This module is the input side: the table every token id is looked up in
before the first decoder layer ever runs. The two matrices are measured and freed independently -- a checkpoint
that ties them (one safetensors key, read twice: once by `AutoModel.from_pretrained` into `embed_tokens.weight`,
once by `load_unembedding` into `engine.unembedding`) still holds two separate device copies today, because nothing
currently shares them; `PRISMYRA_LM_HEAD=lazy` and `PRISMYRA_EMBED_TOKENS=lazy` can be set independently and each
frees its own 0.947 GiB on a 36-layer checkpoint's vocabulary (248,320 rows x 2,048 columns, bf16).

A lookup is a copy, not a computation: which device holds the source rows before the copy cannot change the bytes
copied. So moving the resident matrix to host, pinned memory and gathering only the rows a call actually names is
bit-identical to leaving the whole matrix resident -- the same argument `logits_for` already relies on for the
output embedding (`readout.py`'s module docstring on `logits_for`).
"""

from __future__ import annotations

import types

import torch
from torch import nn


def _lazy_forward(self: nn.Embedding, input_ids: torch.Tensor) -> torch.Tensor:
    """Replacement for `nn.Embedding.forward` once `self.weight` has been moved to host, pinned memory.

    Brings `input_ids` to the weight's device (host) to index it -- fancy indexing requires both operands on the
    same device -- then moves only the gathered rows back to wherever `input_ids` came from. For a context of
    `n` tokens this transfers `n` rows instead of the whole vocabulary; the vocabulary itself never crosses the
    bus more than once (into pinned memory, at `make_lazy` time).
    """
    device = input_ids.device
    rows = self.weight[input_ids.to(self.weight.device)]
    return rows.to(device) if rows.device != device else rows


def make_lazy(module: nn.Embedding) -> None:
    """Move `module.weight` to host, pinned memory and bind `_lazy_forward` in its place.

    Call once, right after the backbone finishes loading (before any context is read) -- `module.weight` is
    expected to be resident on the device at call time, matching how `AutoModel.from_pretrained(..., device_map=...)`
    leaves it. Pinning (rather than a plain `.cpu()`) is what makes the per-call host-to-device copy of the
    gathered rows async-capable; without it every `_lazy_forward` call pays a staging copy first.
    """
    module.weight.data = module.weight.data.to("cpu").pin_memory()
    module.forward = types.MethodType(_lazy_forward, module)
