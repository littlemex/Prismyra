"""A cache holding one context once, and one short tail per branch.

The design rests on one asymmetry:

* the keys and values the context produced are only **read** by the branches, so one copy can serve them all;
* the recurrence and convolution state, and each branch's own tokens, are **written** by each branch, so they cannot
  share storage.

`ForkLayer` stores each of those where it belongs. The context sits in one row and every branch reads it; each
branch's own tokens sit in its own row of a much shorter buffer. A read hands the attention kernel the two joined
together, which costs a copy that lives for one layer and is freed before the next -- 222 MiB at 3,040 tokens and a
group of 32, against 2.17 GiB that used to be held for as long as the context was open.

**Why a transient copy rather than a page table.** The kernel can be handed a page table naming shared pages, which
would remove the copy as well, and that path is measured and available: `tests/test_gpu_paging.py` confirms many rows
may read one set of pages. It is not what this file does, because joining the two buffers is **bit-identical to
replicating them** -- the same bytes in the same order -- while a paged read reduces in a different order and would
change answers by a rounding step in a 40-layer model, where one changed routing logit picks a different expert. So
the storage is fixed first, against an unchanged baseline, and the copy is removed second.

What it costs, measured on the supported model at 3,040 context tokens and a group of 32: an open context held 2.17
GiB and now holds 0.37 GiB, a factor of 5.9. At 20,000 tokens the factor is 18, because the shared part stops being
copied and the per-branch part does not grow with the context.

Neither of the framework's two cache families fits: the dynamic one grows by concatenation, so every request allocates
new tensors; the static one insists the batch it was allocated for is the batch every write arrives at, and here the
context arrives as one row and the branches as many. See docs/FORK.md.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import CacheLayerMixin


class ForkLayer(CacheLayerMixin):
    """One row of context, `rows` rows of branch tail, joined on read.

    Deliberately not a subclass of the framework's static layer. A static layer's contract is "the batch is fixed and
    every write matches it"; this one's is "one row means the context, all of them means the branches".

    No `_layer_type` is declared. `CacheLayerMixin.__init_subclass__` registers any subclass that sets one into the
    framework's global layer-type mapping, which would replace the default layer for every model in the process.
    """

    is_sliding = False

    #: Read by `fork.snapshot` and `fork.restore_and_fork` to tell this layer from a recurrent one. A class attribute
    #: rather than a duck-typed check on `max_cache_len`, because those two functions used to decide by that and would
    #: silently start cloning a paged layer's buffers if the attribute ever moved.
    holds_attention = True

    def __init__(self, max_cache_len: int, max_batch_size: int = 1, max_branch_len: int = 512, **_: object):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size
        #: How long a branch's own run of tokens may be. The context takes the rest of `max_cache_len`.
        self.max_branch_len = max_branch_len
        # The same number twice, for two consumers. The tensor is mutated in place rather than replaced, so anything
        # holding a reference to it keeps seeing the current value. The integer exists because the framework asks for
        # the length on the *host* during the forward pass, and reading it off the tensor there is a device-to-host
        # copy on the request path. The tensor is kept because the framework's own code expects to find one under this
        # name.
        self.cumulative_length = torch.tensor(0, dtype=torch.long)
        self._host_length = 0
        #: Where the context ended, which is where each branch's own tokens begin. Set by the context write, and the
        #: thing a branch write measures its own offset from.
        self.context_length = 0
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        #: One row per branch, `max_branch_len` long. Short, so replicating it is cheap in a way the context is not.
        self.branch_keys: torch.Tensor | None = None
        self.branch_values: torch.Tensor | None = None
        self.is_initialized = False

    # ---------------------------------------------------------------- allocation
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Allocate on the first write, which is the context and arrives as one row."""
        _, heads, _, head_dim = key_states.shape
        self._allocate(heads, head_dim, key_states.dtype, key_states.device)

    def early_initialization(self, batch_size: int, num_heads: int, head_dim: int, dtype, device) -> None:
        """Allocate before the first forward, for a caller that needs the buffers to exist before any work runs."""
        self.max_batch_size = max(self.max_batch_size, batch_size)
        self._allocate(num_heads, head_dim, dtype, device)

    def _allocate(self, heads: int, head_dim: int, dtype, device) -> None:
        self.device, self.dtype = device, dtype
        context_room = max(1, self.max_cache_len - self.max_branch_len)
        # One row for the context. This is the whole change: it used to be `max_batch_size` rows of the same bytes.
        self.keys = torch.zeros((1, heads, context_room, head_dim), dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)
        self.branch_keys = torch.zeros(
            (self.max_batch_size, heads, self.max_branch_len, head_dim), dtype=dtype, device=device
        )
        self.branch_values = torch.zeros_like(self.branch_keys)
        self.cumulative_length = self.cumulative_length.to(device)
        self.is_initialized = True

    # ---------------------------------------------------------------- the one difference
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *_, **__):
        """A one-row write is the context; a full-batch write is the branches.

        Returns contiguous keys and values either way, which is not an incidental convenience. The model's own
        attention is the fallback when the borrowed kernel is unavailable, and it takes tensors rather than a page
        table -- so a layer that returned anything else would not fall back, it would fail on the first cached
        forward.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        assert self.keys is not None and self.branch_keys is not None

        rows = key_states.shape[0]
        count = key_states.shape[-2]

        if rows == 1 and self.max_batch_size > 1:
            return self._write_context(key_states, value_states, count)
        if rows == self.max_batch_size:
            return self._write_branches(key_states, value_states, count)
        raise ValueError(
            f"this layer holds {self.max_batch_size} rows and was given {rows}; a write is either one row, "
            f"meaning the shared context, or all of them, meaning the branches"
        )

    def _write_context(self, key_states: torch.Tensor, value_states: torch.Tensor, count: int):
        """Into the single shared row, and the branches read it from there rather than being given a copy."""
        assert self.keys is not None and self.values is not None
        start = self._host_length
        if start + count > self.keys.shape[-2]:
            raise ValueError(
                f"a context of {start + count} tokens does not fit: this layer holds {self.keys.shape[-2]} context "
                f"tokens beside {self.max_branch_len} for each branch"
            )
        self.keys[:, :, start : start + count] = key_states
        self.values[:, :, start : start + count] = value_states
        self._advance(count)
        self.context_length = self._host_length
        return self.keys[:, :, : self._host_length], self.values[:, :, : self._host_length]

    def _write_branches(self, key_states: torch.Tensor, value_states: torch.Tensor, count: int):
        """Into each branch's own row, then joined with the context for the read.

        The offset is measured from the context's length, not from the position the tokens carry. Those two stop being
        equal as soon as an image is in the context -- the model's three-axis positions advance by a grid rather than
        by a token -- and using a position here would write past what was ever filled.
        """
        assert self.keys is not None and self.values is not None
        assert self.branch_keys is not None and self.branch_values is not None
        at = self._host_length - self.context_length
        if at + count > self.max_branch_len:
            raise ValueError(
                f"a branch of {at + count} tokens does not fit in {self.max_branch_len}; the widest branch is what "
                f"this buffer was sized for"
            )
        self.branch_keys[:, :, at : at + count] = key_states
        self.branch_values[:, :, at : at + count] = value_states
        self._advance(count)

        used = self._host_length - self.context_length
        rows = self.max_batch_size
        # Joined for this layer's read and dropped before the next layer runs. `expand` costs nothing; the copy is the
        # `cat`, and it is the price of keeping this bit-identical to holding the rows separately.
        keys = torch.cat(
            (self.keys[:, :, : self.context_length].expand(rows, -1, -1, -1), self.branch_keys[:, :, :used]), dim=-2
        )
        values = torch.cat(
            (self.values[:, :, : self.context_length].expand(rows, -1, -1, -1), self.branch_values[:, :, :used]),
            dim=-2,
        )
        return keys, values

    def _advance(self, count: int) -> None:
        self._host_length += count
        self.cumulative_length.add_(count)

    # ---------------------------------------------------------------- the rest of the contract
    def get_seq_length(self) -> int:
        """Tokens held, which is not a position. Nothing positional may be derived from this: with media in the context
        the model's own positions run ahead of the token count, and the two must not be confused."""
        return self._host_length if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def get_max_length(self) -> int:
        return self.max_cache_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self._host_length + query_length, 0

    def reset(self) -> None:
        """Rewind without reallocating; reallocating would move the addresses this class exists to keep."""
        if self.is_initialized:
            assert self.keys is not None and self.branch_keys is not None
            self.keys.zero_()
            self.values.zero_()
            self.branch_keys.zero_()
            self.branch_values.zero_()
            self.cumulative_length.zero_()
            self._host_length = 0
            self.context_length = 0

    def rewind_to(self, length: int) -> None:
        """Set the length back to a point already written, leaving the contents.

        This is what makes a second group of branches possible: the first group advanced the length by its own tokens,
        and the next group has to start from the end of the context again. Zeroing would throw the context away.

        Only a rewind to the context's own end is meaningful, because a branch's tokens live in a separate buffer that
        the next group overwrites from its start. Rewinding into the context would leave that buffer's contents
        describing tokens the length no longer claims.
        """
        if not self.is_initialized:
            return
        if self.context_length and length != self.context_length:
            raise ValueError(
                f"this layer can only rewind to the end of its context ({self.context_length} tokens), not to {length}"
            )
        self.cumulative_length.fill_(length)
        self._host_length = int(length)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the branches. The context is one row shared by all of them and is not reordered -- there is nothing
        to permute."""
        if self.is_initialized:
            assert self.branch_keys is not None and self.branch_values is not None
            index = beam_idx.to(self.branch_keys.device)
            self.branch_keys = self.branch_keys.index_select(0, index)
            self.branch_values = self.branch_values.index_select(0, index)

    def crop(self, tokens_to_remove: int) -> None:
        """Refused. Cropping into the context would leave each branch's buffer describing tokens the length no longer
        claims, and cropping a branch is what `rewind_to` is for. A plausible wrong answer is worse than an error."""
        raise NotImplementedError(
            "this cache cannot be cropped: the context and the branches are stored apart, so there is no single run of "
            "tokens to take from the end. Use rewind_to(context_length) to start another group."
        )

    def offload(self) -> None:
        """Not supported: moving these buffers to the host and back would change their addresses."""

    def prefetch(self) -> None:
        pass


def cache_bytes(config, max_cache_len: int, rows: int, dtype: torch.dtype, max_branch_len: int = 512) -> int:
    """How much device memory one cache of this shape needs.

    Worth a function rather than a comment because the number decides how many contexts fit on a card, and it is not
    the obvious product. The context is held **once** and each branch holds only its own short run of tokens, so the
    context length is multiplied by one and the group by `max_branch_len`:

        2 x attention_layers x kv_heads x head_dim x (context + group x branch_len) x bytes

    At 3,040 tokens and a group of 32 on the supported model that is 0.37 GiB, where replicating the context across
    the group made it 2.17 GiB. The recurrent layers are left out: their state is a fixed size per layer whatever the
    context length, so they do not grow with it -- but they are replicated per branch and are the floor this cannot go
    below.

    What this does not count is the transient join: one layer's context and branch rows concatenated for the read, 222
    MiB at those sizes, allocated and freed inside a layer rather than held.
    """
    decoder = getattr(config, "text_config", config)
    # Read off the config rather than through the framework's helper: this function needs only the list of layer
    # kinds, and the helper also returns constructor arguments it does not want, exists under different names across
    # releases, and cannot be called without a framework new enough to have it.
    layer_types = list(getattr(decoder, "layer_types", []) or [])
    if not layer_types:
        layer_types = ["full_attention"] * int(getattr(decoder, "num_hidden_layers", 0))
    attention_layers = sum(1 for kind in layer_types if kind == "full_attention")
    heads = decoder.num_key_value_heads
    head_dim = getattr(decoder, "head_dim", None) or decoder.hidden_size // decoder.num_attention_heads
    per_element = torch.empty((), dtype=dtype).element_size()

    context_room = max(1, max_cache_len - max_branch_len)
    slots = context_room + rows * max_branch_len
    return 2 * attention_layers * heads * head_dim * slots * per_element


def _layer_types(decoder):
    from transformers.cache_utils import get_layer_types_and_kwargs

    return get_layer_types_and_kwargs(decoder)


def build_cache(config, max_cache_len: int, rows: int, dtype: torch.dtype, device: str, max_branch_len: int = 512):
    """A cache for one backbone: `ForkLayer` where attention needs keys and values, the framework's own layer elsewhere.

    The recurrent layers need no replacement. Their state is a fixed size per layer whatever the context length, so
    they are already preallocated in everything but name.
    """
    from transformers.cache_utils import STATIC_LAYER_TYPE_MAPPING, Cache

    decoder = getattr(config, "text_config", config)
    layer_types, kwargs = _layer_types(decoder)
    heads = decoder.num_key_value_heads
    head_dim = getattr(decoder, "head_dim", None) or decoder.hidden_size // decoder.num_attention_heads

    layers: list = []
    for kind in layer_types:
        if kind == "full_attention":
            layer = ForkLayer(max_cache_len=max_cache_len, max_batch_size=rows, max_branch_len=max_branch_len)
            layer.early_initialization(rows, heads, head_dim, dtype, device)
        else:
            cls = STATIC_LAYER_TYPE_MAPPING[kind]
            layer = cls(**{k: v for k, v in kwargs.items() if k != "max_cache_len"})
        layers.append(layer)
    return Cache(layers=layers)
