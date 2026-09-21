"""A cache for one shared context and many short-lived branches.

The design rests on one asymmetry:

* the keys and values the context produced are only **read** by the branches, so one write can serve them all;
* the recurrence and convolution state, and each branch's own tokens, are **written** by each branch, so they cannot
  share storage.

Being read-only is what makes the fan-out *correct*, not what makes it free. `ForkLayer` is preallocated at the full
batch and a one-row context write is copied into every row, so the context's keys and values are physically replicated
`group` times. That is the memory cost `cache_bytes` reports: about 3.4 GiB at 5,000 tokens at the default group of 32,
which this model can afford because it has only two key-value heads. Storing the context once and giving each branch
only its own tail is the obvious improvement, and is not done here.

Neither of the framework's two cache families fits: the dynamic one grows by concatenation, so every request allocates
new tensors; the static one insists the batch it was allocated for is the batch every write arrives at, and here the
context arrives as one row and the branches as many. See docs/FORK.md.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import CacheLayerMixin


class ForkLayer(CacheLayerMixin):
    """Preallocated keys and values: a one-row write lands in every row, a full-batch write lands per row.

    Deliberately not a subclass of the framework's static layer. A static layer's contract is "the batch is fixed and
    every write matches it"; this one's is "the batch is fixed and a single row means all of them".

    No `_layer_type` is declared. `CacheLayerMixin.__init_subclass__` registers any subclass that sets one into the
    framework's global layer-type mapping, which would replace the default layer for every model in the process.
    """

    is_sliding = False

    def __init__(self, max_cache_len: int, max_batch_size: int = 1, **_: object):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size
        # The same number twice, for two consumers. The tensor is mutated in place rather than replaced, so anything
        # holding a reference to it keeps seeing the current value.
        # The integer exists because the framework asks for the length on the *host* during the forward pass, and
        # reading it off the tensor there is a device-to-host copy on the request path. The tensor is kept because the
        # framework's own code expects to find one under this name.
        self.cumulative_length = torch.tensor(0, dtype=torch.long)
        self._host_length = 0
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False

    # ---------------------------------------------------------------- allocation
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Allocate at the full batch, whatever batch the first write arrives at.

        The first write is the context, which is one row. Allocating to that would leave nowhere for the branches.
        """
        b, h, _, d = key_states.shape
        rows = max(self.max_batch_size, b)
        self.max_batch_size = rows
        self.device, self.dtype = key_states.device, key_states.dtype
        self.keys = torch.zeros((rows, h, self.max_cache_len, d), dtype=self.dtype, device=self.device)
        self.values = torch.zeros_like(self.keys)
        self.cumulative_length = self.cumulative_length.to(self.device)
        self.is_initialized = True

    def early_initialization(self, batch_size: int, num_heads: int, head_dim: int, dtype, device) -> None:
        """Allocate before the first forward, for a caller that needs the buffers to exist before any work runs."""
        self.max_batch_size = max(self.max_batch_size, batch_size)
        self.device, self.dtype = device, dtype
        self.keys = torch.zeros(
            (self.max_batch_size, num_heads, self.max_cache_len, head_dim), dtype=dtype, device=device
        )
        self.values = torch.zeros_like(self.keys)
        self.cumulative_length = self.cumulative_length.to(device)
        self.is_initialized = True

    # ---------------------------------------------------------------- the one difference
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *_, **__):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        n = key_states.shape[-2]
        start = self._host_length
        pos = torch.arange(n, device=self.keys.device) + start
        self._host_length = start + n
        self.cumulative_length.add_(n)

        rows = key_states.shape[0]
        if rows == 1 and self.max_batch_size > 1:
            # The context. Every branch reads the same keys, so writing them to every row *is* the fan-out: each row
            # then owns a copy, which is what lets rows be written independently afterwards without aliasing.
            self.keys.index_copy_(2, pos, key_states.expand(self.max_batch_size, -1, -1, -1))
            self.values.index_copy_(2, pos, value_states.expand(self.max_batch_size, -1, -1, -1))
        elif rows == self.max_batch_size:
            self.keys.index_copy_(2, pos, key_states)
            self.values.index_copy_(2, pos, value_states)
        else:
            raise ValueError(
                f"this layer holds {self.max_batch_size} rows and was given {rows}; a write is either one row, "
                f"meaning the shared context, or all of them, meaning the branches"
            )

        end = start + n
        # Written broadly, returned narrowly. Returning every row to a one-row context write would hand the model an
        # attention output with too many rows, which fails downstream as a shape mismatch inside a projection.
        return self.keys[:rows, :, :end], self.values[:rows, :, :end]

    # ---------------------------------------------------------------- the rest of the contract
    def get_seq_length(self) -> int:
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
            self.keys.zero_()
            self.values.zero_()
            self.cumulative_length.zero_()
            self._host_length = 0

    def rewind_to(self, length: int) -> None:
        """Set the length back to a point already written, leaving the contents.

        This is what makes a second group of branches possible: the first group advanced the length by its own tokens,
        and the next group has to start from the end of the context again. Zeroing would throw the context away.
        """
        if self.is_initialized:
            self.cumulative_length.fill_(length)
            self._host_length = int(length)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.is_initialized:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    def crop(self, tokens_to_remove: int) -> None:
        if self.is_initialized:
            removed = min(tokens_to_remove, self._host_length)
            self.cumulative_length.sub_(removed)
            self._host_length -= removed

    def offload(self) -> None:
        """Not supported: moving these buffers to the host and back would change their addresses."""

    def prefetch(self) -> None:
        pass


def cache_bytes(config, max_cache_len: int, rows: int, dtype: torch.dtype) -> int:
    """How much device memory one cache of this shape needs.

    Worth a function rather than a comment, because the number is large and surprising. Every branch gets its own copy
    of the context's keys and values -- that fan-out is what lets rows be written independently -- so the cost is the
    context's key-value cache multiplied by the group. At 5,000 tokens and the default group of 32 that is about 3.4 GiB
    on the supported model, which has two key-value heads; one with sixteen would pay eight times that. The recurrent
    layers are left out: their state is a fixed size per layer whatever the context length, so they do not grow with it.
    """
    decoder = getattr(config, "text_config", config)
    layer_types, _ = _layer_types(decoder)
    attention_layers = sum(1 for kind in layer_types if kind == "full_attention")
    heads = decoder.num_key_value_heads
    head_dim = getattr(decoder, "head_dim", None) or decoder.hidden_size // decoder.num_attention_heads
    per_element = torch.empty((), dtype=dtype).element_size()
    return 2 * attention_layers * rows * heads * max_cache_len * head_dim * per_element


def _layer_types(decoder):
    from transformers.cache_utils import get_layer_types_and_kwargs

    return get_layer_types_and_kwargs(decoder)


def build_cache(config, max_cache_len: int, rows: int, dtype: torch.dtype, device: str):
    """A cache for one backbone: `ForkLayer` where attention needs keys and values, the framework's own layer elsewhere.

    The recurrent layers need no replacement. Their state is a fixed size per layer whatever the context length, so they
    are already preallocated in everything but name.
    """
    from transformers.cache_utils import STATIC_LAYER_TYPE_MAPPING, Cache

    decoder = getattr(config, "text_config", config)
    layer_types, kwargs = _layer_types(decoder)
    heads = decoder.num_key_value_heads
    head_dim = getattr(decoder, "head_dim", None) or decoder.hidden_size // decoder.num_attention_heads

    layers: list = []
    for kind in layer_types:
        if kind == "full_attention":
            layer = ForkLayer(max_cache_len=max_cache_len, max_batch_size=rows)
            layer.early_initialization(rows, heads, head_dim, dtype, device)
        else:
            cls = STATIC_LAYER_TYPE_MAPPING[kind]
            layer = cls(**{k: v for k, v in kwargs.items() if k != "max_cache_len"})
        layers.append(layer)
    return Cache(layers=layers)
