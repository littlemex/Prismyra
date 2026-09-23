"""The same storage as `cache.ForkLayer`, read through a page table so that the read's shape stops depending on the
context's length.

This existed once, was never wired to anything, and was deleted. The reason it is back is not the one it was written
for. As a memory saving it was measured and was worth **nothing**: a branch pass's peak is set elsewhere, and removing
the join changed it by 0.000 GiB. What it actually buys is a **constant shape**, and constant shapes are what a recorded
CUDA graph needs.

The arithmetic. `ForkLayer` joins the context and a branch's own tokens into one tensor for each layer's read, so the
read is `(rows, context + suffix, heads, dim)` and every context length is a different shape -- which means a different
recording. A recorded pass replays in 32.6 ms against 113.5 eagerly, so one recording per open context gives that to a
session asking many groups about one document and nothing to a request asking one group about a document it will not
revisit. A page pool is allocated once for the longest context the cache was built for; the context's length lives in
`seqused_k` as **data**. One recording then serves every context length.

Three things this file has to get right for that, and each of them was wrong somewhere in the first attempt or in the
path it replaces:

* **it has to actually run.** The first version validated its flag, reported `"storage": "paged"` from `stats()` and
  carried eighteen unit tests -- and `_read` built the cache with six positional arguments where the flag was the
  seventh, so the layer was never constructed. Every measurement taken from it was the joined path compared
  with itself, and the giveaway was that a path which reduces in a different order moved no probability at all.
  `PagedForkLayer.pages_read` counts the reads it actually served, and the engine reports it.
* **nothing may be read to the host.** `max_seqlen_k` was taken with `seqused[0].item()`, a device-to-host copy that
  is uncapturable, and being capturable is the whole point of this file. It is the pool's capacity instead: an upper
  bound is what that argument is for, and an upper bound is constant.
* **the table and the lengths are static tensors.** Filled in place when a context is finished and never replaced,
  because a recording bakes in addresses.

**Sharing ends on a page boundary.** The kernel reads a row's pages as one run of `seqused_k` tokens, so a page that is
half context and half nothing cannot sit in the middle of that run. A context of 3,040 tokens fills 190 pages of 16
exactly; one of 3,045 fills 190 and leaves 5 over, and those 5 are copied into each branch's first private page. That
copy is bounded by the page size rather than by the context length, which is the whole difference from the join it
replaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import CacheLayerMixin

#: Tokens per page. Must be a multiple of 16: the borrowed attention kernel requires it of a paged key-value cache, so a
#: smaller page to shrink the remainder is not available.
BLOCK = 16


class PagedUnavailable(RuntimeError):
    """The installed attention kernel cannot be handed a page table. Raised when the paged path is asked for, at
    construction rather than on the first cached forward, because a `TypeError` from inside a kernel ten layers deep is
    not a diagnosis."""


def kernel_supports_pages() -> tuple[bool, str]:
    """Whether the installed kernel takes the arguments this file needs, and why not if it does not.

    Checked by signature rather than by trying it, so the answer is available before a model is loaded. The range this
    package declares for vLLM is wide and the arguments are not in every build of it.
    """
    try:
        import inspect

        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError as e:
        return False, f"vllm.vllm_flash_attn is not importable: {e}"

    try:
        parameters = inspect.signature(flash_attn_varlen_func).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C entry point with no introspectable signature
        return True, "the kernel's signature cannot be read, so the arguments are assumed present"

    missing = [name for name in ("block_table", "seqused_k") if name not in parameters]
    if missing:
        return False, f"the installed flash_attn_varlen_func does not take {', '.join(missing)}"
    return True, "the kernel takes block_table and seqused_k"


@dataclass(frozen=True)
class Layout:
    """How a context and its branches divide into pages. Arithmetic only, so it can be checked without a device.

    Every length here counts **tokens held**, never a position. With media in a context the model's own positions run
    ahead of the token count, and a page index derived from a position would name pages that were never written.
    """

    context_tokens: int
    rows: int
    branch_tokens: int
    block: int = BLOCK

    @property
    def shared_pages(self) -> int:
        """Whole pages of context, which every branch's table names. A partial last page cannot be among them."""
        return self.context_tokens // self.block

    @property
    def remainder(self) -> int:
        """Context tokens left over after the whole pages, copied into each branch rather than shared."""
        return self.context_tokens % self.block

    @property
    def private_pages(self) -> int:
        """Pages each branch owns: its copy of the remainder, then room for its own tokens."""
        return -(-(self.remainder + self.branch_tokens) // self.block)

    @property
    def total_pages(self) -> int:
        return self.shared_pages + self.rows * self.private_pages

    def private_range(self, row: int) -> tuple[int, int]:
        first = self.shared_pages + row * self.private_pages
        return first, first + self.private_pages

    def table(self) -> list[list[int]]:
        """One row per branch: the shared pages, then that row's own.

        The same page index appears in every row, which is the entire mechanism: the kernel reads one copy however
        many rows name it.
        """
        shared = list(range(self.shared_pages))
        return [shared + list(range(*self.private_range(row))) for row in range(self.rows)]

    def bytes_for(self, heads: int, head_dim: int, element_size: int) -> int:
        return 2 * self.total_pages * self.block * heads * head_dim * element_size


class PagedForkLayer(CacheLayerMixin):
    """Keys and values in pages: the context's read by every branch, each branch's own written only by it."""

    is_sliding = False
    holds_attention = True
    #: Read by the attention replacement to decide whether to ask for a page table. A layer attribute rather than a
    #: global setting, because a model can mix layer kinds and only these hold keys and values.
    paged = True

    #: Branch reads served by every layer of this class in this process. See `paged_read`.
    reads_served = 0

    def __init__(self, max_cache_len: int, max_batch_size: int = 1, max_branch_len: int = 512, **_: object):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size
        self.max_branch_len = max_branch_len
        self.cumulative_length = torch.tensor(0, dtype=torch.long)
        self._host_length = 0
        self.context_length = 0
        #: Told rather than guessed, because at a group of one the context and a branch are both one row.
        self.writing_branches = False
        #: How many branch reads this layer has actually served. A witness, not state: the first version of this file
        #: reported itself as installed and never ran, so a claim that the paged path is in use is checked against the
        #: code that would have done the work.
        self.pages_read = 0
        self.last_branch_rows = 0
        self.layout: Layout | None = None
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        #: Static, filled in place when a context is finished. A recording bakes in addresses, so these are never
        #: replaced once allocated -- which is also why they are sized for the widest group and longest context.
        self.block_table: torch.Tensor | None = None
        self.seqused: torch.Tensor | None = None
        self.is_initialized = False

    # ---------------------------------------------------------------- allocation
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        _, heads, _, head_dim = key_states.shape
        self._allocate(heads, head_dim, key_states.dtype, key_states.device)

    def early_initialization(self, batch_size: int, num_heads: int, head_dim: int, dtype, device) -> None:
        self.max_batch_size = max(self.max_batch_size, batch_size)
        self._allocate(num_heads, head_dim, dtype, device)

    def _allocate(self, heads: int, head_dim: int, dtype, device) -> None:
        """Sized for the longest context this cache was built for, because the pages are fixed rather than pooled.

        A pool shared between contexts would fit more of them, and is deliberately not here: it needs admission by free
        pages, rollback when an allocation fails part way through ten layers, and a rule for reusing a page while
        earlier
        work may still be reading it. Those are their own commit.
        """
        room = max(1, self.max_cache_len - self.max_branch_len)
        self.layout = Layout(
            context_tokens=room, rows=self.max_batch_size, branch_tokens=self.max_branch_len, block=BLOCK
        )
        self.device, self.dtype = device, dtype
        self.keys = torch.zeros((self.layout.total_pages, BLOCK, heads, head_dim), dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)
        # Allocated at the widest layout so a recording's addresses stay valid whatever context is opened next. The
        # table's own width shrinks with the context; the tensor's does not, and the unused columns are never named
        # because `seqused` says how far along each row the kernel should read.
        pages = self.layout.shared_pages + self.layout.private_pages
        self.block_table = torch.zeros((self.max_batch_size, pages), dtype=torch.int32, device=device)
        self.seqused = torch.zeros((self.max_batch_size,), dtype=torch.int32, device=device)
        self.cumulative_length = self.cumulative_length.to(device)
        self.is_initialized = True

    # ---------------------------------------------------------------- writing
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *_, **__):
        """A one-row write is the context; a full-batch write is the branches.

        The context write returns the tensors it was given. That is what the prefill's own attention reads, and it is
        correct only because the context arrives in one piece: a second context write would need the earlier tokens
        gathered back out of pages, so it is refused rather than answered with the wrong half.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        assert self.keys is not None and self.values is not None and self.layout is not None

        rows = key_states.shape[0]
        count = key_states.shape[-2]

        if not self.writing_branches:
            if rows != 1:
                raise ValueError(f"a context arrives as one row and this one arrived as {rows}")
            if self._host_length:
                raise NotImplementedError(
                    "this cache takes its context in one piece; a second context write would have to gather the first "
                    "back out of its pages"
                )
            self._write_context(key_states, value_states)
            self._advance(count)
            return key_states, value_states

        if rows > self.max_batch_size:
            raise ValueError(f"this layer holds {self.max_batch_size} branch rows and was given {rows}")
        self._write_branches(key_states, value_states, rows)
        self.last_branch_rows = rows
        self._advance(count)
        # Nothing contiguous to return: the attention replacement reads `paged_read` instead. Returning the inputs would
        # be a plausible-looking lie, since they are only this group's tokens and not the context before them.
        return None, None

    def _write_context(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Into the shared pages, in page order. One row, so there is one copy."""
        assert self.keys is not None and self.values is not None
        count = key_states.shape[-2]
        # (1, heads, tokens, dim) -> (tokens, heads, dim), which is the page layout's own order.
        keys = key_states[0].transpose(0, 1)
        values = value_states[0].transpose(0, 1)
        whole = count // BLOCK
        if whole:
            self.keys[:whole] = keys[: whole * BLOCK].view(whole, BLOCK, *keys.shape[1:])
            self.values[:whole] = values[: whole * BLOCK].view(whole, BLOCK, *values.shape[1:])
        left = count - whole * BLOCK
        if left:
            self.keys[whole, :left] = keys[whole * BLOCK :]
            self.values[whole, :left] = values[whole * BLOCK :]

    def begin_branches(self) -> None:
        """The context is finished. The table and the lengths are filled here, because both need to know how long the
        context turned out to be, and filled **in place** because a recording holds their addresses."""
        if self.writing_branches:
            return
        self.context_length = self._host_length
        self.writing_branches = True
        assert self.keys is not None and self.block_table is not None and self.seqused is not None
        layout = Layout(
            context_tokens=self.context_length, rows=self.max_batch_size, branch_tokens=self.max_branch_len, block=BLOCK
        )
        self.layout = layout
        table = torch.tensor(layout.table(), dtype=torch.int32, device=self.keys.device)
        self.block_table.zero_()
        self.block_table[:, : table.shape[1]] = table
        self._copy_remainder()
        self._set_lengths()

    def _copy_remainder(self) -> None:
        assert self.keys is not None and self.values is not None and self.layout is not None
        layout = self.layout
        if not layout.remainder:
            return
        source = layout.shared_pages
        for row in range(layout.rows):
            first, _ = layout.private_range(row)
            self.keys[first, : layout.remainder] = self.keys[source, : layout.remainder]
            self.values[first, : layout.remainder] = self.values[source, : layout.remainder]

    def _set_lengths(self) -> None:
        """How many tokens each row holds, as device data rather than as a shape. This is the mechanism: the context's
        length moves a number in this tensor and moves nothing about the read's shape."""
        assert self.seqused is not None and self.layout is not None
        layout = self.layout
        used = layout.shared_pages * BLOCK + layout.remainder + (self._host_length - self.context_length)
        self.seqused.fill_(used)

    def _write_branches(self, key_states: torch.Tensor, value_states: torch.Tensor, rows: int) -> None:
        """Into each row's own pages, at an offset measured from the context's length and never from a position."""
        assert self.keys is not None and self.values is not None and self.layout is not None
        layout = self.layout
        count = key_states.shape[-2]
        at = layout.remainder + (self._host_length - self.context_length)
        if at + count > layout.private_pages * BLOCK:
            raise ValueError(
                f"a branch reached token {at + count} of {layout.private_pages * BLOCK} private slots; the widest "
                f"branch is what these pages were sized for"
            )
        keys = key_states.transpose(1, 2)  # (rows, tokens, heads, dim)
        values = value_states.transpose(1, 2)
        for row in range(rows):
            first, _ = layout.private_range(row)
            written = 0
            while written < count:
                page = first + (at + written) // BLOCK
                offset = (at + written) % BLOCK
                take = min(BLOCK - offset, count - written)
                self.keys[page, offset : offset + take] = keys[row, written : written + take]
                self.values[page, offset : offset + take] = values[row, written : written + take]
                written += take

    def _advance(self, count: int) -> None:
        self._host_length += count
        self.cumulative_length.add_(count)
        if self.writing_branches:
            self._set_lengths()

    # ---------------------------------------------------------------- reading
    def paged_read(self, rows: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """The pool, the table, how many tokens each row holds, and a constant upper bound on that.

        `rows` narrows the table and the lengths to the group being answered, because the kernel requires the table to
        have exactly as many rows as there are query sequences, and a group uses only as many rows as it has
        questions.
        Slices rather than copies, so the addresses a recording holds are still the ones it wrote down.

        The bound is the pool's capacity rather than the actual length. `max_seqlen_k` exists so the kernel can size its
        launch, an upper bound satisfies it, and reading the real length would mean `seqused[0].item()` -- a
        device-to-host copy, which is both a stall on the request path and uncapturable. The first version of this file
        did exactly that.
        """
        assert self.keys is not None and self.values is not None and self.layout is not None
        assert self.block_table is not None and self.seqused is not None
        assert self.writing_branches, "the context must be finished before a branch can read it"
        # Counted on the class as well as on the instance. The thing this has to catch is "the flag says paged and
        # nothing ran", and an engine cannot ask a context's layers after that context is closed -- so the total has to
        # outlive them. A diagnostic rather than state, and the first version of this file is why it exists at all.
        PagedForkLayer.reads_served += 1
        self.pages_read += 1
        take = self.max_batch_size if rows is None else rows
        if take > self.max_batch_size:
            raise ValueError(f"this layer holds {self.max_batch_size} rows and a read asked for {take}")
        return (
            self.keys,
            self.values,
            self.block_table[:take],
            self.seqused[:take],
            self.block_table.shape[1] * BLOCK,
        )

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
        if self.is_initialized:
            assert self.keys is not None and self.values is not None
            self.keys.zero_()
            self.values.zero_()
            self.cumulative_length.zero_()
            self._host_length = 0
            self.context_length = 0
            self.writing_branches = False
            if self.block_table is not None:
                self.block_table.zero_()
            if self.seqused is not None:
                self.seqused.zero_()

    def rewind_to(self, length: int) -> None:
        """Start another group from the end of the context, re-copying each row's remainder.

        The re-copy matters: the previous group wrote over the remainder's slots with its own first tokens if the
        context
        did not end on a page boundary. Without it the next group reads the last group's tokens as context.
        """
        if not self.is_initialized:
            return
        if self.context_length and length != self.context_length:
            raise ValueError(
                f"this layer can only rewind to the end of its context ({self.context_length} tokens), not to {length}"
            )
        self.cumulative_length.fill_(length)
        self._host_length = int(length)
        self._copy_remainder()
        if self.writing_branches:
            self._set_lengths()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Permute the table rather than the storage: the rows are page lists, and moving bytes would be pointless."""
        if self.is_initialized and self.block_table is not None:
            picked = self.block_table.index_select(0, beam_idx.to(self.block_table.device))
            self.block_table.copy_(picked)

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError(
            "this cache cannot be cropped: the context and the branches live in different pages, so there is no single "
            "run of tokens to take from the end. Use rewind_to(context_length) to start another group."
        )

    def offload(self) -> None:
        """Not supported: the pages are addressed by index into one allocation."""

    def prefetch(self) -> None:
        pass
