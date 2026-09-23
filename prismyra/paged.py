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

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Held:
    """One document in the pool: where its pages start, and how many tokens it holds.

    A document reserves its whole pages **and**, when its length is not a multiple of the page size, one more to hold
    the leftover tokens until they are copied into each row. That page is reserved rather than borrowed: the single
    document version wrote the leftover into the page just past its own, which is unowned when there is one document and
    is the **next document's first page** when there are two. The failure would have been rows reading another
    document's opening tokens as the end of their own.
    """

    first_page: int
    tokens: int
    block: int = BLOCK

    @property
    def whole_pages(self) -> int:
        return self.tokens // self.block

    @property
    def remainder(self) -> int:
        return self.tokens % self.block

    @property
    def reserved_pages(self) -> int:
        """Pages this document holds in the shared region, including the one staging its leftover."""
        return self.whole_pages + (1 if self.remainder else 0)

    @property
    def staging_page(self) -> int:
        """Where the leftover tokens are written, to be copied into each row. Only meaningful when there is a
        remainder, and asking for it when there is none is a bug rather than a no-op."""
        if not self.remainder:
            raise ValueError(f"a document of {self.tokens} tokens ends on a page boundary and stages nothing")
        return self.first_page + self.whole_pages

    def pages(self) -> list[int]:
        """The run of whole pages every row answering about this document names. The staging page is not among them:
        a page that is half document cannot sit inside a row's run."""
        return list(range(self.first_page, self.first_page + self.whole_pages))


class Full(RuntimeError):
    """The pool has no room for another document. Raised before anything is written, so a refused admission leaves the
    pool exactly as it was -- a partial write across ten layers has no rollback."""


@dataclass
class Pool:
    """How a page pool divides between several documents and the rows answering about them.

    This is the arithmetic that lets **one forward pass carry questions about different documents**, which is the one
    thing the joined storage cannot do: `ForkLayer` puts the context in one row and joins it with each branch's own
    tokens, so every row in a pass reads the same context by construction. Pages are named per row, so they do not.

    The layout is fixed at two ends and grows in the middle:

    * the **tail** holds each row's private pages -- its copy of its document's remainder, then its own tokens. Sized
      for the worst remainder (`block - 1`) rather than for a particular document's, so a row's private pages are at the
      same addresses whatever document it is answering about. A recording holds addresses;
    * the **head** holds documents' whole pages, handed out by a cursor as documents are admitted.

    A row's table is then `[its document's pages] + [its own private pages]` and the rest of the row is padding that is
    never named, because `seqused_k` says how far along each row the kernel reads. Two rows answering about different
    documents therefore differ in the **contents** of a rectangular table and not in its shape, which is what keeps one
    recording valid.
    """

    total_pages: int
    rows: int
    branch_tokens: int
    block: int = BLOCK
    #: Admitted documents, in admission order.
    documents: list[Held] = field(default_factory=list)
    #: Which document each row answers about, by index into `documents`. Shorter than `rows` while rows are unassigned.
    assignment: list[int] = field(default_factory=list)

    @property
    def private_pages(self) -> int:
        """Pages one row owns. Sized for the worst remainder so the addresses do not depend on the document."""
        return -(-(self.block - 1 + self.branch_tokens) // self.block)

    @property
    def private_first(self) -> int:
        """Where the private region begins. Everything below this is shared between rows."""
        return self.total_pages - self.rows * self.private_pages

    def private_range(self, row: int) -> tuple[int, int]:
        first = self.private_first + row * self.private_pages
        return first, first + self.private_pages

    @property
    def used_pages(self) -> int:
        return sum(d.reserved_pages for d in self.documents)

    @property
    def free_pages(self) -> int:
        return self.private_first - self.used_pages

    def pages_for(self, tokens: int) -> int:
        """Pages a document of this length would reserve: its whole pages, plus one to stage a leftover."""
        return -(-tokens // self.block)

    def room_for(self, tokens: int) -> bool:
        return self.pages_for(tokens) <= self.free_pages

    def admit(self, tokens: int) -> Held:
        """Reserve the whole pages for a document and return where they are. Refuses before writing anything."""
        if tokens < 0:
            raise ValueError(f"a document cannot hold {tokens} tokens")
        if not self.room_for(tokens):
            raise Full(
                f"a document of {tokens} tokens needs {self.pages_for(tokens)} pages and {self.free_pages} are free; "
                f"close a document or build the cache for a longer context"
            )
        held = Held(first_page=self.used_pages, tokens=tokens, block=self.block)
        self.documents.append(held)
        return held

    def assign(self, per_document: list[int]) -> None:
        """Say how many rows answer about each admitted document, in admission order.

        Given rather than inferred. A row count per document is the caller's scheduling decision, and a pool that
        guessed it would answer the wrong document's questions plausibly.
        """
        if len(per_document) > len(self.documents):
            raise ValueError(f"{len(per_document)} row counts for {len(self.documents)} documents")
        assignment: list[int] = []
        for d, count in enumerate(per_document):
            assignment += [d] * count
        if len(assignment) > self.rows:
            raise ValueError(f"{len(assignment)} rows asked for and this pool holds {self.rows}")
        self.assignment = assignment

    def width(self) -> int:
        """Columns the table needs: the longest document's pages plus one row's private pages."""
        longest = max((d.whole_pages for d in self.documents), default=0)
        return longest + self.private_pages

    def table(self) -> list[list[int]]:
        """One row per assigned row: its document's pages, then its own, then padding that is never named."""
        width = self.width()
        rows = []
        for row, d in enumerate(self.assignment):
            named = self.documents[d].pages() + list(range(*self.private_range(row)))
            rows.append(named + [0] * (width - len(named)))
        return rows

    def lengths(self, branch_progress: int = 0) -> list[int]:
        """Tokens each row holds: its document's, plus however far its branch has got."""
        return [self.documents[d].tokens + branch_progress for d in self.assignment]

    def branch_offset(self, row: int) -> int:
        """Where a row's own tokens begin inside its private pages -- after its copy of its document's remainder."""
        return self.documents[self.assignment[row]].remainder


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
        #: The same pages, described as a pool several documents can share. `None` until allocation.
        self.pool: Pool | None = None
        #: Documents admitted and not yet released, by the handle the engine gave them. A handle rather than an index so
        #: that releasing one cannot silently renumber another.
        self.held: dict[int, Held] = {}
        #: Which document each row of the group being answered belongs to, as handles in row order.
        self.rows_for: list[int] = []
        #: The document being written. A handle chosen by the caller; zero is the one a single-context caller gets
        #: without naming it.
        self._writing = 0
        #: How far the group being answered has advanced past its documents. Counted rather than derived from
        #: `_host_length`, because with several documents in the batch there is no single length to subtract.
        self._branch_progress = 0
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
        """Sized for the longest context this cache was built for, plus one row's worth of private pages per row.

        The pages are a pool, so the allocation that holds one long context also holds several short ones. What is not
        here is **reuse**: a page a released document held is not handed out again, because that needs a rule for
        reusing a page while earlier work may still be reading it. Until then the cursor only moves forward and
        `Pool.admit` refuses when it reaches the private region.
        """
        room = max(1, self.max_cache_len - self.max_branch_len)
        self.device, self.dtype = device, dtype
        # Sized by the pool, which is the only description of these pages. The single-document `Layout` remains as the
        # arithmetic a reader can check by hand, and is no longer what the storage is built from: two descriptions of
        # one allocation is how a row ends up naming a page that belongs to something else.
        rows = self.max_batch_size
        private = -(-(BLOCK - 1 + self.max_branch_len) // BLOCK)
        total = -(-room // BLOCK) + rows * private
        self.pool = Pool(total_pages=total, rows=rows, branch_tokens=self.max_branch_len, block=BLOCK)
        assert self.pool.private_pages == private
        self.layout = Layout(context_tokens=room, rows=rows, branch_tokens=self.max_branch_len, block=BLOCK)
        self.keys = torch.zeros((total, BLOCK, heads, head_dim), dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)
        # Allocated at the widest layout so a recording's addresses stay valid whatever context is opened next. The
        # table's own width shrinks with the context; the tensor's does not, and the unused columns are never named
        # because `seqused` says how far along each row the kernel should read.
        pages = -(-room // BLOCK) + private
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
                    "this cache takes a document in one piece; a second write to the same one would have to gather "
                    "the first back out of its pages. Another document is admitted with begin_document()."
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
        """Into the pages this document reserved, in page order. One row, so there is one copy.

        The document is admitted here rather than earlier, because the token count that decides how many pages it needs
        is what arrives with the write. An admission that will not fit raises before anything is written.
        """
        assert self.keys is not None and self.values is not None and self.pool is not None
        count = key_states.shape[-2]
        held = self.pool.admit(count)
        self.held[self._writing] = held
        # (1, heads, tokens, dim) -> (tokens, heads, dim), which is the page layout's own order.
        keys = key_states[0].transpose(0, 1)
        values = value_states[0].transpose(0, 1)
        whole = held.whole_pages
        at = held.first_page
        if whole:
            self.keys[at : at + whole] = keys[: whole * BLOCK].view(whole, BLOCK, *keys.shape[1:])
            self.values[at : at + whole] = values[: whole * BLOCK].view(whole, BLOCK, *values.shape[1:])
        if held.remainder:
            # Into this document's own staging page. Writing it just past the whole pages would be writing into whatever
            # is admitted next.
            page = held.staging_page
            self.keys[page, : held.remainder] = keys[whole * BLOCK :]
            self.values[page, : held.remainder] = values[whole * BLOCK :]

    def begin_document(self, handle: int) -> None:
        """About to write the document called `handle`. Its pages are reserved when its tokens arrive.

        A handle rather than a position, so that releasing one document cannot renumber another. A caller that never
        names one writes document zero, which is what a single-context caller does.
        """
        if handle in self.held:
            raise ValueError(f"document {handle} is already in this cache")
        self._writing = handle
        self._host_length = 0
        self.cumulative_length.zero_()
        self.writing_branches = False

    def begin_branches(self, rows_for: list[int] | None = None) -> None:
        """The documents are finished and a group is starting. Build the table and the lengths, in place.

        `rows_for` names the document each row answers about, in row order. Without it every row answers about the last
        document written, which is the single-context case and is what every existing caller means.

        In place because a recording holds these tensors' addresses. Both need to know how long each document turned out
        to be, which is why this cannot happen at allocation.
        """
        if self.writing_branches and rows_for is None:
            return
        assert self.keys is not None and self.block_table is not None and self.seqused is not None
        assert self.pool is not None
        if not self.held:
            # No document was ever admitted -- a group asked for before anything was read. The caller's mistake, and
            # saying so beats a table of zeros that reads page zero as though it held something.
            raise ValueError("this cache holds no document, so there is nothing for a branch to read")
        if rows_for is None:
            rows_for = [self._writing] * self.max_batch_size
        unknown = [h for h in rows_for if h not in self.held]
        if unknown:
            raise ValueError(f"rows were assigned to documents this cache does not hold: {sorted(set(unknown))}")

        self.rows_for = list(rows_for)
        self.writing_branches = True
        self._branch_progress = 0
        # The longest document in the batch. Reported to the framework, which has one length to ask about; the per-row
        # truth is in `seqused` and that is what the kernel reads.
        self.context_length = max(self.held[h].tokens for h in rows_for)
        self._host_length = self.context_length

        order = list(dict.fromkeys(rows_for))
        self.pool.documents = [self.held[h] for h in order]
        self.pool.assign([sum(1 for h in rows_for if h == handle) for handle in order])
        table = torch.tensor(self.pool.table(), dtype=torch.int32, device=self.keys.device)
        self.block_table.zero_()
        self.block_table[: table.shape[0], : table.shape[1]] = table
        self.layout = Layout(
            context_tokens=self.context_length,
            rows=self.max_batch_size,
            branch_tokens=self.max_branch_len,
            block=BLOCK,
        )
        self._copy_remainder()
        self._set_lengths()

    def _copy_remainder(self) -> None:
        """Each row's copy of its own document's leftover tokens, from that document's staging page.

        Per row and from that row's **own** document. Taking one document's leftover for every row is the failure
        this whole file is arranged around, and it would not raise.
        """
        assert self.keys is not None and self.values is not None and self.pool is not None
        for row, handle in enumerate(self.rows_for):
            held = self.held[handle]
            if not held.remainder:
                continue
            first, _ = self.pool.private_range(row)
            self.keys[first, : held.remainder] = self.keys[held.staging_page, : held.remainder]
            self.values[first, : held.remainder] = self.values[held.staging_page, : held.remainder]

    def _set_lengths(self) -> None:
        """How many tokens each row holds, as device data rather than as a shape. This is the mechanism: a document's
        length moves a number in this tensor and moves nothing about the read's shape -- and with several documents
        the numbers differ from row to row while the shape does not."""
        assert self.seqused is not None and self.pool is not None
        if not self.rows_for:
            return
        lengths = self.pool.lengths(branch_progress=self._branch_progress)
        # Rows beyond the group keep the first row's length rather than zero: a zero here would make the kernel read
        # nothing for a row that is never queried, which is harmless, and a wrong non-zero would not be. Filled first so
        # that no slot is left from a previous group.
        self.seqused.fill_(lengths[0])
        self.seqused[: len(lengths)] = torch.tensor(lengths, dtype=torch.int32, device=self.seqused.device)

    def _write_branches(self, key_states: torch.Tensor, value_states: torch.Tensor, rows: int) -> None:
        """Into each row's own pages, at an offset measured from **that row's** document, never from a position."""
        assert self.keys is not None and self.values is not None and self.pool is not None
        pool = self.pool
        count = key_states.shape[-2]
        keys = key_states.transpose(1, 2)  # (rows, tokens, heads, dim)
        values = value_states.transpose(1, 2)
        for row in range(rows):
            at = pool.branch_offset(row) + self._branch_progress
            if at + count > pool.private_pages * BLOCK:
                raise ValueError(
                    f"row {row} reached token {at + count} of {pool.private_pages * BLOCK} private slots; the widest "
                    f"branch is what these pages were sized for"
                )
            first, _ = pool.private_range(row)
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
            self._branch_progress += count
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
        """Empty, and empty of documents. The pool's cursor goes back to the start too: a reset that cleared the bytes
        and kept the admissions would refuse the next document for room it no longer owes anyone."""
        if self.is_initialized:
            assert self.keys is not None and self.values is not None
            self.keys.zero_()
            self.values.zero_()
            self.cumulative_length.zero_()
            self._host_length = 0
            self.context_length = 0
            self.writing_branches = False
            self._branch_progress = 0
            self._writing = 0
            self.held.clear()
            self.rows_for = []
            if self.pool is not None:
                self.pool.documents = []
                self.pool.assignment = []
            if self.block_table is not None:
                self.block_table.zero_()
            if self.seqused is not None:
                self.seqused.zero_()

    def rewind_to(self, length: int) -> None:
        """Start another group from the end of the documents, re-copying each row's own remainder.

        The re-copy matters: if a document did not end on a page boundary, the previous group wrote its own first tokens
        over the slots holding that document's leftover. Without the re-copy the next group reads the last group's
        tokens as the end of its document, and answers plausibly.

        `length` is the longest document in the batch, which is what this layer reports holding. It is checked rather
        than used: each row's own length is `Pool.lengths`, and rewinding to anything but the end of the documents is
        not something this storage can do.
        """
        if not self.is_initialized:
            return
        if self.context_length and length != self.context_length:
            raise ValueError(
                f"this layer can only rewind to the end of its documents ({self.context_length} tokens, the longest in "
                f"the batch), not to {length}"
            )
        self.cumulative_length.fill_(length)
        self._host_length = int(length)
        self._branch_progress = 0
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
