"""What the cache hands back, and what it refuses. No device needed: this is indexing, and indexing is where it breaks.

The change these cover is first that the context is stored once instead of once per branch, and now that a branch
*read* no longer joins the context with each row's own tokens into a dense copy either -- it shares a page table,
the same mechanism `paged.PagedForkLayer` hands the kernel. The property that makes it safe to make is that a read
through that table gathers **exactly** what replicating would have returned -- the same bytes in the same order --
so these tests reconstruct that gather by hand and compare against a replication done by hand too, rather than
against a recorded number.
"""

from __future__ import annotations

import pytest
import torch

from prismyra.cache import ForkLayer, cache_bytes
from prismyra.paged import BLOCK


def layer(rows: int = 4, context: int = 20, branch: int = 8, heads: int = 2, dim: int = 3) -> ForkLayer:
    made = ForkLayer(max_cache_len=context + branch, max_batch_size=rows, max_branch_len=branch)
    made.early_initialization(rows, heads, dim, torch.float32, "cpu")
    return made


def with_context(held: ForkLayer, count: int = 12) -> torch.Tensor:
    """Write a context and hand over to branches, which is the sequence every caller follows.

    The hand-over is explicit because a row count cannot imply it: at a group of one, the context and a branch are
    each one row. Leaving it implicit meant a group of one wrote its context into the branch buffer.
    """
    context = tokens(count)
    held.update(context, context * -1)
    held.begin_branches()
    return context


def tokens(count: int, heads: int = 2, dim: int = 3, rows: int = 1, start: float = 0.0) -> torch.Tensor:
    """Distinguishable values, so a misplaced write is visible rather than plausible."""
    total = rows * heads * count * dim
    return (torch.arange(total, dtype=torch.float32) + start).reshape(rows, heads, count, dim)


def gathered(held: ForkLayer, rows: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """What a branch read actually hands the kernel, reassembled into one dense `(rows, heads, k_len, dim)` tensor
    per row by following that row's own table -- the same gather `unified_attention` would do, done here in plain
    indexing so a test can compare it against a replication done by hand.

    Returns the keys, the values, and `k_len` -- every row's own `seqused`, which this layer always makes equal
    because it only ever holds one document and one branch write advances every row's length by the same count.
    """
    keys, values, table, seqused, _capacity = held.branch_read(rows)
    k_len = int(seqused[0].item())
    assert bool((seqused == k_len).all()), "every row of one document should hold the same length"
    take = table.shape[0]
    pages_needed = -(-k_len // BLOCK)
    gathered_keys = keys[table[:, :pages_needed]].reshape(take, pages_needed * BLOCK, *keys.shape[2:])[:, :k_len]
    gathered_values = values[table[:, :pages_needed]].reshape(take, pages_needed * BLOCK, *values.shape[2:])[:, :k_len]
    return gathered_keys.transpose(1, 2), gathered_values.transpose(1, 2), k_len


def test_a_branch_read_is_exactly_what_replicating_would_have_returned():
    """The property the whole change rests on. Not a tolerance: the same bytes in the same order."""
    held = layer(rows=4, context=20, branch=8)
    context = with_context(held)

    branch = tokens(5, rows=4, start=1000)
    keys_out, values_out = held.update(branch, branch * -1)
    assert keys_out is None and values_out is None, "a branch write returns nothing contiguous; branch_read does"

    keys, values, k_len = gathered(held, rows=4)
    expected_keys = torch.cat((context.expand(4, -1, -1, -1), branch), dim=-2)
    assert k_len == 17
    assert torch.equal(keys, expected_keys)
    assert torch.equal(values, expected_keys * -1)
    assert keys.shape == (4, 2, 17, 3)


def test_the_context_is_stored_once():
    """The point of the change, asserted on the storage rather than inferred from a memory figure."""
    held = layer(rows=32, context=100, branch=8)
    assert held.keys.shape[0] == 1, "the context must occupy one row however many branches read it"
    assert held.branch_keys.shape[0] == 32, "each branch needs its own tokens"


def test_a_context_write_returns_one_row():
    """The model's own attention uses this return value during the context pass, and it is a one-row pass."""
    held = layer(rows=4, context=20)
    context = tokens(12)
    context = tokens(12)
    keys, _ = held.update(context, context)
    assert keys.shape == (1, 2, 12, 3)
    assert torch.equal(keys, context)


def test_a_second_group_starts_from_the_context_and_overwrites_the_first_group_tokens():
    """The mistake most likely to be made, and invisible without more than one group.

    The first group advances the length by its own tokens. The second has to start from the end of the context again
    and write over the first group's, rather than after them.
    """
    held = layer(rows=4, context=20, branch=8)
    context = with_context(held)

    first = tokens(5, rows=4, start=1000)
    held.update(first, first)
    assert held.get_seq_length() == 17

    held.rewind_to(held.context_length)
    assert held.get_seq_length() == 12

    second = tokens(3, rows=4, start=9000)
    held.update(second, second)
    keys, _, k_len = gathered(held, rows=4)
    assert k_len == 15
    assert keys.shape == (4, 2, 15, 3)
    # The second group's tokens are there and the first group's are gone from the read, not appended after.
    assert torch.equal(keys[:, :, 12:], second)
    assert torch.equal(keys[:, :, :12], context.expand(4, -1, -1, -1))


def test_groups_of_the_same_shape_read_identically():
    """Bit-exact across groups, which a stale cursor or an unreset buffer would break."""
    held = layer(rows=4, context=20, branch=8)
    with_context(held)

    branch = tokens(5, rows=4, start=1000)
    held.update(branch, branch)
    first, _, _ = gathered(held, rows=4)
    first = first.clone()

    held.rewind_to(held.context_length)
    other = tokens(4, rows=4, start=7000)
    held.update(other, other)
    held.rewind_to(held.context_length)

    held.update(branch, branch)
    third, _, _ = gathered(held, rows=4)
    assert torch.equal(first, third)


def test_a_branch_offset_is_measured_from_the_context_not_from_a_position():
    """With media in a context the model's positions run ahead of the token count. A branch's storage offset must come
    from the token count, and this checks the layer never learns a position at all."""
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    assert held.context_length == 12
    assert held.get_seq_length() == 12
    # There is nowhere to put a position, which is the strongest form of the guarantee. `_remainder` is derived from
    # `context_length` rather than a position and is excluded for the same reason `context_length` itself would be.
    assert not any("position" in name for name in vars(held))


def test_a_context_that_does_not_fit_is_refused():
    held = layer(rows=4, context=20, branch=8)
    too_long = tokens(21)
    with pytest.raises(ValueError, match="does not fit"):
        held.update(too_long, too_long)


def test_a_group_of_one_writes_its_context_to_the_context_buffer():
    """Found by a benchmark rather than a test, which is why it is a test now.

    At a group of one a context write and a branch write are each one row, so a layer deciding by row count put a
    3,040-token context into a buffer sized for a 512-token branch and refused it. Only being told tells them apart.
    """
    held = layer(rows=1, context=20, branch=8)
    context = tokens(12)
    keys, _ = held.update(context, context)
    assert keys.shape == (1, 2, 12, 3)

    held.begin_branches()
    branch = tokens(5, rows=1, start=1000)
    held.update(branch, branch)
    joined, _, k_len = gathered(held, rows=1)
    assert k_len == 17
    assert joined.shape == (1, 2, 17, 3)
    assert torch.equal(joined[:, :, 12:], branch)


def test_a_branch_longer_than_the_buffer_is_refused():
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    too_long = tokens(9, rows=4)
    with pytest.raises(ValueError, match="does not fit in 8"):
        held.update(too_long, too_long)


def test_more_rows_than_the_layer_holds_is_refused():
    held = layer(rows=4, context=20)
    with_context(held)
    with pytest.raises(ValueError, match="holds 4 branch rows and was given 6"):
        held.update(tokens(3, rows=6), tokens(3, rows=6))


def test_a_narrow_group_reads_exactly_what_a_full_one_would_have_read():
    """Fewer rows than the layer holds is the ordinary case for a request with few questions, and the saving is real:
    the shared pages are a view whatever the row count, and only the (small) private region is per row -- so a group
    of two no longer carries anything proportional to a group of four, let alone to the context.

    Checked against the same layer answering the same two branches in a full-width group, because the whole point is
    that narrowing changes the cost and not the answer."""
    narrow = layer(rows=4, context=20, branch=8)
    with_context(narrow)
    branch = tokens(5, rows=2, start=1000)
    narrow.update(branch, branch * -1)
    keys, values, _ = gathered(narrow, rows=2)
    assert keys.shape == (2, 2, 17, 3)

    wide = layer(rows=4, context=20, branch=8)
    with_context(wide)
    four = torch.cat((branch, tokens(5, rows=2, start=7000)), dim=0)
    wide.update(four, four * -1)
    wide_keys, wide_values, _ = gathered(wide, rows=4)
    assert torch.equal(keys, wide_keys[:2])
    assert torch.equal(values, wide_values[:2])


def test_a_narrow_group_does_not_read_the_rows_it_did_not_write():
    """The rows beyond the group hold whatever the last group left there, and a read that returned them would mix one
    request's branches into another's."""
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    held.update(tokens(5, rows=4, start=1000), tokens(5, rows=4, start=1000))
    held.rewind_to(held.context_length)
    branch = tokens(5, rows=2, start=9000)
    held.update(branch, branch)
    keys, _, _ = gathered(held, rows=2)
    assert keys.shape[0] == 2


def test_rewinding_anywhere_but_the_context_end_is_refused():
    """A rewind into the context would leave the branch buffer describing tokens the length no longer claims."""
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    held.update(tokens(5, rows=4), tokens(5, rows=4))
    with pytest.raises(ValueError, match="only rewind to the end of its context"):
        held.rewind_to(6)


def test_cropping_is_refused_rather_than_half_done():
    held = layer()
    with pytest.raises(NotImplementedError, match="cannot be cropped"):
        held.crop(3)


def test_reordering_permutes_the_branches_and_leaves_the_context_alone():
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    branch = tokens(5, rows=4, start=1000)
    held.update(branch, branch)

    held.reorder_cache(torch.tensor([3, 2, 1, 0]))
    # Storage is token-major, `(rows, tokens, heads, dim)`, while the branch written above is head-major as every
    # caller's is. So the comparison transposes rather than asserting on a layout this class does not use. The first
    # `context_length % BLOCK` slots are this row's copy of the context's partial last page, made once by
    # `begin_branches` and identical for every row, so the branch's own tokens start right after it.
    remainder = held.context_length % BLOCK
    assert torch.equal(held.branch_keys[0, remainder : remainder + 5], branch[3].transpose(0, 1))
    assert held.keys.shape[0] == 1


def test_resetting_clears_both_buffers_and_the_context_mark():
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    held.update(tokens(5, rows=4), tokens(5, rows=4))
    held.reset()
    assert held.get_seq_length() == 0
    assert held.context_length == 0
    assert held.keys.abs().sum().item() == 0.0
    assert held.branch_keys.abs().sum().item() == 0.0


def test_the_memory_figure_counts_the_context_once():
    """The number that decides how many contexts fit on a card, so it is checked against the arithmetic rather than
    against a remembered value."""

    from typing import ClassVar

    class Config:
        num_key_value_heads = 2
        num_attention_heads = 16
        head_dim = 256
        hidden_size = 2048
        layer_types: ClassVar[list[str]] = ["full_attention", "linear_attention", "full_attention"]

    branch, context, rows = 512, 3040, 32
    got = cache_bytes(Config(), context + branch, rows, torch.bfloat16, branch)
    # `ForkLayer`'s own branch buffer is rounded up to whole pages, the same rounding `cache_bytes`'s `paged` branch
    # uses: room for the context's worst-case partial last page (`BLOCK - 1` tokens) ahead of the branch's own, so
    # the buffer's shape does not depend on a particular context's actual remainder. At `branch=512` (already a
    # multiple of `BLOCK`) that is one extra page per row.
    private_pages = -(-(BLOCK - 1 + branch) // BLOCK)
    expected = 2 * 2 * 2 * 256 * (context + rows * private_pages * BLOCK) * 2
    assert got == expected
    assert private_pages * BLOCK > branch, "the rounding this test checks for must not have vanished"

    # Replicating the context across the group is what this replaces, and it is much larger.
    replicated = 2 * 2 * 2 * 256 * rows * (context + branch) * 2
    assert replicated / got > 5


def test_the_layer_records_the_row_count_it_was_actually_given():
    """A witness for the claim that a narrow group runs narrow.

    Worth its own test because the alternative is trusting a caller's intention. A flag that validated its arguments,
    reported itself in `stats()`, and never reached the code it configured is what made a whole round of measurements
    worthless, and the defence is to ask the code that would have done the work whether it did.
    """
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    assert held.last_branch_rows == 0
    held.update(tokens(3, rows=2), tokens(3, rows=2))
    assert held.last_branch_rows == 2
    held.rewind_to(held.context_length)
    held.update(tokens(3, rows=4), tokens(3, rows=4))
    assert held.last_branch_rows == 4


def test_a_branch_read_actually_ran():
    """A witness for the read itself, for the same reason `last_branch_rows` witnesses the write.

    The paged pool this mechanism was borrowed from was once wired to nothing and reported itself installed anyway
    (`docs/PERFORMANCE.md`, "the paged path that never ran"). `pages_read` and the class-level counter beside it are
    what this file checks instead of trusting that `branch_read` was called because the kernel is supposed to call
    it.
    """
    held = layer(rows=4, context=20, branch=8)
    with_context(held)
    held.update(tokens(5, rows=4), tokens(5, rows=4))
    before = ForkLayer.reads_served
    assert held.pages_read == 0
    held.branch_read(4)
    assert held.pages_read == 1
    assert ForkLayer.reads_served == before + 1


def test_slicing_the_context_into_whole_pages_is_a_view_not_a_copy():
    """Most of the saving, asserted directly on the step that makes it possible.

    `branch_read` still copies the context's shared pages once per read -- the kernel needs one tensor it can index
    by one table, so the view below is concatenated with the (small) private region before it reaches the kernel,
    and `cat` always allocates. What `branch_read` does *not* do is pay for that slice once per row: the input to
    the `cat` is this view, taken fresh from storage every time at no cost, rather than a second buffer that held a
    standing copy. This is the step the test isolates."""
    held = layer(rows=4, context=32, branch=8)
    with_context(held, count=32)  # a whole number of BLOCK=16 pages, so there is no partial last page to copy
    assert held.context_length % BLOCK == 0
    shared_pages = held.context_length // BLOCK
    view = held.keys[0, : shared_pages * BLOCK].view(shared_pages, BLOCK, *held.keys.shape[2:])
    assert view.data_ptr() == held.keys.data_ptr()
    assert view.untyped_storage().data_ptr() == held.keys.untyped_storage().data_ptr()
