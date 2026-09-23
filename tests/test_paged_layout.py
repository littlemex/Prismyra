"""The page arithmetic, and the one property the whole reason for this path rests on.

Eighteen tests of this kind existed before and passed while the code they described was never called, which is why the
last test here is the important one: it asserts that the shapes a read presents do **not** move with the context length.
That is the property a recorded graph needs, and it is checkable without a device.
"""

from __future__ import annotations

import pytest
import torch

from prismyra.paged import BLOCK, Layout, PagedForkLayer


def layer(context: int, rows: int = 4, branch: int = 8) -> PagedForkLayer:
    made = PagedForkLayer(max_cache_len=context + branch, max_batch_size=rows, max_branch_len=branch)
    made.early_initialization(rows, 2, 3, torch.float32, "cpu")
    return made


def tokens(count: int, heads: int = 2, dim: int = 3, rows: int = 1, start: float = 0.0) -> torch.Tensor:
    total = rows * heads * count * dim
    return (torch.arange(total, dtype=torch.float32) + start).reshape(rows, heads, count, dim)


def test_a_whole_number_of_pages_is_shared_and_a_remainder_is_not():
    exact = Layout(context_tokens=BLOCK * 3, rows=4, branch_tokens=8)
    assert exact.shared_pages == 3
    assert exact.remainder == 0
    over = Layout(context_tokens=BLOCK * 3 + 5, rows=4, branch_tokens=8)
    assert over.shared_pages == 3
    assert over.remainder == 5


def test_every_row_names_the_same_shared_pages_and_its_own_private_ones():
    made = Layout(context_tokens=BLOCK * 2, rows=3, branch_tokens=BLOCK)
    table = made.table()
    assert len(table) == 3
    shared = table[0][: made.shared_pages]
    for row in table:
        assert row[: made.shared_pages] == shared, "the shared pages are the mechanism and must be identical"
    private = [tuple(row[made.shared_pages :]) for row in table]
    assert len(set(private)) == 3, "a private page named by two rows would let one branch write into another"


def test_the_remainder_is_copied_into_each_row_rather_than_shared():
    """A page half context and half nothing cannot sit in the middle of a run the kernel reads as one sequence."""
    held = layer(context=BLOCK + 5)
    context = tokens(BLOCK + 5)
    held.update(context, context * -1)
    held.begin_branches()
    # The pool is where the pages are, and it is the only description of them: `Layout` is arithmetic a reader can check
    # by hand, and asking it where a row's pages are gave the answer from before documents could share a pool.
    pool = held.pool
    document = held.held[0]
    assert document.remainder == 5
    for row in range(pool.rows):
        first, _ = pool.private_range(row)
        assert torch.equal(held.keys[first, :5], held.keys[document.staging_page, :5])


def test_the_read_does_not_present_a_shape_that_depends_on_the_context_length():
    """The whole reason this path exists. Two contexts of very different lengths, the same shapes out.

    A joined read is `(rows, context + suffix, heads, dim)`, so every length is a different shape and every length needs
    its own recorded graph. Here the pool, the table and the lengths are allocated once for the longest context, and the
    context's length moves a *number* in `seqused` rather than a dimension.
    """
    shapes = []
    for context in (BLOCK * 2, BLOCK * 30 + 7):
        held = PagedForkLayer(max_cache_len=BLOCK * 64, max_batch_size=4, max_branch_len=BLOCK * 2)
        held.early_initialization(4, 2, 3, torch.float32, "cpu")
        written = tokens(context)
        held.update(written, written * -1)
        held.begin_branches()
        branch = tokens(4, rows=4, start=900)
        held.update(branch, branch * -1)
        keys, values, table, seqused, capacity = held.paged_read()
        shapes.append((keys.shape, values.shape, table.shape, seqused.shape, capacity))
    assert shapes[0] == shapes[1], "a shape moved with the context length, which defeats the point of this file"


def test_a_read_before_the_context_is_finished_is_refused():
    held = layer(context=BLOCK)
    written = tokens(BLOCK)
    held.update(written, written * -1)
    with pytest.raises(AssertionError, match="context must be finished"):
        held.paged_read()


def test_the_layer_counts_the_reads_it_served():
    """The witness. A flag that said `paged` while nothing ran is what made the first attempt worthless."""
    held = layer(context=BLOCK)
    written = tokens(BLOCK)
    held.update(written, written * -1)
    held.begin_branches()
    assert held.pages_read == 0
    branch = tokens(3, rows=4, start=500)
    held.update(branch, branch * -1)
    held.paged_read()
    held.paged_read()
    assert held.pages_read == 2


def test_a_branch_wider_than_its_private_pages_is_refused():
    held = layer(context=BLOCK, branch=BLOCK)
    written = tokens(BLOCK)
    held.update(written, written * -1)
    held.begin_branches()
    too_long = tokens(BLOCK * 3, rows=4)
    with pytest.raises(ValueError, match="private slots"):
        held.update(too_long, too_long * -1)


def test_cropping_is_refused_because_there_is_no_single_run_to_take_from():
    held = layer(context=BLOCK)
    with pytest.raises(NotImplementedError, match="cannot be cropped"):
        held.crop(3)


def two_documents(
    first: int, second: int, rows: int = 4, branch: int = 8, capacity: int | None = None
) -> PagedForkLayer:
    """One layer holding two documents, written one after the other, with distinguishable bytes.

    A cache big enough for both: the pool's cursor only moves forward, so the capacity has to cover the sum. `capacity`
    is separate from the documents' lengths so that two cases can be compared at the **same** allocation -- sizing it
    from the documents made the shape comparison below compare two different pools and fail for that reason.
    """
    held = layer(context=capacity if capacity is not None else first + second + 2 * BLOCK, rows=rows, branch=branch)
    held.begin_document(0)
    a = tokens(first, start=1000.0)
    held.update(a, a * -1)
    held.begin_document(1)
    b = tokens(second, start=90000.0)
    held.update(b, b * -1)
    return held


def test_two_documents_live_in_one_pool_without_overlapping():
    held = two_documents(BLOCK * 2, BLOCK * 3)
    one, two = held.held[0], held.held[1]
    assert set(one.pages()).isdisjoint(two.pages())
    # And the bytes are where the table says. Document zero opens with 1000.0 and document one with 90000.0.
    assert held.keys[one.pages()[0], 0, 0, 0].item() == 1000.0
    assert held.keys[two.pages()[0], 0, 0, 0].item() == 90000.0


def test_each_row_reads_only_its_own_document():
    """The property continuous batching across documents rests on. A row's table must name its own document's pages and
    no others, and this is asserted on the tensor the kernel is handed rather than on the arithmetic behind it."""
    held = two_documents(BLOCK * 2, BLOCK * 3)
    held.begin_branches([0, 0, 1, 1])
    _, _, table, seqused, _ = held.paged_read(rows=4)
    one, two = held.held[0], held.held[1]
    for row in (0, 1):
        named = set(table[row].tolist()[: one.whole_pages])
        assert named == set(one.pages())
        assert named.isdisjoint(two.pages())
    for row in (2, 3):
        named = set(table[row].tolist()[: two.whole_pages])
        assert named == set(two.pages())
        assert named.isdisjoint(one.pages())
    # And each row is told its own document's length, which is how one shape serves two lengths.
    assert seqused.tolist() == [BLOCK * 2, BLOCK * 2, BLOCK * 3, BLOCK * 3]


def test_a_mixed_batch_keeps_one_shape_across_two_document_lengths():
    """Two documents of different lengths, and the read presents the same shapes as one document would. Without this a
    mixed batch would need its own recording, which is the cost the paged path exists to avoid."""
    room = BLOCK * 32
    same = two_documents(BLOCK * 2, BLOCK * 2, capacity=room)
    mixed = two_documents(BLOCK * 2, BLOCK * 5, capacity=room)
    same.begin_branches([0, 0, 1, 1])
    mixed.begin_branches([0, 0, 1, 1])
    shapes = []
    for held in (same, mixed):
        keys, values, table, seqused, bound = held.paged_read(rows=4)
        shapes.append((keys.shape, values.shape, table.shape, seqused.shape, bound))
    assert shapes[0] == shapes[1], shapes


def test_each_row_copies_its_own_documents_leftover():
    """Two documents whose lengths differ modulo the page size. A row taking the other document's leftover would read
    that document's last tokens as the end of its own, and would not raise."""
    held = two_documents(BLOCK * 2 + 5, BLOCK * 3 + 2)
    held.begin_branches([0, 1, 1, 1])
    one, two = held.held[0], held.held[1]
    assert (one.remainder, two.remainder) == (5, 2)
    first, _ = held.pool.private_range(0)
    assert torch.equal(held.keys[first, :5], held.keys[one.staging_page, :5])
    for row in (1, 2, 3):
        at, _ = held.pool.private_range(row)
        assert torch.equal(held.keys[at, :2], held.keys[two.staging_page, :2])


def test_a_row_assigned_to_a_document_that_was_never_written_is_refused():
    held = two_documents(BLOCK, BLOCK)
    with pytest.raises(ValueError) as raised:
        held.begin_branches([0, 1, 2, 2])
    assert "does not hold" in str(raised.value)


def test_a_document_cannot_be_admitted_twice_under_one_handle():
    held = two_documents(BLOCK, BLOCK)
    with pytest.raises(ValueError) as raised:
        held.begin_document(0)
    assert "already in this cache" in str(raised.value)


def test_a_released_documents_pages_are_handed_to_the_next_one():
    held = two_documents(BLOCK * 2, BLOCK * 3, capacity=BLOCK * 32)
    first = held.held[0]
    held.release_document(0)
    held.begin_document(2)
    third = tokens(BLOCK * 2, start=7000.0)
    held.update(third, third * -1)
    assert held.held[2].first_page == first.first_page
    assert 0 not in held.held


def test_a_document_being_answered_cannot_be_released():
    """The failure that would not raise: the run would be handed to another document while a row's table still names it,
    and the two would read each other's tokens."""
    held = two_documents(BLOCK, BLOCK, capacity=BLOCK * 32)
    held.begin_branches([0, 0, 1, 1])
    with pytest.raises(ValueError, match="being answered"):
        held.release_document(0)
    # The one no row is answering about can go.
    held.rows_for = [1, 1, 1, 1]
    held.release_document(0)


def test_a_reset_gives_the_cursor_back():
    held = two_documents(BLOCK * 2, BLOCK * 2, capacity=BLOCK * 32)
    assert held.pool.cursor > 0
    held.reset()
    assert held.pool.cursor == 0
    assert held.pool.released == []
    assert held.held == {}
