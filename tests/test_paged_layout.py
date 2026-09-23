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
    layout = held.layout
    assert layout.remainder == 5
    for row in range(layout.rows):
        first, _ = layout.private_range(row)
        assert torch.equal(held.keys[first, :5], held.keys[layout.shared_pages, :5])


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
