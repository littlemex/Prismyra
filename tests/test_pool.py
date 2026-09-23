"""The arithmetic that lets one forward pass carry questions about different documents.

Every failure this can have is an index that names the wrong page, and an index that names the wrong page does not raise
-- it answers about the wrong document, fluently. So these tests assert the exact page lists rather than properties of
them.
"""

from __future__ import annotations

import pytest

from prismyra.paged import BLOCK, Full, Pool


def pool(total_pages: int = 64, rows: int = 4, branch_tokens: int = 32) -> Pool:
    return Pool(total_pages=total_pages, rows=rows, branch_tokens=branch_tokens)


def test_a_rows_private_pages_do_not_move_when_the_document_changes():
    """The property the whole design rests on. Private pages are sized for the worst remainder rather than for the
    document's own, so a recording's addresses stay valid whatever is admitted next."""
    a, b = pool(total_pages=256), pool(total_pages=256)
    a.admit(160)
    b.admit(3045)
    assert a.private_range(0) == b.private_range(0)
    assert a.private_range(3) == b.private_range(3)


def test_two_documents_get_disjoint_pages_and_each_row_names_only_its_own():
    p = pool(rows=4)
    first = p.admit(2 * BLOCK)
    second = p.admit(3 * BLOCK)
    assert first.pages() == [0, 1]
    assert second.pages() == [2, 3, 4]

    p.assign([2, 2])
    table = p.table()
    assert p.assignment == [0, 0, 1, 1]
    # Row 0 and row 1 read the first document; rows 2 and 3 read the second. No row names another document's pages.
    assert table[0][:2] == [0, 1]
    assert table[2][:3] == [2, 3, 4]
    for row in (0, 1):
        assert set(table[row]) & set(second.pages()) == set()
    for row in (2, 3):
        assert set(table[row]) & set(first.pages()) == set()


def test_every_rows_table_is_its_document_then_its_own_pages():
    p = pool(rows=3)
    p.admit(2 * BLOCK)
    p.admit(1 * BLOCK)
    p.assign([1, 2])
    table = p.table()
    for row, document in enumerate(p.assignment):
        named = p.documents[document].pages() + list(range(*p.private_range(row)))
        assert table[row][: len(named)] == named
        # Whatever follows is padding, and the kernel never reaches it because `lengths` stops first.
        assert len(table[row]) == p.width()


def test_the_table_is_rectangular_across_documents_of_different_lengths():
    """Different documents, one shape. A ragged table would be a different recording per batch, which is the thing
    being avoided."""
    p = pool(rows=4)
    p.admit(1 * BLOCK)
    p.admit(5 * BLOCK)
    p.assign([2, 2])
    widths = {len(row) for row in p.table()}
    assert widths == {p.width()}


def test_each_row_reports_its_own_documents_length():
    p = pool(rows=3)
    p.admit(100)
    p.admit(48)
    p.assign([1, 2])
    assert p.lengths() == [100, 48, 48]
    assert p.lengths(branch_progress=7) == [107, 55, 55]


def test_a_rows_own_tokens_start_after_its_own_documents_remainder():
    """Two documents whose lengths differ modulo the page size. The offset is per row and taking one document's for
    both is the failure mode this exists to prevent."""
    p = pool(rows=2)
    p.admit(2 * BLOCK + 5)
    p.admit(3 * BLOCK)
    p.assign([1, 1])
    assert p.branch_offset(0) == 5
    assert p.branch_offset(1) == 0


def test_the_pool_refuses_before_it_writes_anything():
    p = pool(total_pages=32, rows=4, branch_tokens=32)
    free = p.free_pages
    with pytest.raises(Full) as raised:
        p.admit((free + 1) * BLOCK)
    assert "are free" in str(raised.value)
    # Refused means unchanged: a partial write across ten layers has no rollback.
    assert p.documents == []
    assert p.free_pages == free


def test_a_partial_last_page_is_not_shared():
    """A page half document and half nothing cannot sit in the middle of a row's run, so only whole pages are shared and
    the remainder is copied into each row."""
    p = pool()
    held = p.admit(3 * BLOCK + 7)
    assert held.whole_pages == 3
    assert held.remainder == 7
    assert held.pages() == [0, 1, 2]
    # A fourth page is reserved to stage the leftover, so the next document cannot be written over it -- which is what
    # would happen if only the whole pages were reserved.
    assert held.staging_page == 3
    assert held.reserved_pages == 4
    assert p.admit(BLOCK).first_page == 4


def test_documents_and_rows_cannot_exceed_what_the_pool_holds():
    p = pool(rows=2)
    p.admit(BLOCK)
    p.admit(BLOCK)
    with pytest.raises(ValueError):
        p.assign([2, 2])
    with pytest.raises(ValueError):
        p.assign([1, 1, 1])


def test_a_document_on_a_page_boundary_stages_nothing_and_says_so():
    """Asking where the leftover is staged when there is none is a bug in the caller, so it raises rather than returning
    a page that belongs to whatever comes next."""
    p = pool()
    held = p.admit(2 * BLOCK)
    assert held.reserved_pages == 2
    with pytest.raises(ValueError):
        _ = held.staging_page


def test_the_staging_page_is_never_inside_another_documents_run():
    """The bug this reservation exists for. With only whole pages reserved, the first document's leftover would land on
    the second document's opening page, and rows would read those tokens as the end of their own document."""
    p = pool(total_pages=256)
    first = p.admit(3 * BLOCK + 7)
    second = p.admit(4 * BLOCK)
    assert first.staging_page not in second.pages()
    assert set(first.pages()).isdisjoint(second.pages())


def test_a_released_run_is_handed_out_again():
    """Without reuse the cursor only moves forward, so the pool refuses a document while holding released pages."""
    p = pool(total_pages=64, rows=4, branch_tokens=32)
    room = p.free_pages
    first = p.admit(BLOCK * 2)
    p.release(first)
    again = p.admit(BLOCK * 2)
    assert again.first_page == first.first_page
    assert p.free_pages == room - 2, "reuse should not consume more of the cursor"


def test_releasing_the_newest_document_gives_its_pages_back_to_the_cursor():
    """A run that reaches the cursor is not a run, it is unallocated space, and a pool that empties should be as good as
    a fresh one."""
    p = pool(total_pages=64)
    room = p.free_pages
    held = [p.admit(BLOCK * 2) for _ in range(3)]
    assert p.free_pages == room - 6
    for one in reversed(held):
        p.release(one)
    assert p.free_pages == room, "an emptied pool kept pages it no longer owes anyone"
    assert p.released == []


def test_released_runs_that_touch_are_joined():
    """Otherwise a pool that has held and released many documents cannot admit a large one while holding twice its pages
    in small runs."""
    p = pool(total_pages=64)
    a = p.admit(BLOCK)
    b = p.admit(BLOCK)
    c = p.admit(BLOCK)
    keep = p.admit(BLOCK * 8)  # so releasing a, b and c does not simply rewind the cursor
    p.release(a)
    p.release(c)
    p.release(b)
    assert p.released == [(a.first_page, 3)], p.released
    # And the joined run can hold a document none of the three could.
    wide = p.admit(BLOCK * 3)
    assert wide.first_page == a.first_page
    assert keep.first_page not in wide.pages()


def test_a_document_reusing_a_larger_run_gives_the_whole_run_back():
    """Otherwise the run shrinks to the size of whatever last used it, and a pool degrades every time it is reused."""
    p = pool(total_pages=64)
    big = p.admit(BLOCK * 6)
    keep = p.admit(BLOCK)
    p.release(big)
    assert p.released == [(big.first_page, 6)]
    small = p.admit(BLOCK)
    assert small.first_page == big.first_page
    p.release(small)
    assert p.released == [(big.first_page, 6)], "the run shrank to the size of the document that reused it"
    assert keep.reserved_pages == 1


def test_a_refusal_names_the_largest_released_run():
    """A caller refused while the pool holds released pages needs to know whether any of them could have helped."""
    p = pool(total_pages=32, rows=4, branch_tokens=32)
    small = p.admit(BLOCK)
    p.admit(BLOCK)
    p.release(small)
    with pytest.raises(Full) as raised:
        p.admit(BLOCK * (p.free_pages + 2))
    assert "largest released run is 1" in str(raised.value), str(raised.value)


def test_reuse_does_not_hand_the_same_run_to_two_documents():
    """The failure that would not raise. Two documents on one run would read each other's tokens."""
    p = pool(total_pages=64)
    one = p.admit(BLOCK * 2)
    p.release(one)
    a = p.admit(BLOCK)
    b = p.admit(BLOCK)
    assert a.first_page != b.first_page
    assert set(a.pages()).isdisjoint(b.pages())
