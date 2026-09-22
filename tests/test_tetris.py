"""The game the Tetris example plays, checked without a model.

Here because the comparison it exists for is worthless if the rules are wrong: an agent that looks brilliant because
rows clear when they should not, or hopeless because a legal placement is refused, would be measured precisely and mean
nothing. These are the properties the agents rely on, not a full specification of Tetris.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "tetris"))

from game import HEIGHT, PIECES, WIDTH, Board, Move, apply, bag, cells, landing_row, legal_moves


def filled(rows: list[str]) -> Board:
    """A board from pictures, bottom-aligned, so a test reads as the shape it is about."""
    board = Board()
    for i, row in enumerate(reversed(rows)):
        assert len(row) == WIDTH
        for x, char in enumerate(row):
            board.rows[HEIGHT - 1 - i][x] = char == "#"
    return board


def test_every_rotation_of_every_piece_has_four_cells():
    """Four, or it is not a tetromino, and a typo in the table would otherwise show up as an agent playing badly."""
    for name, rotations in PIECES.items():
        for rotation, shape in enumerate(rotations):
            assert len(set(shape)) == 4, f"{name} rotation {rotation}"


def test_rotations_are_distinct_so_a_choice_between_them_does_something():
    """The symmetric pieces have fewer rotations on purpose. If O had four, three of an agent's options would be the
    same move and its measured decisions would be spread across duplicates."""
    for name, rotations in PIECES.items():
        seen = {tuple(sorted(shape)) for shape in rotations}
        assert len(seen) == len(rotations), f"{name} has a duplicate rotation"
    assert len(PIECES["O"]) == 1
    assert len(PIECES["I"]) == 2
    assert len(PIECES["T"]) == 4


def test_a_piece_is_never_offered_a_column_it_hangs_off():
    for name in PIECES:
        for move in legal_moves(name):
            assert all(0 <= x < WIDTH for _, x in cells(name, move)), f"{name} {move}"


def test_a_piece_falls_to_the_floor_on_an_empty_board():
    board = Board()
    drop = landing_row(board, "O", Move(0, 0))
    assert drop == HEIGHT - 2  # the O is two rows deep, so its top row sits one above the floor


def test_a_piece_lands_on_top_of_what_is_already_there():
    board = filled(["##........"])
    after, cleared = apply(board, "O", Move(0, 0))
    assert cleared == 0
    assert after.heights()[0] == 3
    assert after.heights()[2] == 0


def test_a_full_row_clears_and_what_was_above_it_comes_down():
    board = filled([".........#", ".#########"])
    after, cleared = apply(board, "I", Move(1, 0))
    # A vertical I in column 0 reaches the floor, completing the bottom row, which vanishes. The lone cell that was in
    # column 9 one row up descends into the row the clear vacated, and three of the I's four cells remain.
    assert cleared == 1
    assert after.heights()[9] == 1
    assert after.heights()[0] == 3
    assert after.holes() == 0


def test_a_piece_cannot_slide_under_an_overhang():
    """The simplification the whole comparison rests on, so it is asserted rather than described.

    Column 0's bottom cell is empty and the cell above it is filled. A real player tucks a piece in there; here a piece
    dropped into column 0 lands on the overhang, and the gap stays a hole. Every agent faces that same restriction,
    which is what makes their choices comparable.
    """
    board = filled(["#........#", ".#########"])
    after, cleared = apply(board, "I", Move(1, 0))
    assert cleared == 0
    assert not after.rows[HEIGHT - 1][0], "the bottom cell of column 0 should still be empty"
    assert after.holes() >= 1


def test_clearing_every_row_empties_the_board():
    board = filled(["#########." for _ in range(4)])
    after, cleared = apply(board, "I", Move(1, 9))
    assert cleared == 4
    assert after.holes() == 0
    assert sum(after.heights()) == 0


def test_a_placement_that_does_not_fit_is_refused_rather_than_stacked_outside_the_board():
    board = Board()
    for y in range(HEIGHT):
        board.rows[y][0] = True
    assert apply(board, "I", Move(1, 0)) is None


def test_a_hole_is_an_empty_cell_under_a_filled_one():
    assert filled([".........."]).holes() == 0
    assert filled(["#........."]).holes() == 0
    # Filled cells stacked with nothing under them are not holes, however tall the stack.
    assert filled(["#.........", "#.........", "#........."]).holes() == 0
    # A ledge with a gap beneath it: one hole, and the thing every Tetris heuristic is mostly counting.
    board = Board()
    board.rows[HEIGHT - 2][3] = True
    assert board.holes() == 1


def test_the_bag_gives_every_piece_once_per_seven_and_repeats_for_a_seed():
    drawn = bag(3, 14)
    assert sorted(drawn[:7]) == sorted(PIECES)
    assert sorted(drawn[7:]) == sorted(PIECES)
    assert bag(3, 14) == drawn
    assert bag(4, 14) != drawn


def test_the_render_labels_columns_so_a_move_naming_one_can_be_checked():
    text = filled(["##........"]).render()
    assert "0123456789" in text
    assert text.count("#") == 2


@pytest.mark.parametrize("piece", list(PIECES))
def test_every_piece_has_at_least_one_playable_move_on_an_empty_board(piece):
    assert any(apply(Board(), piece, move) is not None for move in legal_moves(piece))
