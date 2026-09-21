"""Branch padding and width pinning. No device needed: these are index arithmetic."""

from __future__ import annotations

import pytest
import torch

from prismyra import Boolean
from prismyra.fork import WIDTHS, TooWide, build_suffixes, round_width


class Tok:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        # One id per character keeps the lengths visible in the test rather than hidden in a vocabulary.
        return {"input_ids": [ord(c) % 1000 + 1 for c in text]}


def test_widths_are_rounded_up_to_a_bucket():
    assert round_width(1) == WIDTHS[0]
    assert round_width(WIDTHS[0]) == WIDTHS[0]
    assert round_width(WIDTHS[0] + 1) == WIDTHS[1]


def test_a_question_past_the_widest_bucket_is_refused_rather_than_accommodated():
    """It used to return the exact width, which looked accommodating and wrote past the end of the cache.

    The cache is allocated for the context plus `WIDTHS[-1]`, so a wider suffix is an out-of-bounds device write -- and
    on CUDA an asynchronous one, surfacing later on an unrelated call or not at all.
    """
    with pytest.raises(TooWide, match="widest branch"):
        round_width(WIDTHS[-1] + 1)


def test_every_row_is_read_at_its_own_last_real_token():
    questions = [Boolean(id="a", prompt="short"), Boolean(id="b", prompt="a considerably longer prompt here")]
    ids, read_at, real = build_suffixes(questions, Tok(), "cpu", rows=4, width=512)
    assert ids.shape == (4, 512)
    assert int(real) == 2
    assert read_at[0] < read_at[1]  # the shorter question is read earlier in its row
    for r in range(2):
        assert ids[r, int(read_at[r])] != 0  # a real token, not padding


def test_padded_rows_repeat_a_real_question_so_they_compute_something_well_formed():
    questions = [Boolean(id="a", prompt="only one")]
    ids, read_at, real = build_suffixes(questions, Tok(), "cpu", rows=4, width=512)
    assert int(real) == 1
    for r in range(1, 4):
        assert torch.equal(ids[r], ids[0])
        assert read_at[r] == read_at[0]


def test_too_many_questions_for_the_pinned_rows_is_refused():
    questions = [Boolean(id=f"q{i}", prompt="p") for i in range(5)]
    with pytest.raises(ValueError, match="will not fit"):
        build_suffixes(questions, Tok(), "cpu", rows=4, width=512)


def test_a_question_wider_than_the_pinned_width_is_refused():
    questions = [Boolean(id="a", prompt="x" * 100)]
    with pytest.raises(ValueError, match="pinned to"):
        build_suffixes(questions, Tok(), "cpu", rows=1, width=32)
