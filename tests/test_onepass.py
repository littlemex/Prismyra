"""What the one-pass recordings promise, checked without a device.

Whether a replay computes the same bits as the eager read is a question only a device can answer, and
`tests/test_gpu.py` asks it. What is checked here is the bookkeeping that decides what is recorded and what is not:
which projections become islands, that an island is invisible outside a recording, and which bucket serves a length.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from prismyra import onepass


class Block(nn.Module):
    """A bf16 projection, a block-quantised one, and a replaced one kept as its replacement's `inner`."""

    def __init__(self):
        super().__init__()
        self.bf16 = nn.Linear(8, 4, bias=False)
        self.fp8 = nn.Linear(8, 4, bias=False)
        self.fp8.weight = nn.Parameter(torch.zeros(4, 8).to(torch.float8_e4m3fn), requires_grad=False)
        self.wrapper = nn.Module()
        self.wrapper.inner = nn.Linear(8, 4, bias=False)


def test_an_island_is_one_function_call_outside_a_recording():
    x = torch.randn(3, 8)
    linear = nn.Linear(8, 4, bias=False)
    assert torch.equal(onepass.rows_exact(linear, x), linear(x))


def test_only_the_projections_that_choose_by_row_count_become_islands():
    """A bf16 `F.linear` is one; a block-quantised projection is not, and neither is a module nothing calls any more."""
    block = Block()
    x = torch.randn(5, 8)
    want = block.bf16(x)
    assert onepass.install_islands(block) == 1
    assert getattr(block.bf16, "_prismyra_island", False)
    assert not getattr(block.fp8, "_prismyra_island", False)
    assert not getattr(block.wrapper.inner, "_prismyra_island", False)
    # Installed, and with no recording in progress the module computes exactly what it did.
    assert torch.equal(block.bf16(x), want)
    # Idempotent: installing twice does not wrap twice.
    assert onepass.install_islands(block) == 1


def test_a_length_is_served_by_the_smallest_bucket_that_holds_it():
    held = onepass.OnePassGraphs()
    held.buckets = {64: "a", 128: "b", 256: "c"}  # type: ignore[dict-item]
    assert held.bucket_for(1) == "a"
    assert held.bucket_for(64) == "a"
    assert held.bucket_for(65) == "b"
    assert held.bucket_for(256) == "c"
    assert held.bucket_for(257) is None


def test_a_declined_bucket_leaves_its_lengths_to_the_next_one_up():
    held = onepass.OnePassGraphs()
    held.buckets = {64: "a", 256: "c"}  # type: ignore[dict-item]
    held.declined = {128: "a padded replay sat 1e-3 from the eager read"}
    assert held.bucket_for(100) == "c"


def test_every_bucket_shares_one_output_buffer_per_island():
    """The buffer is allocated at the longest length once, and each bucket reads a view of its own length."""
    islands = onepass.Islands(longest=16, width=8, dtype=torch.float32, device="cpu")
    short = islands.output(0, torch.empty(4, 3))
    long = islands.output(0, torch.empty(16, 3))
    assert short.shape == (4, 3) and long.shape == (16, 3)
    assert short.data_ptr() == long.data_ptr()
    assert len(islands.outputs) == 1


def test_an_island_whose_shape_changes_between_buckets_is_refused():
    islands = onepass.Islands(longest=16, width=8, dtype=torch.float32, device="cpu")
    islands.output(0, torch.empty(4, 3))
    with pytest.raises(ValueError):
        islands.output(0, torch.empty(4, 5))


def test_staging_refuses_an_input_larger_than_it_holds():
    islands = onepass.Islands(longest=4, width=8, dtype=torch.float32, device="cpu")
    staged = islands.stage(torch.ones(4, 8))
    assert torch.equal(staged, torch.ones(4, 8))
    with pytest.raises(ValueError):
        islands.stage(torch.ones(5, 8))


def test_the_buckets_are_increasing():
    assert list(onepass.BUCKETS) == sorted(set(onepass.BUCKETS))
