"""`fork.widen_for_branch`'s `release` flag. No device needed: the buffer it frees is a plain dict entry on a
plain object, and freeing it is testable by identity (did a second widen get a *new* tensor, or the same one)
rather than by anything CUDA-specific.

Why this exists: a gated-delta-net layer's `_owned` buffer is sized at the engine's own `group` (every row the
engine's construction-time default allows), not at how many rows this one request actually has, and it is kept
on the layer -- across every layer this request touches -- until the whole cache is dropped at the end of the
request. That is right for a request with a second group still to come (`fork._owned`'s own docstring: the
buffer's address is what makes a recorded pass replay cheaply), and pure waste for a request that only ever has
one -- see `cache.state_bytes`'s own docstring for what one gated-delta-net layer's own copy of this actually
costs, and `interleave.read_and_branch`'s `only_group` parameter for where `release=True` comes from.
"""

from __future__ import annotations

import torch

from prismyra.fork import OWNED, widen_for_branch


class RecurrentLayer:
    """The two attributes `widen_for_branch` reads and writes -- nothing else about a real GDN layer matters
    here, since `widen_for_branch` only ever touches these two dicts and the `OWNED` attribute it manages
    itself."""

    def __init__(self):
        self.recurrent_states = {0: torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)}
        self.conv_states = {0: torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)}


def test_widen_for_branch_restores_the_one_row_tensors_either_way():
    for release in (False, True):
        layer = RecurrentLayer()
        before_recurrent = layer.recurrent_states[0].clone()
        before_conv = layer.conv_states[0].clone()
        with widen_for_branch(layer, rows=3, width=8, release=release):
            assert layer.recurrent_states[0].shape[0] == 3
            assert layer.conv_states[0].shape[0] == 3
            # What the branch rows actually see is the context's own one row, broadcast -- not zeros, not a
            # neighbour's own garbage, whichever `release` setting is under test.
            assert torch.equal(layer.recurrent_states[0], before_recurrent.expand(3, *before_recurrent.shape[1:]))
        assert torch.equal(layer.recurrent_states[0], before_recurrent)
        assert torch.equal(layer.conv_states[0], before_conv)


def test_release_false_keeps_the_owned_buffer_for_a_second_group_to_reuse():
    layer = RecurrentLayer()
    with widen_for_branch(layer, rows=3, width=8, release=False):
        first_full = getattr(layer, OWNED)[("recurrent_states", 0)]["full"]
    # The store is still there, and a second widen at the same width reuses the very same storage -- the
    # address stability `fork._owned`'s own docstring says a recorded pass depends on.
    assert ("recurrent_states", 0) in getattr(layer, OWNED)
    with widen_for_branch(layer, rows=3, width=8, release=False):
        second_full = getattr(layer, OWNED)[("recurrent_states", 0)]["full"]
    assert first_full.data_ptr() == second_full.data_ptr()


def test_release_true_drops_the_owned_buffer_so_a_later_widen_reallocates():
    layer = RecurrentLayer()
    with widen_for_branch(layer, rows=3, width=8, release=True):
        first_full = getattr(layer, OWNED)[("recurrent_states", 0)]["full"]
    # Gone from the store -- nothing is left pinning this request's full-group-width buffer alive.
    assert ("recurrent_states", 0) not in getattr(layer, OWNED)
    assert ("conv_states", 0) not in getattr(layer, OWNED)
    # A later widen on the same layer (a request this ever happens for never calls it again, but the mechanism
    # itself must not corrupt a layer that is reused) gets a brand new buffer rather than the freed one's
    # address, and still reads the layer's own current one-row state correctly.
    with widen_for_branch(layer, rows=3, width=8, release=True):
        second_full = getattr(layer, OWNED)[("recurrent_states", 0)]["full"]
        assert layer.recurrent_states[0].shape[0] == 3
    assert first_full.data_ptr() != second_full.data_ptr()


def test_release_only_drops_the_keys_this_call_actually_widened():
    """A layer with a `None` entry (a key the framework declared but never wrote) is left alone -- `widen_for_branch`
    already skips `None` values on the way in, and `release` must not touch a key it never put in `held`."""
    layer = RecurrentLayer()
    layer.recurrent_states[1] = None
    with widen_for_branch(layer, rows=3, width=8, release=True):
        pass
    assert layer.recurrent_states[1] is None
    assert not getattr(layer, OWNED)
