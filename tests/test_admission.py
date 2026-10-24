"""What admission promises, checked without a device.

These are arithmetic and bookkeeping tests. Whether a pass actually fits is a question only an allocator can answer,
and `Prismyra` catches that separately; what is checked here is that the engine budgets the answering transient at all,
since budgeting only the held cache is what let a context be admitted and then run out of memory on its first question.
"""

from __future__ import annotations

import time

import pytest
import torch

from prismyra import chunking
from prismyra.cache import join_bytes_per_token
from prismyra.engine import ANSWERING_MARGIN, RECLAIM_COOLDOWN_S, WIDTHS, Prismyra, PrismyraError, row_constant


class Config:
    """The two shapes the arithmetic part of the budget reads, as the supported model declares them."""

    num_key_value_heads = 2
    head_dim = 256
    hidden_size = 2048
    num_attention_heads = 16


class Fake:
    """A bare engine with only the fields these methods read. Constructing a real one loads 35 GiB of weights."""

    group = 32
    dtype = torch.bfloat16
    config = Config()
    _observed_row_constant: int | None = None
    _observed_at_rows = 0
    #: The borrowed kernel reads the join in the layout it is stored in; the framework's fallback is handed a strided
    #: view and is budgeted at twice the slope because nothing has measured whether it copies.
    _borrowed_kernel = True

    answering_bytes = Prismyra.answering_bytes
    budget_is_evidenced = Prismyra.budget_is_evidenced
    _observe_peak = Prismyra._observe_peak


PER_TOKEN = join_bytes_per_token(Config(), torch.bfloat16)


def test_nothing_is_budgeted_before_a_pass_has_been_observed():
    """Zero rather than a guess. The constant part is one model's activations and kernel workspace, and the refusal
    message says outright that the first pass is unbudgeted."""
    assert Fake().answering_bytes(10_000) == 0


def test_the_slope_is_arithmetic_and_halved_by_token_major_storage():
    """2,048 bytes per context token per row from these shapes: keys and values, and two layers' joins overlapping.

    It was 4,096 while the join was head-major and the kernel's `transpose(1, 2).reshape(...)` had to copy the whole
    thing a second time to reach the layout it reads. The measured slope was 3,838 against that 4,096, so the
    arithmetic over-stated by 7%; with the copy gone the prediction halves and `prismyra-bench widths` is what says
    whether the measurement followed it.
    """
    assert PER_TOKEN == 2_048


def test_the_budget_grows_with_the_rows_a_request_will_use():
    engine = Fake()
    engine._observed_row_constant = 1_000
    per_row = 1_000 + PER_TOKEN * (1_000 + WIDTHS[-1])
    assert engine.answering_bytes(1_000, questions=1) == int(per_row * ANSWERING_MARGIN)
    assert engine.answering_bytes(1_000, questions=4) == int(4 * per_row * ANSWERING_MARGIN)
    # Capped by the group: a request with more questions than the group is answered in several passes, and what has to
    # fit is one pass.
    assert engine.answering_bytes(1_000, questions=1_000) == int(32 * per_row * ANSWERING_MARGIN)


def test_a_longer_context_is_budgeted_higher_by_the_arithmetic_slope():
    engine = Fake()
    engine._observed_row_constant = 1_000
    short = engine.answering_bytes(3_000, questions=1)
    longer = engine.answering_bytes(30_000, questions=1)
    assert longer - short == pytest.approx(27_000 * PER_TOKEN * ANSWERING_MARGIN, rel=1e-6)


def test_nothing_is_observed_without_a_device():
    engine = Fake()
    engine._observe_peak(None, 5_000, 4)
    assert engine._observed_row_constant is None


def test_the_observed_constant_is_the_peak_with_the_scaling_part_taken_out():
    assert row_constant(None, 1_000 + 4_096 * 10, 10, 4_096) == 1_000


def test_the_constant_ratchets_upwards_and_is_comparable_across_context_lengths():
    """The whole reason for subtracting the slope. Two passes at very different lengths yield the same constant, so an
    observation at either is usable for the other -- which is what stops the estimate drifting towards refusal.

    Keeping the largest observed *prediction* instead would ratchet the other way: a prediction scaled up from a short
    context is larger per token than one from a long context, so the shortest context ever seen would win and be kept,
    and a full-width pass at 24,327 tokens would be budgeted at 17.7 GiB when it needs 4.97.
    """
    short = row_constant(None, 1_000 + 4_096 * 3_552, 3_552, 4_096)
    both = row_constant(short, 1_000 + 4_096 * 24_839, 24_839, 4_096)
    assert short == both == 1_000
    assert row_constant(both, 500, 10, 4_096) == 1_000


def test_a_peak_below_the_arithmetic_slope_floors_at_zero():
    """A measured peak smaller than the slope alone means the layers never overlapped the way the slope assumes. A
    negative constant would budget less than the arithmetic, which is the one part that is not in doubt."""
    assert row_constant(None, 100, 10, 4_096) == 0


def test_a_pass_wider_than_any_measured_is_not_claimed_to_be_budgeted():
    """The estimate exists for every width; the claim that it is evidence does not.

    Per-row cost used not to be flat in width -- 0.111 GiB per row at width 1 against 0.155 at widths 8 and 32 -- so an
    observation at width 1 under-stated a width-32 pass by 29%. Storing the join in the layout the kernel reads removed
    the second copy that caused it, and the two now agree to three decimals. This stays true anyway: one measurement on
    one model is not a reason to promise the shape holds, and refusing on a figure nothing supports is the failure this
    guards against.
    """
    engine = Fake()
    engine._observed_row_constant = 1_000
    engine._observed_at_rows = 8
    assert engine.budget_is_evidenced(questions=4)
    assert engine.budget_is_evidenced(questions=8)
    assert not engine.budget_is_evidenced(questions=9)
    assert not engine.budget_is_evidenced()
    # The estimate is still produced, because one that already exceeds free memory is a refusal either way.
    assert engine.answering_bytes(1_000, questions=32) > 0


def test_nothing_is_evidenced_before_the_first_pass():
    assert not Fake().budget_is_evidenced(questions=1)


def test_the_fallback_path_is_budgeted_at_twice_the_slope():
    """Not because it is known to copy, but because it is not known not to.

    The join is stored in the layout the borrowed kernel reads. The framework's own attention takes the same storage as
    a strided view, and whether it works on those strides or makes itself a contiguous copy is inside the framework. A
    path nobody has measured is budgeted at the larger figure.
    """
    assert join_bytes_per_token(Config(), torch.bfloat16, doubled=True) == 2 * PER_TOKEN

    fast, slow = Fake(), Fake()
    fast._observed_row_constant = slow._observed_row_constant = 0
    slow._borrowed_kernel = False
    assert slow.answering_bytes(10_000, questions=8) == 2 * fast.answering_bytes(10_000, questions=8)


def test_visible_free_is_mem_get_info_unchanged_without_a_fraction_set(monkeypatch):
    """The common case, checked first so a bug in the correction cannot hide behind it: a process nobody has capped
    is the whole point of `get_per_process_memory_fraction` defaulting to `1.0`, and that default has to make the
    `min` below a no-op rather than something that happens to compute the same answer."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (5 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda index: 1.0)
    free, total = chunking.visible_free(torch.device("cuda", 0))
    assert (free, total) == (5 * 2**30, 32 * 2**30)


def test_visible_free_corrects_for_a_process_capped_tighter_than_the_device(monkeypatch):
    """`mem_get_info` answers for the device: on a 32 GiB card given a 0.5 fraction and already holding 12 GiB, it
    still reports whatever the device's own free figure is -- found, on real hardware, to be most of the 32 GiB,
    tens of gigabytes `should_chunk` and `Prismyra._check_fits` had no way to tell from room this process could
    actually use. The correction is `cap - reserved`, not `cap - allocated`: the allocator's own idle pool between
    the two counts against the cap just as much as a live tensor does."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda index: 0.5)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 12 * 2**30)
    free, total = chunking.visible_free(torch.device("cuda", 0))
    assert total == 16 * 2**30
    assert free == 4 * 2**30  # cap (16 GiB) - reserved (12 GiB), not the device's 20 GiB of raw free


def test_visible_free_does_not_go_negative_once_the_cap_is_already_spent(monkeypatch):
    """A process already reserved past where a *newly lowered* cap would put it (the cap is set once at process
    start in practice, but nothing stops a caller from lowering it later) has no headroom, not negative headroom --
    the raw figure, however large, is capped at zero rather than passed through unmodified or allowed to subtract
    below it."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda index: 0.5)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 18 * 2**30)
    free, _total = chunking.visible_free(torch.device("cuda", 0))
    assert free == 0


def test_visible_free_resolves_an_indexless_device_to_the_current_one(monkeypatch):
    """`get_per_process_memory_fraction` raises on `torch.device('cuda')` (no index); `mem_get_info` does not. A
    device named the way `Prismyra.torch_device` names it must still reach the fraction call, through whichever
    device CUDA already made current."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    seen_index = {}

    def fraction(index):
        seen_index["value"] = index
        return 1.0

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 2))
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", fraction)
    chunking.visible_free(torch.device("cuda"))
    assert seen_index["value"] == 0


class FittingFake:
    """A bare engine with only what `_check_fits` reads directly, for exercising its reclaim-and-recheck branch --
    added after a real 32 GiB card measurement (`_check_fits`'s own comment on the fix) -- without the arithmetic
    `cache_bytes`/`answering_bytes`/`reading_bytes` the tests above already cover. Fixed costs, chosen by the
    test, stand in for a model shape no assertion here needs.
    """

    group = 32
    torch_device = torch.device("cuda", 0)
    _one_pass = None

    def __init__(self, held: int, answering: int, reading: int):
        self._held, self._answering, self._reading = held, answering, reading
        self._reclaim_cooldown_until = 0.0
        self._empty_cache_calls = 0
        self._empty_cache_ms = 0.0

    def cache_bytes(self, context_tokens: int) -> int:
        return self._held

    def answering_bytes(self, context_tokens: int) -> int:
        return self._answering

    def reading_bytes(self, context_tokens: int) -> int:
        return self._reading

    def budget_is_evidenced(self, questions: int | None = None) -> bool:
        return True

    _check_fits = Prismyra._check_fits


def _two_valued(monkeypatch, name, before, after):
    """Patch `torch.cuda.<name>` to answer `before` the first time and `after` every time afterwards -- the shape
    every test below needs: one number for the state admission finds before a reclaim, a second for the state
    `empty_cache()` is meant to have left behind."""
    calls = {"n": 0}

    def fn(*_a, **_k):
        calls["n"] += 1
        return before if calls["n"] == 1 else after

    monkeypatch.setattr(torch.cuda, name, fn)
    return calls


def test_check_fits_reclaims_and_admits_when_the_reclaim_recovers_enough(monkeypatch):
    """The bug this branch fixes: a context refused on the raw `mem_get_info`/`spare` prediction can still fit once
    `empty_cache()` actually returns the allocator's idle pool to the driver. Free starts 1 GiB under the 8 GiB
    wanted and `empty_cache()` is simulated to free the 7.5 GiB a real 32 GiB card measurement found stuck in
    `reserved`, which is more than enough; admission must retry its own check with the post-`empty_cache` numbers
    rather than the ones that triggered it, and must not raise."""
    fake = FittingFake(held=3 * 2**30, answering=0, reading=5 * 2**30)  # wanted = 8 GiB
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda index: 1.0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    _two_valued(monkeypatch, "mem_get_info", before=(1 * 2**30, 32 * 2**30), after=(8 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)

    fake._check_fits(63_207)  # does not raise

    assert fake._empty_cache_calls == 1


def test_check_fits_still_refuses_when_the_reclaim_does_not_help(monkeypatch):
    """A genuinely tight device: `empty_cache()` is tried once (it is cheap next to refusing work that would have
    OOM'd for real), but the free figure does not move, so admission still refuses -- and refuses using the
    post-reclaim numbers, not stale ones, since a caller reading the message should see what is true right now."""
    fake = FittingFake(held=3 * 2**30, answering=0, reading=5 * 2**30)  # wanted = 8 GiB
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda index: 1.0)
    calls = {"n": 0}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)

    with pytest.raises(PrismyraError, match=r"3\.0 GiB of 32\.0 GiB is available"):
        fake._check_fits(63_207)

    assert calls["n"] == 1, "a genuinely tight device is still worth one confirm -- the cheap direction to be wrong in"
    assert fake._empty_cache_calls == 1
    assert fake._reclaim_cooldown_until > time.monotonic(), "a confirmed-tight reclaim should start the cooldown"


def test_check_fits_does_not_retry_the_reclaim_within_the_cooldown(monkeypatch):
    """`RECLAIM_COOLDOWN_S` is shared with `_run_recorded`'s own margin check for the same reason it exists there:
    the `empty_cache()` call is a real CUDA synchronisation, and a tight stretch must not pay for it on every
    single admission while free memory stays provably stuck at the same low number."""
    fake = FittingFake(held=3 * 2**30, answering=0, reading=5 * 2**30)
    fake._reclaim_cooldown_until = time.monotonic() + RECLAIM_COOLDOWN_S
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda index: 1.0)
    calls = {"n": 0}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)

    with pytest.raises(PrismyraError):
        fake._check_fits(63_207)

    assert calls["n"] == 0, "the cooldown from an earlier confirmed-tight reclaim was not honoured"


def test_check_fits_admits_outright_when_there_is_already_enough_free(monkeypatch):
    """The common case this branch must not slow down or touch: when the first, unconfirmed prediction already
    shows enough room, `empty_cache()` is never called at all."""
    fake = FittingFake(held=3 * 2**30, answering=0, reading=5 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda index: 1.0)
    calls = {"n": 0}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (9 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)

    fake._check_fits(63_207)  # does not raise

    assert calls["n"] == 0
    assert fake._empty_cache_calls == 0
