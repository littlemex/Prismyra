"""When recording a pass is cheaper than not recording it."""

from prismyra.graphs import EAGER_MS, PROVING_REPLAYS, REPLAY_MS, keeping_pays, pays_from

#: What a pass cost eagerly and replayed at each suffix width, measured on the supported model at thirty-two rows and a
#: 3,000-token context. These are the numbers the engine's decision is made from, so they are what the decision is
#: tested against -- round invented figures would agree with the arithmetic and say nothing about the mechanism.
MEASURED = {16: (109.4, 74.8), 32: (110.7, 94.7), 64: (144.2, 142.1), 128: (268.1, 267.3)}

WARMUPS = 3


def cost_in_passes(replay_ms: float = REPLAY_MS, eager_ms: float = EAGER_MS) -> float:
    r = replay_ms / eager_ms
    return (WARMUPS + 1) + PROVING_REPLAYS * r


def saving_in_passes(replay_ms: float = REPLAY_MS, eager_ms: float = EAGER_MS) -> float:
    return 1 - replay_ms / eager_ms


def test_the_threshold_is_the_smallest_count_that_actually_pays():
    """Not seven because seven was chosen -- seven because six does not cover the cost and seven does."""
    at = pays_from()
    assert at * saving_in_passes() > cost_in_passes()
    assert (at - 1) * saving_in_passes() <= cost_in_passes()


def test_the_warm_ups_are_counted():
    """The bug this function was rewritten for. Three warm-up passes and a capture pass are four passes of cost, so a
    threshold that ignores them lands at three where the truth is seven -- and three was measured losing 2.16x."""
    assert pays_from(warmups=0) < pays_from(warmups=3)
    assert pays_from(warmups=3) >= 7


def test_a_replay_that_saves_nothing_is_never_worth_recording():
    """A shape where the replay is no cheaper than the pass. The answer is a count no caller can reach, rather than a
    large number that a long enough session would eventually cross."""
    assert pays_from(replay_ms=EAGER_MS) > 1_000_000
    assert pays_from(replay_ms=EAGER_MS * 2) > 1_000_000


def test_a_free_replay_still_pays_for_the_warm_ups():
    """Even a replay that costs nothing has to earn back the four passes spent taking the recording."""
    assert pays_from(replay_ms=0.0) == WARMUPS + 2


def test_the_measured_ratios_climb_towards_one_with_the_suffix_width():
    """The fact the whole decision exists for. A recording removes the device's waiting, kernel launches are
    asynchronous, and once each kernel outlasts its launch there is no waiting left -- so the saving vanishes as the
    suffix grows, rather than staying at the 0.268 measured at one short suffix."""
    ratios = [replay / eager for eager, replay in MEASURED.values()]
    assert ratios == sorted(ratios), ratios
    assert ratios[0] < 0.7 and ratios[-1] > 0.99


def test_a_wide_suffix_is_refused_and_a_narrow_one_is_kept_at_the_same_expectation():
    """One expectation, two shapes, opposite answers -- which is exactly what a fixed threshold cannot express and is
    why two fixed thresholds were shipped and measured losing."""
    # Enough passes for the narrow shape and nowhere near enough for the wide one. Seventeen against two thousand is
    # the span a single number would have to cover, which is why there is no single number.
    narrow_eager, narrow_replay = MEASURED[16]
    expected = pays_from(narrow_replay, narrow_eager)
    assert keeping_pays(narrow_eager, narrow_replay, expected) is None
    refused = keeping_pays(*MEASURED[128], expected)
    assert refused is not None
    # The refusal carries the figures it was decided from, so a caller can check the decision rather than trust it.
    assert "267.3 ms against 268.1" in refused and "pays from" in refused


def test_the_refusal_names_how_many_passes_would_have_been_needed():
    for width, (eager, replay) in MEASURED.items():
        why = keeping_pays(eager, replay, 0)
        assert why is not None and str(pays_from(replay, eager)) in why, (width, why)
