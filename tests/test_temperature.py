"""What a temperature promises, and the one thing it must never do.

The promise is honest numbers; the prohibition is changing a decision. Both are checkable without a device, which is why
they are here rather than in a benchmark.
"""

from __future__ import annotations

import math

import pytest

from prismyra.schema import Answer, PrismyraError
from prismyra.temperature import (
    Temperature,
    brier,
    cross_fit,
    expected_calibration_error,
    negative_log_likelihood,
    scaled,
)


def overconfident(n: int = 200) -> tuple[list[list[float]], list[int]]:
    """Answers that claim 0.95 and are right 0.75 of the time, which is the shape a read-out actually has.

    Deterministic rather than sampled: a test whose failure depends on a seed is a test that will fail for someone else
    on a different platform's random number generator.
    """
    probabilities, correct = [], []
    for i in range(n):
        probabilities.append([0.95, 0.05])
        # Paired, so that splitting on even and odd indices gives both folds the same rate of wrong answers. Keying the
        # label on `i % 4` instead put every wrong answer in one fold, and the two folds then fitted temperatures 0.6
        # apart -- a failure of the test's data rather than of the fit, and worth the comment because it looked like the
        # latter.
        correct.append(1 if (i // 2) % 4 == 0 else 0)
    return probabilities, correct


def test_scaling_cannot_change_which_option_wins():
    """The property that lets this be applied to an answer already returned. Dividing every score by the same positive
    number cannot reorder them, so accuracy is untouched and only the number attached to the decision moves."""
    p = [0.5, 0.3, 0.15, 0.05]
    for temperature in (0.2, 0.5, 1.0, 2.0, 4.0):
        got = scaled(p, temperature)
        assert max(range(len(got)), key=got.__getitem__) == 0
        assert math.isclose(sum(got), 1.0, rel_tol=1e-9)


def test_a_temperature_above_one_flattens_and_below_one_sharpens():
    p = [0.9, 0.1]
    assert scaled(p, 2.0)[0] < p[0]
    assert scaled(p, 0.5)[0] > p[0]
    assert scaled(p, 1.0) == pytest.approx(p, rel=1e-9)


def test_a_non_positive_temperature_is_refused():
    for bad in (0.0, -1.0):
        with pytest.raises(PrismyraError, match="must be positive"):
            scaled([0.5, 0.5], bad)


def test_fitting_finds_a_temperature_above_one_for_overconfident_answers():
    """The direction is the whole test. Answers claiming more than they deliver need flattening, and a fit coming back
    below one would be making them worse while reporting an improvement."""
    probabilities, correct = overconfident()
    fitted = Temperature.fit(probabilities, correct)
    assert fitted.value > 1.0
    assert fitted.after["expected_calibration_error"] < fitted.before["expected_calibration_error"]
    # Accuracy is identical, necessarily. Reported so that an improvement cannot be mistaken for a better answer.
    assert fitted.after["accuracy"] == fitted.before["accuracy"]


def test_the_fit_reports_both_sides_so_an_improvement_cannot_be_quoted_alone():
    fitted = Temperature.fit(*overconfident())
    for side in (fitted.before, fitted.after):
        assert set(side) == {"accuracy", "negative_log_likelihood", "expected_calibration_error", "brier"}


def test_calibration_error_is_zero_when_confidence_matches_the_rate():
    """Eight answers at 0.75 confidence with six right. Nothing to correct, and a measure that reported otherwise would
    make every fit look useful."""
    probabilities = [[0.75, 0.25]] * 8
    correct = [0, 0, 0, 0, 0, 0, 1, 1]
    assert expected_calibration_error(probabilities, correct) == pytest.approx(0.0, abs=1e-12)


def test_brier_rewards_the_truth_and_punishes_confidence_in_the_wrong_answer():
    assert brier([[1.0, 0.0]], [0]) == pytest.approx(0.0)
    assert brier([[0.0, 1.0]], [0]) == pytest.approx(2.0)
    assert brier([[0.5, 0.5]], [0]) == pytest.approx(0.5)


def test_the_log_score_is_unbounded_where_brier_is_not():
    """Why both are reported. One confident mistake can dominate a log score and cannot dominate a Brier score, so a
    caller comparing two methods needs to know which one they are reading."""
    assert negative_log_likelihood([[1e-30, 1.0]], [0]) > 60
    assert brier([[1e-30, 1.0]], [0]) <= 2.0


def test_cross_fitting_scales_every_answer_with_a_temperature_it_did_not_help_choose():
    probabilities, correct = overconfident()
    out, temperatures = cross_fit(probabilities, correct, folds=2)
    assert len(out) == len(correct)
    assert len(temperatures) == 2
    # Folds agreeing is the signal that one temperature is supportable. Reported for exactly that reason.
    assert abs(temperatures[0] - temperatures[1]) < 0.6


def test_cross_fitting_refuses_what_it_cannot_split():
    with pytest.raises(PrismyraError, match="at least two folds"):
        cross_fit([[0.5, 0.5]], [0], folds=1)
    with pytest.raises(PrismyraError, match="cannot be split"):
        cross_fit([[0.5, 0.5]], [0], folds=2)


def test_mismatched_lengths_are_refused_rather_than_zipped_short():
    """A silent zip is how an off-by-one becomes a calibration figure."""
    for call in (negative_log_likelihood, expected_calibration_error, brier):
        with pytest.raises(PrismyraError, match="same non-zero length"):
            call([[0.5, 0.5], [0.5, 0.5]], [0])


def test_rescoring_an_answer_keeps_its_decision_and_changes_its_numbers():
    answer = Answer(id="q", kind="boolean", value=True, option="yes", probabilities={"yes": 0.95, "no": 0.05})
    rescored = Temperature(value=2.0).rescore(answer)
    assert rescored.option == answer.option
    assert rescored.value is answer.value
    assert rescored.probabilities["yes"] < answer.probabilities["yes"]
    assert sum(rescored.probabilities.values()) == pytest.approx(1.0)
