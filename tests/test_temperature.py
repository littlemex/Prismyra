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


# --------------------------------------------------------------------------- GroupedTemperature
from prismyra.temperature import GroupedTemperature  # noqa: E402


def test_no_spec_is_empty_and_leaves_every_option_count_untouched():
    gt = GroupedTemperature(None)
    assert not gt
    assert gt.for_options(["a", "b", "c", "d"]) == 1.0
    assert gt.for_options(["yes", "no"]) == 1.0


def test_a_spec_entry_is_looked_up_by_option_count_not_by_which_options():
    gt = GroupedTemperature([{"nchoices": 4, "temperature": 0.9111, "fit_on": "unit test"}])
    assert gt
    assert gt.for_options(["a", "b", "c", "d"]) == pytest.approx(0.9111)
    assert gt.for_options(["w", "x", "y", "z"]) == pytest.approx(0.9111)  # same count, different options
    assert gt.for_options(["yes", "no"]) == 1.0  # 2-choice has no entry: untouched
    assert gt.for_options(["a", "b", "c", "d", "e"]) == 1.0  # 5-choice has no entry: untouched


def test_extra_fields_in_an_entry_are_kept_for_a_human_and_never_consumed():
    """A spec entry is allowed to carry provenance (fit_on, fit_date, validated_by, ...); only nchoices and
    temperature are read."""
    gt = GroupedTemperature(
        [{"nchoices": 2, "temperature": 1.05, "fit_on": "BoolQ", "fit_date": "2026-10-27", "notes": "irrelevant"}]
    )
    assert gt.for_options(["yes", "no"]) == pytest.approx(1.05)


def test_a_non_positive_temperature_in_a_spec_is_refused():
    with pytest.raises(PrismyraError, match="must be positive"):
        GroupedTemperature([{"nchoices": 4, "temperature": 0.0}])
    with pytest.raises(PrismyraError, match="must be positive"):
        GroupedTemperature([{"nchoices": 4, "temperature": -1.0}])


def test_a_spec_declaring_the_same_option_count_twice_is_refused():
    with pytest.raises(PrismyraError, match="twice"):
        GroupedTemperature([{"nchoices": 4, "temperature": 0.9}, {"nchoices": 4, "temperature": 1.1}])


def test_a_spec_can_be_loaded_from_a_json_file(tmp_path):
    import json

    path = tmp_path / "calibration_temperature.json"
    path.write_text(json.dumps([{"nchoices": 4, "temperature": 0.9111}]))
    gt = GroupedTemperature(path)
    assert gt.for_options(["a", "b", "c", "d"]) == pytest.approx(0.9111)


def test_the_shipped_spec_is_valid_and_matches_the_measured_candidate():
    """The spec `Prismyra` ships (`prismyra/calibration_temperature.json`) is exactly the one candidate
    docs/CALIBRATION.md's 5-fold cross-validation found to survive: 4-choice questions only."""
    import json
    from pathlib import Path

    spec_path = Path(__file__).resolve().parent.parent / "prismyra" / "calibration_temperature.json"
    gt = GroupedTemperature(json.loads(spec_path.read_text()))
    assert set(gt.by_nchoices) == {4}
    assert gt.by_nchoices[4] == pytest.approx(0.9111)
