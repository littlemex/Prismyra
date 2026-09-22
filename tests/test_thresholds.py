"""Choosing the decision point per question. No device needed: the fitting takes answers, not a model.

That is the property worth testing as well as asserting. If these tests needed a GPU, the mechanism would not be the
one described -- something a caller can fit offline from what it already logged.
"""

from __future__ import annotations

import pytest

from prismyra.schema import Answer, PrismyraError, Result, Timing
from prismyra.thresholds import DEFAULT_CUT, ThresholdError, Thresholds


def answer(probability: float, question_id: str = "flag") -> Answer:
    """One boolean answer at a given probability of yes, as the read-out would have produced it."""
    return Answer(
        id=question_id,
        kind="boolean",
        value=probability >= DEFAULT_CUT,
        option="yes" if probability >= DEFAULT_CUT else "no",
        probabilities={"yes": probability, "no": 1.0 - probability},
    )


def history(pairs: list[tuple[float, bool]], question_id: str = "flag"):
    """One result per observation, which is the shape a caller's log has."""
    return [
        (Result(answers={question_id: answer(p, question_id)}, timing=Timing(), model="m"), {question_id: label})
        for p, label in pairs
    ]


def test_a_rare_class_moves_the_point_up():
    """The measured failure this exists for: the read-out says yes to nearly everything.

    Ninety observations where yes is rare and only the highest probabilities are true. The argmax at 0.5 would call
    everything above 0.5 a yes and be wrong most of the time; the fitted point should sit above them.
    """
    pairs = [(0.55 + i * 0.001, False) for i in range(80)] + [(0.95 + i * 0.005, True) for i in range(10)]
    cuts, report = Thresholds.fit(history(pairs))

    # The cut has to separate the two classes. Where exactly it sits between them is the midpoint rule's business, and
    # asserting a particular value here would be asserting that rule twice.
    highest_no = max(p for p, label in pairs if not label)
    lowest_yes = min(p for p, label in pairs if label)
    assert highest_no < cuts.cuts["flag"] <= lowest_yes
    assert cuts.support["flag"] == (10, 80)
    assert f"{cuts.cuts['flag']:.3f}" in report.explain()
    assert "10 yes, 80 no" in report.explain()


def test_a_cut_fitted_on_too_little_is_refused():
    """A point chosen from four examples is a point chosen from noise, and it would then apply to everything."""
    cuts, report = Thresholds.fit(history([(0.9, True), (0.8, True), (0.2, False), (0.1, False)]))
    assert "flag" not in cuts.cuts
    assert cuts.support["flag"] == (2, 2)
    assert "fewer than 20 answers" in report.explain()


def test_a_cut_needs_positives_not_merely_rows():
    """Twenty rows at a 1.5% base rate usually hold one positive, and a point fitted on one positive is noise.

    An earlier version counted rows alone, which let exactly that through while the docstring claimed it did not.
    """
    pairs = [(0.3, False)] * 58 + [(0.9, True)] * 2
    cuts, report = Thresholds.fit(history(pairs))
    assert "flag" not in cuts.cuts
    assert "fewer than 5 positive labels" in report.explain()


def test_a_cut_sits_between_observations_rather_than_on_one():
    """A cut equal to a training probability flips on a value a hair below it. The midpoint does not."""
    pairs = [(0.40, False)] * 30 + [(0.60, True)] * 10
    cuts, _ = Thresholds.fit(history(pairs))
    assert 0.40 < cuts.cuts["flag"] < 0.60


def test_a_label_that_is_neither_yes_nor_no_is_refused():
    """Reading `"unfair"` as no would shift every cut fitted afterwards, and nothing in the output would say so."""
    result = Result(answers={"flag": answer(0.9)}, timing=Timing(), model="m")
    with pytest.raises(ThresholdError, match="neither yes nor no"):
        Thresholds.fit([(result, {"flag": "unfair"})] * 40)


def test_cuts_from_another_model_are_refused():
    """The likely operational mistake, and silent without a check: the probabilities still look like probabilities."""
    cuts, _ = Thresholds.fit(history([(0.6, False)] * 40 + [(0.99, True)] * 10))
    elsewhere = Result(answers={"flag": answer(0.7)}, timing=Timing(), model="a different model")
    with pytest.raises(ThresholdError, match="fitted on probabilities from"):
        cuts.decide(elsewhere)
    assert cuts.decide(elsewhere, strict=False)["flag"].value is False


def test_the_provenance_tag_is_not_added_when_nothing_moved():
    cuts, _ = Thresholds.fit(history([(0.6, False)] * 40 + [(0.99, True)] * 10))
    unchanged = Result(answers={"flag": answer(0.99)}, timing=Timing(), model="m")
    assert cuts.decide(unchanged).scoring == "raw"

    moved = cuts.decide(Result(answers={"flag": answer(0.7)}, timing=Timing(), model="m"))
    assert moved.scoring == "raw+thresholds"
    # And not twice, however many times it is applied.
    assert cuts.decide(moved).scoring == "raw+thresholds"


def test_deciding_changes_the_answer_and_says_so():
    pairs = [(0.6, False)] * 40 + [(0.99, True)] * 20
    cuts, _ = Thresholds.fit(history(pairs))

    borderline = Result(answers={"flag": answer(0.7)}, timing=Timing(), model="m")
    assert borderline["flag"].value is True

    decided = cuts.decide(borderline)
    assert decided["flag"].value is False
    assert decided["flag"].option == "no"
    assert decided.scoring == "raw+thresholds"
    # The evidence is untouched: a threshold moves the decision, not the probabilities.
    assert decided["flag"].probabilities == borderline["flag"].probabilities


def test_an_unknown_question_keeps_the_default():
    """A cut for one question means nothing for another, so it is not borrowed."""
    cuts = Thresholds(cuts={"other": 0.9})
    result = Result(answers={"flag": answer(0.7)}, timing=Timing(), model="m")
    assert cuts.decide(result)["flag"].value is True


def test_a_choice_is_left_alone():
    """Moving a point between `seller` and `buyer` is a preference, not a calibration, so it is refused."""
    choice = Answer(id="who", kind="choice", value="buyer", option="buyer", probabilities={"seller": 0.4, "buyer": 0.6})
    result = Result(answers={"who": choice}, timing=Timing(), model="m")
    cuts, _ = Thresholds.fit([(result, {"who": "seller"})] * 40)
    assert "who" not in cuts.cuts
    assert cuts.decide(result)["who"].option == "buyer"


def test_cuts_survive_a_round_trip(tmp_path):
    cuts, _ = Thresholds.fit(history([(0.6, False)] * 40 + [(0.99, True)] * 20))
    path = tmp_path / "cuts.json"
    cuts.save(path)
    again = Thresholds.load(path)
    assert again.cuts == cuts.cuts
    assert again.support == cuts.support
    assert again.metric == cuts.metric


def test_an_unknown_metric_is_refused():
    with pytest.raises(PrismyraError, match="unknown metric"):
        Thresholds.fit(history([(0.9, True)] * 40), metric="auc")


def test_a_file_from_another_format_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "cuts.json"
    path.write_text('{"format": 99, "cuts": {"flag": 0.9}}')
    with pytest.raises(ThresholdError, match="format 99"):
        Thresholds.load(path)


def test_an_impossible_cut_is_refused(tmp_path):
    path = tmp_path / "cuts.json"
    path.write_text('{"format": 1, "cuts": {"flag": 1.7}}')
    with pytest.raises(ThresholdError, match="impossible cut"):
        Thresholds.load(path)
