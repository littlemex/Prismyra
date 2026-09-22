"""One scalar that makes a probability mean what it says.

A read-out returns a probability and a caller is entitled to read it as one: if a hundred answers come back at 0.8,
about eighty should be right. They are not, out of the box. A model's output distribution over a few declared options
is systematically too confident, and the fix is one number -- divide the scores by a temperature before the softmax,
choose that temperature on labelled data, and stop.

This is not the same thing as `prismyra.thresholds`, and the difference decides which one a caller wants:

* a **threshold** changes decisions. It moves where "yes" begins, which is what a rare class needs -- at a positive rate
  of one in seventy, taking the larger of two probabilities is cutting in the wrong place, and moving the cut took F1
  from 10.5% to 47.4%. It does not make the probabilities honest.
* a **temperature** changes no decision at all. Scaling every option's score by the same positive number cannot reorder
  them, so the argmax is untouched and the accuracy is identical. What it changes is the number attached to that
  decision, which is what a caller routing on confidence is reading.

Use both, for different jobs. Fit on answers that are not the ones being scored.

Borrowed, with thanks, from `ikermoel/open-alternative-jev` (Apache-2.0), which measured a fitted temperature of 1.3
to 1.5 halving expected calibration error on MMLU and RACE. This package had no calibration of the probabilities at
all, having tried and rejected a different idea -- subtracting what an empty context prefers, which cost 15 points of
accuracy on RACE because an empty context still contains the question. Temperature scaling cannot do that, because it
cannot change a decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .schema import PrismyraError

#: Temperatures searched by `fit`. Coarse on purpose: the curve is shallow near its minimum and a finer grid buys a
#: third decimal place on a number whose effect is measured in percentage points.
GRID = tuple(x / 20 for x in range(4, 81))


def scaled(probabilities: Sequence[float], temperature: float) -> list[float]:
    """The same distribution, sharper or flatter. Takes probabilities rather than raw scores, so it composes with a
    read-out that has already normalised over the options a question declared."""
    if temperature <= 0:
        raise PrismyraError(f"temperature must be positive, not {temperature}")
    logs = [math.log(max(p, 1e-30)) / temperature for p in probabilities]
    top = max(logs)
    weights = [math.exp(x - top) for x in logs]
    total = sum(weights)
    return [w / total for w in weights]


def negative_log_likelihood(probabilities: Sequence[Sequence[float]], correct: Sequence[int]) -> float:
    """What the fit minimises. Rewards a confident right answer and punishes a confident wrong one, which is the
    property that makes it the right objective here -- accuracy cannot see confidence at all."""
    if len(probabilities) != len(correct) or not correct:
        raise PrismyraError("probabilities and correct must be the same non-zero length")
    return -sum(math.log(max(p[y], 1e-30)) for p, y in zip(probabilities, correct, strict=True)) / len(correct)


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]], correct: Sequence[int], bins: int = 10
) -> float:
    """How far the stated confidence is from the observed rate, averaged over bins of confidence.

    Reported rather than the negative log likelihood because it is the number a caller can act on: an error of 5% means
    an answer claiming 0.80 is right about 0.75 of the time. Binned, which makes it sensitive to the bin count -- it is
    a summary, not a test.
    """
    if len(probabilities) != len(correct) or not correct:
        raise PrismyraError("probabilities and correct must be the same non-zero length")
    confidence = [max(p) for p in probabilities]
    hit = [max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probabilities, correct, strict=True)]
    total = 0.0
    for b in range(bins):
        inside = [i for i, c in enumerate(confidence) if min(int(c * bins), bins - 1) == b]
        if inside:
            claimed = sum(confidence[i] for i in inside) / len(inside)
            actual = sum(hit[i] for i in inside) / len(inside)
            total += len(inside) / len(correct) * abs(claimed - actual)
    return total


def brier(probabilities: Sequence[Sequence[float]], correct: Sequence[int]) -> float:
    """Squared error against the truth, summed over options. Bounded and proper, where the negative log likelihood is
    unbounded -- one confident mistake can dominate a log score and cannot dominate this one."""
    if len(probabilities) != len(correct) or not correct:
        raise PrismyraError("probabilities and correct must be the same non-zero length")
    total = 0.0
    for p, y in zip(probabilities, correct, strict=True):
        total += sum((value - (i == y)) ** 2 for i, value in enumerate(p))
    return total / len(correct)


@dataclass
class Temperature:
    """A fitted scalar, and the numbers that say whether fitting it helped."""

    value: float = 1.0
    #: Before and after, on whatever was fitted. Kept so a caller cannot report an improvement without the pair.
    before: dict | None = None
    after: dict | None = None

    @classmethod
    def fit(cls, probabilities: Sequence[Sequence[float]], correct: Sequence[int]) -> Temperature:
        """Choose the temperature that best explains these labels, and record what it did.

        Fitted on answers that are **not** the ones being reported. Fitting and scoring the same answers gives a
        temperature that flatters itself; `cross_fit` is the honest way to use one slice for both.
        """
        if len(probabilities) != len(correct) or not correct:
            raise PrismyraError("probabilities and correct must be the same non-zero length")
        best = min(GRID, key=lambda t: negative_log_likelihood([scaled(p, t) for p in probabilities], correct))
        after = [scaled(p, best) for p in probabilities]
        return cls(value=best, before=_report(probabilities, correct), after=_report(after, correct))

    def apply(self, probabilities: Sequence[float]) -> list[float]:
        return scaled(probabilities, self.value)

    def rescore(self, answer):
        """The same answer with honest numbers. The chosen option cannot change: scaling by a positive number cannot
        reorder a distribution, so this is a promise and not a hope."""
        from .schema import Answer

        options = list(answer.probabilities)
        values = self.apply([answer.probabilities[o] for o in options])
        return Answer(
            id=answer.id,
            kind=answer.kind,
            value=answer.value,
            option=answer.option,
            probabilities=dict(zip(options, values, strict=True)),
        )


def cross_fit(
    probabilities: Sequence[Sequence[float]], correct: Sequence[int], folds: int = 2
) -> tuple[list[list[float]], list[float]]:
    """Scale every answer with a temperature fitted on the answers it was not part of.

    The honest way to report a calibration improvement when there is only one labelled slice. Returns the scaled
    probabilities and the temperature each fold chose, because temperatures that disagree across folds are the signal
    that there is not enough data to fit one.
    """
    if folds < 2:
        raise PrismyraError(f"cross fitting needs at least two folds, not {folds}")
    if len(correct) < folds:
        raise PrismyraError(f"{len(correct)} answers cannot be split into {folds} folds")
    out: list[list[float] | None] = [None] * len(correct)
    chosen = []
    for fold in range(folds):
        held = [i for i in range(len(correct)) if i % folds == fold]
        rest = [i for i in range(len(correct)) if i % folds != fold]
        fitted = Temperature.fit([probabilities[i] for i in rest], [correct[i] for i in rest])
        chosen.append(fitted.value)
        for i in held:
            out[i] = fitted.apply(probabilities[i])
    assert all(row is not None for row in out)
    return [row for row in out if row is not None], chosen


def _report(probabilities: Sequence[Sequence[float]], correct: Sequence[int]) -> dict:
    return {
        "accuracy": round(
            sum(max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probabilities, correct, strict=True))
            / len(correct),
            4,
        ),
        "negative_log_likelihood": round(negative_log_likelihood(probabilities, correct), 4),
        "expected_calibration_error": round(expected_calibration_error(probabilities, correct), 4),
        "brier": round(brier(probabilities, correct), 4),
    }
