"""Where to stand when the interesting answer is rare.

The read-out gives a probability. Turning that into an answer means choosing a point, and taking the larger of two
probabilities chooses 0.5. That is the right point only when the two answers are equally likely before you look.

Measured on LexGLUE's unfair terms-of-service clauses, where about 1.5% of the answers are yes: the read-out reaches
60% recall at 6% precision, so it says yes to nearly everything. The same model generating the answer does the same,
so this is not the read-out's limit -- it is the absence of a decision point.

**This is supervised operating-point selection, and it is worth naming as such.** Choosing a decision threshold from
labelled examples is textbook cost-sensitive decision making, not a new idea. What it has going for it is the price:
no gradient, no device, no second model, and the inputs are answers already given beside what turned out to be true,
both of which a caller that logs its requests already holds. That makes it the floor any expensive mechanism has to
clear, and the reason it lives here rather than in a notebook. It does not make it label-free: it needs labels, and a
fine-tune given the same labels will go further.

    from prismyra.thresholds import Thresholds

    history = [(engine.ask(text, questions), known_answers) for text, known_answers in labelled]
    cuts, report = Thresholds.fit(history)
    print(report.explain())            # why each question got a cut, or did not
    cuts.save("cuts.json")

    answered = cuts.decide(engine.ask(new_text, questions))

Fit on answers the cuts are not then judged on. A cut fitted on the answers it is scored against is an oracle and will
look far better than it is.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .schema import Answer, PrismyraError, Result

#: The point a two-option answer defaults to, which is what taking the larger probability does.
DEFAULT_CUT = 0.5

#: Stored beside the cuts so a file from a later version is refused rather than misread.
FORMAT = 1

#: Gold values that mean yes. Anything outside this and `NO_VALUES` is refused rather than read as no: a gold of
#: `"unfair"` or `None` silently counted as a negative is label corruption, and it corrupts every cut fitted after it.
YES_VALUES = frozenset({"true", "yes", "1"})
NO_VALUES = frozenset({"false", "no", "0"})


class ThresholdError(PrismyraError):
    """A cut cannot be fitted, stored or applied as asked."""


@dataclass
class Fitted:
    """What was decided for one question, and on what.

    `positives` and `negatives` separately, because a count of rows says nothing at a 1.5% base rate: twenty rows
    usually hold zero or one positive, and a point fitted on one positive is a point fitted on noise.
    """

    cut: float | None
    positives: int
    negatives: int
    score: float | None = None
    refused: str | None = None


@dataclass
class Report:
    """Per question, what happened and why. Returned rather than logged, because the interesting part of a fit is
    usually which questions did not get one."""

    questions: dict[str, Fitted] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    def explain(self) -> str:
        lines = []
        for name, fitted in sorted(self.questions.items()):
            seen = f"{fitted.positives} yes, {fitted.negatives} no"
            if fitted.cut is None:
                lines.append(f"{name}: no cut ({seen}) -- {fitted.refused}")
            else:
                lines.append(f"{name}: cut {fitted.cut:.3f} ({seen}), fitting score {fitted.score:.3f}")
        for reason, count in sorted(self.skipped.items()):
            lines.append(f"skipped {count} answers: {reason}")
        return "\n".join(lines) or "nothing to fit"


@dataclass
class Thresholds:
    """One decision point per question id, with what it was fitted on and what it was fitted for.

    Keyed by question id because that is what a caller names a recurring question by, and recurring questions are the
    case this serves: the same eight asked of ten thousand contracts. An id is a weak key -- it says nothing about the
    wording, the options, the model or the read-out -- so the provenance below travels with it and `decide` complains
    when it does not match.
    """

    cuts: dict[str, float] = field(default_factory=dict)
    support: dict[str, tuple[int, int]] = field(default_factory=dict)
    metric: str = "f1"
    #: What produced the probabilities these cuts were fitted on. A cut fitted against one model applied to another's
    #: probabilities is the likely operational mistake, and it is silent without this.
    model: str | None = None
    scoring: str | None = None
    fitted_on: str | None = None

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        history: Iterable[tuple[Result, Mapping[str, object]]],
        minimum_positives: int = 5,
        minimum_observations: int = 20,
        metric: str = "f1",
    ) -> tuple[Thresholds, Report]:
        """Choose each question's point from answers already given and what turned out to be true.

        `history` pairs a `Result` with the correct answers for it, keyed the same way, and may be any iterable so a
        log can be streamed rather than held. Only two-option boolean questions are fitted: with three options there
        is no single point to move, and with a two-way choice like seller against buyer, moving the point is a
        preference rather than a calibration.

        Returns the cuts and a report. Everything refused is refused loudly in that report rather than quietly
        omitted, because the useful question after a fit is which questions did not get one.
        """
        if metric not in ("f1", "accuracy"):
            raise ThresholdError(f"unknown metric {metric!r}; expected f1 or accuracy")

        observed: dict[str, list[tuple[float, bool]]] = {}
        report = Report()
        model, scoring = None, None

        for result, gold in history:
            model = model or result.model
            scoring = scoring or result.scoring
            for answer in result.values():
                reason = _why_not(answer, gold)
                if reason:
                    report.skipped[reason] = report.skipped.get(reason, 0) + 1
                    continue
                yes = _positive_option(answer)
                assert yes is not None  # `_why_not` has already established this
                observed.setdefault(answer.id, []).append(
                    (float(answer.probabilities[yes]), _as_bool(answer.id, gold[answer.id]))
                )

        cuts: dict[str, float] = {}
        support: dict[str, tuple[int, int]] = {}
        for name, pairs in observed.items():
            positives = sum(1 for _, label in pairs if label)
            negatives = len(pairs) - positives
            support[name] = (positives, negatives)

            if len(pairs) < minimum_observations:
                report.questions[name] = Fitted(
                    None, positives, negatives, refused=f"fewer than {minimum_observations} answers"
                )
                continue
            if positives < minimum_positives:
                report.questions[name] = Fitted(
                    None, positives, negatives, refused=f"fewer than {minimum_positives} positive labels"
                )
                continue
            cut, score = _best_cut(pairs, metric)
            cuts[name] = cut
            report.questions[name] = Fitted(cut, positives, negatives, score=score)

        return (
            cls(
                cuts=cuts,
                support=support,
                metric=metric,
                model=model,
                scoring=scoring,
                fitted_on=time.strftime("%Y-%m-%d"),
            ),
            report,
        )

    # ------------------------------------------------------------------ using
    def decide(self, result: Result, strict: bool = True) -> Result:
        """The same result with two-option answers re-decided at their own point.

        A new `Result`, and `scoring` records what happened -- but only when an answer actually moved, and never
        twice. Provenance that says a threshold was applied when none was is worse than none.

        With `strict`, a result from a different model or a different read-out than the cuts were fitted against is
        refused. That mistake is otherwise silent: the probabilities still look like probabilities.

        The probabilities are left exactly as they were. A threshold moves the decision, not the evidence, so an
        answer after this may name an option that is not the largest probability -- deliberately, and visibly through
        `scoring`.
        """
        if strict:
            self._check(result)

        changed, touched = {}, False
        for answer in result.values():
            cut = self.cuts.get(answer.id)
            yes = _positive_option(answer) if cut is not None else None
            if cut is None or yes is None or len(answer.probabilities) != 2:
                changed[answer.id] = answer
                continue
            says_yes = float(answer.probabilities[yes]) >= cut
            option = yes if says_yes else next(o for o in answer.probabilities if o != yes)
            touched = touched or option != answer.option
            changed[answer.id] = dataclasses.replace(answer, option=option, value=says_yes)

        scoring = result.scoring
        if touched and not scoring.endswith("+thresholds"):
            scoring = f"{scoring}+thresholds"
        return dataclasses.replace(result, answers=changed, scoring=scoring)

    def _check(self, result: Result) -> None:
        if self.model and result.model != self.model:
            raise ThresholdError(
                f"these cuts were fitted on probabilities from {self.model!r} and this result came from "
                f"{result.model!r}; pass strict=False to apply them anyway"
            )
        # Compared without the tag this method adds, so applying cuts to an already-decided result is not treated as a
        # mismatch with itself. A test caught that: the second call refused what the first had produced.
        already = result.scoring.removesuffix("+thresholds")
        if self.scoring and already != self.scoring.removesuffix("+thresholds"):
            raise ThresholdError(
                f"these cuts were fitted against {self.scoring!r} scoring and this result used {result.scoring!r}; "
                f"pass strict=False to apply them anyway"
            )

    # ------------------------------------------------------------------ keeping
    def save(self, path: str | Path) -> None:
        """Written through a temporary file and renamed, so an interrupted save leaves the previous cuts intact."""
        target = Path(path)
        payload = {
            "format": FORMAT,
            "metric": self.metric,
            "model": self.model,
            "scoring": self.scoring,
            "fitted_on": self.fitted_on,
            "cuts": self.cuts,
            "support": {name: list(counts) for name, counts in self.support.items()},
        }
        beside = target.with_suffix(target.suffix + ".partial")
        beside.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(beside, target)

    @classmethod
    def load(cls, path: str | Path) -> Thresholds:
        """Read cuts back, refusing a file this version cannot read rather than misreading it."""
        stored = json.loads(Path(path).read_text())
        if not isinstance(stored, dict):
            raise ThresholdError(f"{path} does not hold a thresholds file")
        version = stored.get("format")
        if version != FORMAT:
            raise ThresholdError(f"{path} is format {version!r} and this version reads {FORMAT}")
        cuts = stored.get("cuts") or {}
        for name, cut in cuts.items():
            if not isinstance(cut, int | float) or not math.isfinite(cut) or not 0.0 <= float(cut) <= 1.0:
                raise ThresholdError(f"{path} has an impossible cut for {name!r}: {cut!r}")
        metric = stored.get("metric", "f1")
        if metric not in ("f1", "accuracy"):
            raise ThresholdError(f"{path} was fitted for an unknown metric {metric!r}")
        return cls(
            cuts={name: float(cut) for name, cut in cuts.items()},
            support={name: tuple(counts) for name, counts in (stored.get("support") or {}).items()},
            metric=metric,
            model=stored.get("model"),
            scoring=stored.get("scoring"),
            fitted_on=stored.get("fitted_on"),
        )


def _best_cut(pairs: list[tuple[float, bool]], metric: str) -> tuple[float, float]:
    """Try every point the observed probabilities offer and keep the best, with the score it got.

    Exhaustive on purpose: one parameter, at most a few thousand observations, and anything cleverer would fit the
    noise more carefully.

    Two details that are decisions rather than accidents. A candidate sits at the **midpoint** between adjacent
    observed probabilities rather than on one of them, so a value a hair below a training example does not flip; and
    one candidate sits above every observation, so predicting no to everything is expressible -- without it, a
    question where that is the best available answer cannot say so.
    """
    values = sorted({probability for probability, _ in pairs})
    candidates = {DEFAULT_CUT, math.nextafter(values[-1], math.inf)}
    candidates.update((low + high) / 2 for low, high in itertools.pairwise(values))
    candidates.add(values[0] / 2 if values[0] > 0 else 0.0)

    actual = sum(1 for _, label in pairs if label)
    best, best_score = DEFAULT_CUT, -1.0
    for candidate in sorted(candidates):
        hits = sum(1 for probability, label in pairs if probability >= candidate and label)
        claimed = sum(1 for probability, _ in pairs if probability >= candidate)
        if metric == "accuracy":
            score = sum(1 for probability, label in pairs if (probability >= candidate) == label) / len(pairs)
        else:
            precision = hits / claimed if claimed else 0.0
            recall = hits / actual if actual else 0.0
            score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


def _why_not(answer: Answer, gold: Mapping[str, object]) -> str | None:
    """Why this answer cannot contribute to a fit, or None if it can."""
    if answer.id not in gold:
        return "no label"
    if answer.kind != "boolean":
        return f"not a boolean question ({answer.kind})"
    if len(answer.probabilities) != 2:
        return f"{len(answer.probabilities)} options"
    if _positive_option(answer) is None:
        return "no option meaning yes"
    return None


def _positive_option(answer: Answer) -> str | None:
    """Which of two options a threshold is a threshold on. Only a boolean has one."""
    if answer.kind != "boolean":
        return None
    return next((option for option in answer.probabilities if option.lower() in YES_VALUES), None)


def _as_bool(question_id: str, value: object) -> bool:
    """A gold label as a boolean, refusing anything it cannot read.

    Reading an unrecognised value as no is the quiet failure this exists to prevent: a gold of `"unfair"`, `None` or
    `"y"` counted as a negative shifts every cut fitted afterwards, and nothing in the output says so.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    raise ThresholdError(
        f"the label for {question_id!r} is {value!r}, which is neither yes nor no. Pass a bool, or one of "
        f"{sorted(YES_VALUES | NO_VALUES)}."
    )
