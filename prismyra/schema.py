"""Questions, requests and answers. The read-out contract lives here because the read-out is the product.

What a probability means is stated once, in `SCORING`, and versioned. Three different things could be called "the
probability of an option" -- a first-token logit, a sequence likelihood, a renormalisation over the declared options --
and they do not agree. See docs/READOUT.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

#: Bumped whenever the meaning of a probability changes. Callers that store answers should store this beside them.
SCORING_VERSION = 1

SCORING = """\
For each declared option, the model's own output embedding scores the single token that spells that option, at the
branch's final position. Those scores are softmaxed **over the declared options only**. So a probability is a relative
preference among the options you offered, not a calibrated belief and not comparable across different option sets.\
"""

MAX_OPTIONS = 255


class PrismyraError(Exception):
    """Base for every error this library raises deliberately."""


class QuestionError(PrismyraError):
    """A question cannot be scored as written.

    Raised as early as the check allows. What can be seen from the question alone -- an empty prompt, one option, a
    repeat -- is refused at construction. Two refusals need the model's tokenizer and so belong to the engine: an option
    that is more than one token, and two options sharing a first token. `Prismyra.validate` runs those before the
    context is read, so they still cost no device time.
    """


@dataclass(frozen=True)
class Question:
    """A question with a closed set of answers.

    `id` names the answer in the result. `prompt` is the text the model reads. Subclasses declare `options`.
    """

    id: str
    prompt: str

    @property
    def options(self) -> list[str]:
        raise NotImplementedError

    @property
    def kind(self) -> str:
        raise NotImplementedError

    def value_of(self, option: str) -> object:
        """The typed value an option stands for. Overridden where the option text is not the value."""
        return option

    def __post_init__(self) -> None:
        if not self.id:
            raise QuestionError("a question needs an id")
        if not self.prompt.strip():
            raise QuestionError(f"question {self.id!r} has an empty prompt")
        opts = self.options
        if len(opts) < 2:
            raise QuestionError(f"question {self.id!r} declares {len(opts)} options; at least two are needed")
        if len(opts) > MAX_OPTIONS:
            raise QuestionError(f"question {self.id!r} declares {len(opts)} options; the limit is {MAX_OPTIONS}")
        if len(set(opts)) != len(opts):
            raise QuestionError(f"question {self.id!r} repeats an option: {opts}")
        for opt in opts:
            if not isinstance(opt, str) or not opt.strip():
                raise QuestionError(f"question {self.id!r} has an empty or non-text option: {opt!r}")


@dataclass(frozen=True)
class Boolean(Question):
    """Yes or no. `value` is a bool."""

    @property
    def options(self) -> list[str]:
        return ["no", "yes"]

    @property
    def kind(self) -> Literal["boolean"]:
        return "boolean"

    def value_of(self, option: str) -> bool:
        return option == "yes"


@dataclass(frozen=True)
class Choice(Question):
    """One of a named set. `value` is the option's own text."""

    choices: Sequence[str] = ()

    @property
    def options(self) -> list[str]:
        return list(self.choices)

    @property
    def kind(self) -> Literal["choice"]:
        return "choice"

    def __post_init__(self) -> None:
        # A string is a sequence of characters, so `choices="yes"` would quietly become three options.
        if isinstance(self.choices, str | bytes):
            raise QuestionError(f"question {self.id!r} was given one string as its choices; pass a list of options")
        # Copied into a tuple, which is what makes this dataclass frozen in fact and not only in name. The options are
        # read several times during one answer -- to find their tokens, to render the prompt, to label the probabilities
        # -- and a list the caller can still reorder between those reads would attach the scores to the wrong names.
        object.__setattr__(self, "choices", tuple(self.choices))
        super().__post_init__()


@dataclass(frozen=True)
class Scale(Question):
    """An integer range, inclusive. `value` is an int."""

    low: int = 1
    high: int = 5

    @property
    def options(self) -> list[str]:
        return [str(n) for n in range(self.low, self.high + 1)]

    @property
    def kind(self) -> Literal["scale"]:
        return "scale"

    def value_of(self, option: str) -> int:
        return int(option)

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise QuestionError(f"question {self.id!r} has high {self.high} not above low {self.low}")
        span = self.high - self.low + 1
        if span > MAX_OPTIONS:
            raise QuestionError(f"question {self.id!r} spans {span} values; the limit is {MAX_OPTIONS}")
        super().__post_init__()


@dataclass(frozen=True)
class Answer:
    """One question's answer.

    `value` is typed by the question's kind: a bool, the chosen option's text, or an int. `option` is always the raw
    option text. `probabilities` is over the declared options only -- see `SCORING`. There is deliberately no
    "confidence" field: the largest probability is not a calibrated one, and naming it confidence invites reading it
    as one.
    """

    id: str
    kind: str
    value: object
    option: str
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class Timing:
    """Where the milliseconds went. Waiting and working are separate because their fixes are.

    `queue_ms` is time spent waiting for the device, which batching or another device fixes. `context_ms` and
    `readout_ms` are device time, which only kernels fix.
    """

    queue_ms: float = 0.0
    context_ms: float = 0.0
    readout_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.queue_ms + self.context_ms + self.readout_ms


@dataclass(frozen=True)
class Request:
    """One context and the questions to ask about it."""

    context: str
    questions: Sequence[Question]

    def __post_init__(self) -> None:
        if not self.context.strip():
            raise QuestionError("a request needs a context")
        if not self.questions:
            raise QuestionError("a request needs at least one question")
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise QuestionError(f"duplicate question ids: {ids}")


@dataclass(frozen=True)
class Result(Mapping[str, Answer]):
    """The answers to one request, with what produced them.

    An envelope rather than a bare mapping so that the scoring version and the model travel with the numbers: an answer
    stored without them cannot be compared to a later one.
    """

    answers: Mapping[str, Answer]
    timing: Timing
    model: str
    scoring_version: int = SCORING_VERSION
    context_tokens: int = 0

    def __getitem__(self, question_id: str) -> Answer:
        return self.answers[question_id]

    def __iter__(self):
        """Ids, as a mapping iterates keys. `values()` gives the answers.

        Deliberately not the answers themselves. Something that answers `result["id"]` and `"id" in result` but hands
        out values when iterated breaks every piece of code that treats a mapping as a mapping, `dict(result)` first.
        """
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)


def render_question(question: Question) -> str:
    """The text a branch reads. Options are listed in a canonical order, and that is load-bearing.

    Listing them in the caller's order made answers depend on it, by up to 0.165. Omitting the list entirely put
    three-way questions at chance. So they are listed, sorted, always.
    """
    options = ", ".join(sorted(question.options))
    return f"Question: {question.prompt}\nAnswer with one of: {options}\nAnswer:"
