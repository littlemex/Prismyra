"""The things that choose a move, and the reference that says whether choosing well is happening at all.

Four of them, and the order matters. `Random` is the floor; `Heuristic` is what a few lines of arithmetic achieve and is
the only one here that is known to play; `ReadOut` is this package; `Generate` is the same model deciding the same way
with the answer decoded instead of read out. A comparison of the last two is worth nothing without the first two, since
two agents that both play like `Random` can be compared on speed forever without either of them playing Tetris.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import pairwise

from game import Board, Move, apply, legal_moves

#: Weights over the features below. Not tuned here and not meant to be: this is a reference point, and a reference that
#: was fitted against the same games it is a reference for would be a competitor. These are the long-published
#: Dellacherie-style features with signs and rough magnitudes that are common to every version of them.
WEIGHTS = {"lines": 3.0, "holes": -4.0, "height": -0.5, "bumpiness": -0.2, "well": -1.0}


def features(before: Board, after: Board, cleared: int) -> dict[str, float]:
    heights = after.heights()
    return {
        "lines": float(cleared),
        "holes": float(after.holes()),
        "height": float(sum(heights)),
        "bumpiness": float(sum(abs(a - b) for a, b in pairwise(heights))),
        "well": float(max(heights) - min(heights)),
    }


@dataclass
class Choice:
    """What an agent decided and what it cost. `seconds` is decision time only -- the game itself is free."""

    move: Move
    seconds: float
    considered: int


@dataclass
class Random:
    """The floor. Any agent that cannot beat this is not playing."""

    name: str = "random"
    seed: int = 0
    _rng: object = field(default=None, repr=False)

    def choose(self, board: Board, piece: str) -> Choice | None:
        import random as _random

        if self._rng is None:
            self._rng = _random.Random(self.seed)
        started = time.perf_counter()
        playable = [m for m in legal_moves(piece) if apply(board, piece, m) is not None]
        if not playable:
            return None
        return Choice(self._rng.choice(playable), time.perf_counter() - started, len(playable))  # type: ignore[attr-defined]


@dataclass
class Heuristic:
    """Arithmetic over the board that results from each placement. The reference."""

    name: str = "heuristic"

    def choose(self, board: Board, piece: str) -> Choice | None:
        started = time.perf_counter()
        best, score = None, None
        considered = 0
        for move in legal_moves(piece):
            outcome = apply(board, piece, move)
            if outcome is None:
                continue
            considered += 1
            after, cleared = outcome
            value = sum(WEIGHTS[k] * v for k, v in features(board, after, cleared).items())
            if score is None or value > score:
                best, score = move, value
        if best is None:
            return None
        return Choice(best, time.perf_counter() - started, considered)


#: How a board and a piece are put to the model. One string, used by both model agents, so the only difference between
#: them is how the answer comes out.
PROMPT = """You are playing Tetris. Row 0 is the top and row 19 is the floor. A '#' is a filled cell and a '.' is
empty. A piece is dropped straight down into a column at a chosen rotation; it cannot slide sideways underneath an
overhang. A row vanishes when all ten of its cells are filled.

A good placement keeps the stack low and flat and leaves no empty cell with a filled cell above it, because such a cell
cannot be filled until the rows above it are cleared.

The board:

{board}

The piece to place is {piece}."""


@dataclass
class ReadOut:
    """This package: one pass over the board, then every legal placement scored as a row of one batch.

    The shape this design is for, and the first use of it here that is not a document. A piece has up to 34 placements,
    so one board is asked 34 questions -- which is the case the crossover measurement says the read-out wins, and it
    wins it on a state that the previous decision produced rather than on a passage somebody wrote.
    """

    engine: object
    name: str = "readout"

    def choose(self, board: Board, piece: str) -> Choice | None:
        from prismyra import Boolean

        playable = [m for m in legal_moves(piece) if apply(board, piece, m) is not None]
        if not playable:
            return None
        questions = [
            Boolean(id=f"m{i}", prompt=f"Is placing the piece as {m.describe(piece)} a good move?")
            for i, m in enumerate(playable)
        ]
        started = time.perf_counter()
        result = self.engine.ask(PROMPT.format(board=board.render(), piece=piece), questions)  # type: ignore[attr-defined]
        seconds = time.perf_counter() - started
        # Ranked by the probability of yes rather than by the yes/no decision. Every placement may come back "no" on a
        # bad board and a decision still has to be made, so the ranking is the signal here and the threshold is not.
        best = max(range(len(playable)), key=lambda i: result[f"m{i}"].probabilities["yes"])
        return Choice(playable[best], seconds, len(playable))


@dataclass
class Generate:
    """The same model, the same questions, the answer decoded instead of read out.

    Here so that a difference between this and `ReadOut` is the extraction and nothing else: same prompt, same
    placements, same ranking by the probability of yes. Slow on purpose -- one generation per placement -- because what
    is being measured is what the extraction costs and what it buys.
    """

    engine: object
    name: str = "generate"
    max_tokens: int = 4

    def choose(self, board: Board, piece: str) -> Choice | None:
        import torch

        playable = [m for m in legal_moves(piece) if apply(board, piece, m) is not None]
        if not playable:
            return None
        engine = self.engine
        context = PROMPT.format(board=board.render(), piece=piece)
        started = time.perf_counter()
        scores = []
        for move in playable:
            text = (
                f"{context}\n\nIs placing the piece as {move.describe(piece)} a good move? "
                f"Answer yes or no.\n<think>\n\n</think>\n\n"
            )
            ids = engine.tokenizer(text, return_tensors="pt").to(engine.device)  # type: ignore[attr-defined]
            with torch.inference_mode():
                out = engine.backbone(**ids)  # type: ignore[attr-defined]
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            logits = engine.unembedding(hidden[:, -1].float())  # type: ignore[attr-defined]
            yes = engine.tokenizer(" yes", add_special_tokens=False)["input_ids"][-1]  # type: ignore[attr-defined]
            no = engine.tokenizer(" no", add_special_tokens=False)["input_ids"][-1]  # type: ignore[attr-defined]
            pair = torch.softmax(logits[0, [yes, no]], dim=-1)
            scores.append(float(pair[0]))
        seconds = time.perf_counter() - started
        best = max(range(len(playable)), key=lambda i: scores[i])
        return Choice(playable[best], seconds, len(playable))


#: The same rules and board, but each question carries the board the placement would produce. The branch suffix has
#: room for it -- a rendered board is about 150 tokens against a 512 limit -- so this is still one context pass with
#: every candidate as a row, and it asks the model to judge a position rather than to imagine one.
OUTCOME_QUESTION = """After that placement the board would look like this:

{after}

Is that a good position to be in?"""


@dataclass
class ReadOutOutcome:
    """The read-out again, shown the consequence of each placement instead of a description of it.

    Here because the first framing scored at the random floor, and a floor score has two explanations: the model cannot
    judge Tetris positions, or it cannot work out which position a phrase like "left edge in column 4" produces. This
    separates them by doing the placement itself and asking only the judgement. If this also scores at the floor, the
    first explanation is the one left standing.
    """

    engine: object
    name: str = "readout_outcome"

    def choose(self, board: Board, piece: str) -> Choice | None:
        from prismyra import Boolean

        outcomes = [(m, apply(board, piece, m)) for m in legal_moves(piece)]
        playable = [(m, o) for m, o in outcomes if o is not None]
        if not playable:
            return None
        questions = [
            Boolean(id=f"m{i}", prompt=OUTCOME_QUESTION.format(after=after.render()))
            for i, (_, (after, _)) in enumerate(playable)
        ]
        started = time.perf_counter()
        result = self.engine.ask(PROMPT.format(board=board.render(), piece=piece), questions)  # type: ignore[attr-defined]
        seconds = time.perf_counter() - started
        best = max(range(len(playable)), key=lambda i: result[f"m{i}"].probabilities["yes"])
        return Choice(playable[best][0], seconds, len(playable))


def agreement(engine, board: Board, piece: str, outcome_framing: bool = False) -> dict:
    """How the model's ranking of the placements compares with the arithmetic reference's.

    A better instrument than the game's outcome for the question "is there any signal here". A game ends when one bad
    placement buries a column, so its result is a few decisions amplified by many; this looks at every decision at once
    and asks whether the model's order resembles an order that is known to play.

    Reported as Spearman's rank correlation and as whether the model's own pick is among the reference's best three.
    Correlation near zero with the reference near one means the ranking carries nothing, which cannot be read off a
    score of twenty-six pieces.
    """
    from prismyra import Boolean

    outcomes = [(m, apply(board, piece, m)) for m in legal_moves(piece)]
    playable = [(m, o) for m, o in outcomes if o is not None]
    if len(playable) < 3:
        return {}

    reference = [
        sum(WEIGHTS[k] * v for k, v in features(board, after, cleared).items()) for _, (after, cleared) in playable
    ]
    if outcome_framing:
        questions = [
            Boolean(id=f"m{i}", prompt=OUTCOME_QUESTION.format(after=after.render()))
            for i, (_, (after, _)) in enumerate(playable)
        ]
    else:
        questions = [
            Boolean(id=f"m{i}", prompt=f"Is placing the piece as {m.describe(piece)} a good move?")
            for i, (m, _) in enumerate(playable)
        ]
    result = engine.ask(PROMPT.format(board=board.render(), piece=piece), questions)
    mine = [result[f"m{i}"].probabilities["yes"] for i in range(len(playable))]

    return {
        "placements": len(playable),
        "spearman": _spearman(mine, reference),
        "model_pick_in_reference_top3": max(range(len(mine)), key=lambda i: mine[i])
        in sorted(range(len(reference)), key=lambda i: -reference[i])[:3],
        "probability_spread": round(max(mine) - min(mine), 4),
    }


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, written out because this example should not add a dependency to compare two short lists."""
    ranks_a, ranks_b = _ranks(a), _ranks(b)
    n = len(a)
    mean = (n - 1) / 2
    top = sum((x - mean) * (y - mean) for x, y in zip(ranks_a, ranks_b, strict=True))
    left = sum((x - mean) ** 2 for x in ranks_a) ** 0.5
    right = sum((y - mean) ** 2 for y in ranks_b) ** 0.5
    return round(top / (left * right), 4) if left and right else 0.0


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, which matter here: a model that answers every placement identically must come out as no
    correlation rather than as whatever the sort order happened to be."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out
