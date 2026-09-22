"""Play the same games with each agent and report what happened.

Same seed, same pieces, same order, for every agent -- a difference is then the agent's and not the draw's. Prints
pieces placed, rows cleared and decision time, and nothing else: a use case is worth measuring on its outcome, and how
pretty the board looked on the way there is not the outcome.

    python examples/tetris/play.py --agents heuristic,random --pieces 300 --games 3
    python examples/tetris/play.py --agents heuristic,readout --pieces 40 --games 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents import (
    Generate,
    Heuristic,
    Random,
    ReadOut,
    ReadOutFeatures,
    ReadOutOutcome,
    agreement,
    perception,
    regret,
)
from game import Board, apply, bag

NEEDS_MODEL = {"readout", "readout_outcome", "readout_features", "readout_score", "generate"}


def play(agent, seed: int, pieces: int) -> dict:
    """One game. Ends when a piece will not fit, which is what losing is."""
    board, cleared, placed, seconds, considered = Board(), 0, 0, [], []
    given_up = []
    for piece in bag(seed, pieces):
        choice = agent.choose(board, piece)
        if choice is None:
            break
        # Per decision rather than per game, because a game ends when one bad placement buries a column and its outcome
        # is a few decisions amplified by many. This is how much value the agent gave up on each one.
        given_up.append(regret(board, piece, choice.move))
        outcome = apply(board, piece, choice.move)
        if outcome is None:
            # The agent named a placement that does not fit. A bug in an agent, not a loss, so it is said out loud
            # rather than scored as one.
            raise RuntimeError(f"{agent.name} chose {choice.move} for {piece}, which does not fit")
        board, rows = outcome
        cleared += rows
        placed += 1
        seconds.append(choice.seconds)
        considered.append(choice.considered)
    return {
        "placed": placed,
        "cleared": cleared,
        "median_regret": round(statistics.median(given_up), 3) if given_up else 0.0,
        "mean_regret": round(statistics.mean(given_up), 3) if given_up else 0.0,
        "holes_at_end": board.holes(),
        "median_decision_ms": round(statistics.median(seconds) * 1e3, 1) if seconds else 0.0,
        "total_decision_s": round(sum(seconds), 1),
        "median_placements_considered": statistics.median(considered) if considered else 0,
        "survived": placed == pieces,
    }


def _perception(engine, args) -> int:
    """Whether the board is read, before asking whether it is judged.

    Boards from the reference agent playing, because the question is whether a board an agent would actually meet can be
    read, and a board random play produced is a wreck with nothing to get right.
    """
    board = Board()
    reference = Heuristic()
    rows = []
    for piece in bag(args.seed, args.perception):
        found = perception(engine, board)
        rows.append(found)
        print(
            f"{found['right']:3d}/{found['asked']:<3d} right ({found['accuracy']:.2f}), answering no to everything "
            f"would score {found['always_no_would_score']:.2f}"
        )
        choice = reference.choose(board, piece)
        if choice is None:
            break
        board, _ = apply(board, piece, choice.move)

    right = sum(r["right"] for r in rows)
    asked = sum(r["asked"] for r in rows)
    lazy = statistics.mean(r["always_no_would_score"] for r in rows)
    print(
        f"\n{right}/{asked} = {right / asked:.1%} over {len(rows)} boards, against {lazy:.1%} for answering no to\n"
        f"everything. These answers are mechanically known and depend on no policy, so this separates 'cannot judge a\n"
        f"position' from 'cannot see one' -- which the first conclusion from this example did not."
    )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


def _diagnose(engine, args) -> int:
    """Whether the model's ranking resembles one that plays, on boards a player would actually meet.

    The boards come from the reference agent playing, not from random placements: a ranking is only interesting on
    positions that arise in a game, and a board that random play produced is a wreck no ranking can rescue.
    """
    board = Board()
    pieces = bag(args.seed, args.diagnose + 1)
    reference = Heuristic()
    rows = []
    for piece in pieces[: args.diagnose]:
        for framing in (False, True):
            found = agreement(engine, board, piece, outcome_framing=framing)
            if found:
                found["framing"] = "outcome" if framing else "description"
                found["piece"] = piece
                rows.append(found)
                print(
                    f"{found['framing']:12s} {piece}  {found['placements']:3d} placements  "
                    f"spearman {found['spearman']:+.3f}  top3 {found['model_pick_in_reference_top3']!s:5s}  "
                    f"spread {found['probability_spread']:.4f}"
                )
        choice = reference.choose(board, piece)
        if choice is None:
            break
        board, _ = apply(board, piece, choice.move)

    for framing in ("description", "outcome"):
        mine = [r["spearman"] for r in rows if r["framing"] == framing]
        hits = [r["model_pick_in_reference_top3"] for r in rows if r["framing"] == framing]
        if mine:
            print(
                f"\n{framing}: mean rank correlation with the reference {statistics.mean(mine):+.3f} over "
                f"{len(mine)} boards, and the model's pick was among the reference's best three "
                f"{sum(hits)}/{len(hits)} times"
            )
    print(
        "\nA correlation near zero means the ranking carries nothing, which a score of twenty-six pieces cannot\n"
        "establish on its own: a game ends when one bad placement buries a column, so its outcome is a few decisions\n"
        "amplified by many. Chance puts the model's pick in the best three about 3 in 17 times."
    )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mann_whitney(a: list[int], b: list[int]) -> str:
    """The chance that a game from the first agent beats one from the second, and an exact two-sided p-value.

    Rank based and exact rather than a t-test: survival counts are heavy tailed and these samples are small. Written out
    because this example should not add a dependency to compare two short lists, and because the alternative -- quoting
    two means and calling one of them a floor -- is what a reviewer objected to.
    """
    import itertools
    import math

    if not a or not b:
        return "not enough games"
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    effect = wins / (len(a) * len(b))
    # Exact permutation over which positions of the pooled sample belong to the first group.
    pooled = a + b
    n, k = len(pooled), len(a)
    observed = abs(effect - 0.5)
    total = extreme = 0
    if math.comb(n, k) <= 20_000:
        for picked in itertools.combinations(range(n), k):
            left = [pooled[i] for i in picked]
            right = [pooled[i] for i in range(n) if i not in picked]
            got = sum((x > y) + 0.5 * (x == y) for x in left for y in right) / (k * (n - k))
            total += 1
            extreme += abs(got - 0.5) >= observed
        return f"wins {effect:.2f} of pairings, exact two-sided p = {extreme / total:.4f} over {total} arrangements"
    return f"wins {effect:.2f} of pairings (too many arrangements for an exact p at {n} games)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tetris", description=__doc__)
    parser.add_argument("--agents", default="heuristic,random")
    parser.add_argument("--pieces", type=int, default=300, help="pieces offered per game, so a cap on placements")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument(
        "--diagnose",
        type=int,
        default=0,
        help="instead of playing, compare the model's ranking of the placements with the reference's, on this many "
        "boards that the reference itself reached",
    )
    parser.add_argument(
        "--perception",
        type=int,
        default=0,
        help="instead of playing, ask questions about this many boards whose answers are mechanically known, to find "
        "out whether the board is being read at all",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    wanted = [name.strip() for name in args.agents.split(",") if name.strip()]
    engine = None
    # `--diagnose` and `--perception` ask the model directly, so they need it whatever `--agents` says. Left implicit it
    # loaded nothing and failed ten minutes in, on the first question rather than at the argument -- twice, because
    # fixing it for one of them and not looking for the other is how the same bug gets found a second time.
    if NEEDS_MODEL & set(wanted) or args.diagnose or args.perception:
        from prismyra import Prismyra

        engine = Prismyra(args.model)
        print(f"kernels: {engine.applied.as_dict()['applied']}\n")

    if args.perception:
        return _perception(engine, args)
    if args.diagnose:
        return _diagnose(engine, args)

    made = {
        "random": lambda: Random(seed=args.seed),
        "heuristic": Heuristic,
        "readout": lambda: ReadOut(engine=engine),
        "readout_outcome": lambda: ReadOutOutcome(engine=engine),
        "readout_features": lambda: ReadOutFeatures(engine=engine),
        "readout_score": lambda: ReadOutFeatures(engine=engine, with_score=True),
        "generate": lambda: Generate(engine=engine),
    }

    results: dict[str, list[dict]] = {}
    for name in wanted:
        results[name] = []
        for game in range(args.games):
            agent = made[name]()
            outcome = play(agent, args.seed + game, args.pieces)
            results[name].append(outcome)
            print(
                f"{name:10s} game {game}: placed {outcome['placed']:4d}/{args.pieces}  cleared "
                f"{outcome['cleared']:3d}  holes {outcome['holes_at_end']:3d}  "
                f"{outcome['median_decision_ms']:8.1f} ms/move  ({outcome['total_decision_s']}s total)"
            )

    print(f"\n{'agent':>16} {'games':>6} {'placed':>16} {'cleared':>14} {'regret':>8} {'ms/move':>10} {'capped':>7}")
    for name, games in results.items():
        placed = [g["placed"] for g in games]
        rows = [g["cleared"] for g in games]
        spread = f"{statistics.mean(placed):6.1f} +/- {_stdev(placed):4.1f}"
        print(
            f"{name:>16} {len(games):6d} {spread:>16} "
            f"{statistics.mean(rows):6.1f} +/- {_stdev(rows):4.1f} "
            f"{statistics.median(g['mean_regret'] for g in games):8.2f} "
            f"{statistics.median(g['median_decision_ms'] for g in games):10.1f} "
            f"{sum(g['survived'] for g in games):7d}"
        )
    # Asserted before with three games and no spread: "26.0 against random's 22.7" is not a claim until the spread is
    # beside it, and random Tetris survival is heavy tailed.
    if "random" in results:
        floor = [g["placed"] for g in results["random"]]
        for name, games in results.items():
            if name == "random":
                continue
            mine = [g["placed"] for g in games]
            print(f"\n{name} against random on pieces placed: {_mann_whitney(mine, floor)}")
    print(
        "\n`capped` counts games that ran out of pieces rather than losing: a mean over capped games is censored and\n"
        "understates how much better that agent is."
    )
    print(
        "\nRead the floor and the reference before the two model agents. `random` is what not playing looks like and\n"
        "`heuristic` is what a few lines of arithmetic achieve; a model agent between them is choosing worse than\n"
        "arithmetic, and one at the floor is not choosing at all, whatever its decision time says."
    )
    if args.json:
        args.json.write_text(json.dumps({"pieces": args.pieces, "seed": args.seed, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
