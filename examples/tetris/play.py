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

from agents import Generate, Heuristic, Random, ReadOut, ReadOutOutcome, agreement
from game import Board, apply, bag

NEEDS_MODEL = {"readout", "readout_outcome", "generate"}


def play(agent, seed: int, pieces: int) -> dict:
    """One game. Ends when a piece will not fit, which is what losing is."""
    board, cleared, placed, seconds, considered = Board(), 0, 0, [], []
    for piece in bag(seed, pieces):
        choice = agent.choose(board, piece)
        if choice is None:
            break
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
        "holes_at_end": board.holes(),
        "median_decision_ms": round(statistics.median(seconds) * 1e3, 1) if seconds else 0.0,
        "total_decision_s": round(sum(seconds), 1),
        "median_placements_considered": statistics.median(considered) if considered else 0,
        "survived": placed == pieces,
    }


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
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    wanted = [name.strip() for name in args.agents.split(",") if name.strip()]
    engine = None
    # `--diagnose` asks the model to rank placements, so it needs the model whatever `--agents` says. Left implicit it
    # loaded nothing and failed ten minutes in, on the first question rather than at the argument.
    if NEEDS_MODEL & set(wanted) or args.diagnose:
        from prismyra import Prismyra

        engine = Prismyra(args.model)
        print(f"kernels: {engine.applied.as_dict()['applied']}\n")

    if args.diagnose:
        return _diagnose(engine, args)

    made = {
        "random": lambda: Random(seed=args.seed),
        "heuristic": Heuristic,
        "readout": lambda: ReadOut(engine=engine),
        "readout_outcome": lambda: ReadOutOutcome(engine=engine),
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

    print(f"\n{'agent':>10} {'placed':>8} {'cleared':>8} {'ms/move':>10} {'placements':>11}")
    for name, games in results.items():
        print(
            f"{name:>10} {statistics.mean(g['placed'] for g in games):8.1f} "
            f"{statistics.mean(g['cleared'] for g in games):8.1f} "
            f"{statistics.median(g['median_decision_ms'] for g in games):10.1f} "
            f"{statistics.median(g['median_placements_considered'] for g in games):11.1f}"
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
