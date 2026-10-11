"""learn3's structured feature builder for the `snake-move-up-v1` tag (DISTILL-RL-DESIGN-v2.md section 3,
extended by learn3's own task: "タグごとに、構造のある特徴の作り方を指定できるようにする").

learn2's own finding (RUN-learn2.md section 7) is why this exists: the hashed-TF-IDF default never represents
the board's *connectivity* -- which cells are reachable from where -- only the bag of tokens the board's text
description happens to contain. `snake_board_v1` reads the board back out of that same text (the exact template
`snake_port.build_request` writes it in -- see `parse_context`) and recomputes, in Python, over the real grid:
per direction, whether the next step is safe, how many cells a BFS can still reach from there, the distance to
the food (now, and after that step), the distance to the snake's own tail, and the distance to the nearest
wall; plus the snake's length, its head and food positions, and its current heading.

Also here: `simple_rule_choice_is_up`, the hand-rule baseline the brief asks for ("特徴だけを使った単純な決まり")
-- among the directions `snake_board_v1` marked safe, pick the one with the most reachable cells, so a later
script can report its accuracy next to the trained student's, and so a student that merely matched this trivial
rule (rather than learning anything a LightGBM split could not already get from the rule itself) would be
visible as such.

Not part of the package -- a one-off feature definition for this change's own verification, the same position
`snake_port.py` (which this module imports and reuses, rather than re-deriving the board model) already has.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snake_port import DELTA, GRID, SnakeState, _manhattan, _open_area  # noqa: E402

from prismyra.learn.features import register_features  # noqa: E402

_BODY_RE = re.compile(
    r"The snake's body, head first: (?P<body>.+?)\. It is currently moving (?P<dir>\w+)\. "
    r"The food is at \((?P<fx>\d+),(?P<fy>\d+)\)"
)
_CELL_RE = re.compile(r"\((\d+),(\d+)\)")


def parse_context(context: str) -> SnakeState:
    """The inverse of `snake_port.build_request`'s own context template -- reads the board state back out of
    exactly the text it was written into, rather than inventing a second way to describe a board. Raises
    `ValueError` on anything that does not match that template (a context from a different tag, say), the same
    "fail loudly on a shape mismatch" instinct the rest of this package's `learn` subpackage follows.
    """
    m = _BODY_RE.search(context)
    if not m:
        raise ValueError(f"context does not look like a snake_port board: {context[:120]!r}")
    body = [(int(x), int(y)) for x, y in _CELL_RE.findall(m.group("body"))]
    if not body:
        raise ValueError(f"context's body text had no (x,y) cells: {m.group('body')!r}")
    food = (int(m.group("fx")), int(m.group("fy")))
    return SnakeState(body=body, food=food, dir=m.group("dir"))


def snake_board_v1(context: str, question: str) -> dict[str, float]:  # noqa: ARG001 - question unused, see module docstring
    """One board, turned into named, numeric columns -- the structured alternative to hashed TF-IDF this
    module registers under this name. `question` is accepted (the registry's whole function shape, section 3:
    "入力 (文書と質問)") but unused: `snake-move-up-v1` asks the same fixed question of every record, carrying
    no information `tokenize` did not already discard for the same reason (see `features.py`'s own docstring).
    """
    state = parse_context(context)
    hx, hy = state.body[0]
    tail = state.body[-1]
    body_set = set(state.body)
    food_dist_now = float(_manhattan((hx, hy), state.food))

    feats: dict[str, float] = {
        "length": float(len(state.body)),
        "head_x": float(hx),
        "head_y": float(hy),
        "food_x": float(state.food[0]),
        "food_y": float(state.food[1]),
    }
    for d in DELTA:
        feats[f"dir_is_{d}"] = 1.0 if state.dir == d else 0.0

    for d, (dx, dy) in DELTA.items():
        nx, ny = hx + dx, hy + dy
        in_bounds = 0 <= nx < GRID and 0 <= ny < GRID
        will_eat = in_bounds and (nx, ny) == state.food
        blocks_body = in_bounds and (nx, ny) in body_set and not ((nx, ny) == tail and not will_eat)
        safe = in_bounds and not blocks_body

        # Distance to the wall this direction faces, independent of safety: how many cells lie between the
        # current head and the grid's edge along this axis.
        wall_dist = float((GRID - 1 - hx) if dx > 0 else hx if dx < 0 else (GRID - 1 - hy) if dy > 0 else hy)

        if safe:
            body_after = state.body if will_eat else state.body[:-1]
            blocked = {(x, y) for x, y in body_after[1:]}
            open_area = float(_open_area(blocked, (nx, ny)))
            food_dist_after = float(_manhattan((nx, ny), state.food))
            tail_dist = float(_manhattan((nx, ny), tail))
        else:
            # An unsafe direction is never a candidate move, so its "after the move" figures are undefined --
            # a fixed sentinel (worse than any real in-grid value, since GRID*2 exceeds any real Manhattan
            # distance or reachable-cell count on a GRID x GRID board) rather than 0, so a GBDT split can tell
            # "unsafe" apart from "safe but genuinely cramped" (open_area == 0 would otherwise mean both).
            open_area = 0.0
            food_dist_after = float(GRID * 2)
            tail_dist = float(GRID * 2)

        feats[f"{d}_safe"] = 1.0 if safe else 0.0
        feats[f"{d}_open_area"] = open_area
        feats[f"{d}_food_dist_now"] = food_dist_now
        feats[f"{d}_food_dist_after"] = food_dist_after
        feats[f"{d}_tail_dist"] = tail_dist
        feats[f"{d}_wall_dist"] = wall_dist
    return feats


def simple_rule_choice(context: str) -> str:
    """The hand-rule baseline: among the directions `snake_board_v1` marked safe, the one with the largest
    `open_area` (ties broken by the smaller `food_dist_after`, then the smaller `tail_dist`, for a fixed
    answer rather than dict-order luck). Returns a direction name, never "none" -- `snake-move-up-v1` only
    ever logs boards where "up" is legal (`snake_port.sample_states_with_up_legal`), so at least one safe
    direction always exists.
    """
    feats = snake_board_v1(context, "")
    best_dir: str | None = None
    best_key: tuple[float, float, float] | None = None
    for d in DELTA:
        if feats[f"{d}_safe"] < 0.5:
            continue
        key = (feats[f"{d}_open_area"], -feats[f"{d}_food_dist_after"], -feats[f"{d}_tail_dist"])
        if best_key is None or key > best_key:
            best_key, best_dir = key, d
    if best_dir is None:
        raise ValueError("no safe direction found -- not a snake-move-up-v1 board")
    return best_dir


def simple_rule_says_up(context: str) -> bool:
    return simple_rule_choice(context) == "up"


register_features("snake_board_v1", snake_board_v1)
