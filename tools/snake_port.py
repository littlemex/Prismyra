"""A minimal, direct port of `examples`'s webapp Snake demo (`demo4/webapp/snake.js`, not shipped in this
repo) used only to generate realistic board states and the exact context/question text Prismyra was asked in
that demo -- for learn1's own stage-2 student-training check on a "structured input" tag. Not part of the
package; a one-off data generator for this change's own verification.

Ported, not reused, because the original is JavaScript running in a browser; the game rules themselves
(legality, the three per-direction facts, the context template, the question wording) are copied exactly so a
registered tag's `question` text matches what a real deployment of that demo would send.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

GRID = 12
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


@dataclass
class SnakeState:
    body: list[tuple[int, int]]  # head first
    food: tuple[int, int]
    dir: str


def legal_options(state: SnakeState) -> list[str]:
    hx, hy = state.body[0]
    body_set = set(state.body)
    tail = state.body[-1]
    out = []
    for d, (dx, dy) in DELTA.items():
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            continue
        will_eat = (nx, ny) == state.food
        body_blocks = (nx, ny) in body_set and not ((nx, ny) == tail and not will_eat)
        if body_blocks:
            continue
        out.append(d)
    return out


def _open_area(blocked: set[tuple[int, int]], start: tuple[int, int]) -> int:
    seen = {start}
    q = deque([start])
    count = 0
    while q:
        x, y = q.popleft()
        count += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
                continue
            if (nx, ny) in seen or (nx, ny) in blocked:
                continue
            seen.add((nx, ny))
            q.append((nx, ny))
    return count


def _runway(blocked: set[tuple[int, int]], start: tuple[int, int], d: str) -> int:
    dx, dy = DELTA[d]
    x, y = start
    steps = 0
    for _ in range(GRID * 2):
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            break
        if (nx, ny) in blocked:
            break
        steps += 1
        x, y = nx, ny
    return steps


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class Candidate:
    dir: str
    food_closer: bool
    open_area: int
    runway: int


def candidates(state: SnakeState) -> list[Candidate]:
    hx, hy = state.body[0]
    old_dist = _manhattan((hx, hy), state.food)
    out = []
    for d in legal_options(state):
        dx, dy = DELTA[d]
        new_head = (hx + dx, hy + dy)
        will_eat = new_head == state.food
        body_after = state.body if will_eat else state.body[:-1]
        blocked = {(x, y) for (x, y) in body_after[1:]}
        new_dist = _manhattan(new_head, state.food)
        out.append(
            Candidate(
                dir=d,
                food_closer=new_dist < old_dist,
                open_area=_open_area(blocked, new_head),
                runway=_runway(blocked, new_head, d),
            )
        )
    return out


def build_request(state: SnakeState) -> tuple[str, list[dict], list[Candidate]]:
    """Exactly `Snake.buildRequest` in the original: the context, the question list, and the candidate table
    the gold label below is computed from."""
    cands = candidates(state)
    body_text = ", ".join(f"({x},{y})" for x, y in state.body)
    header = "  #  direction  food_closer  open_area  runway"
    lines = [header]
    for i, c in enumerate(cands):
        lines.append(f"{i:>3}  {c.dir:<9}  {'yes' if c.food_closer else 'no':<11}  {c.open_area:>9}  {c.runway:>6}")
    context = (
        f"A game of Snake on a {GRID}x{GRID} grid. Columns run 0 to {GRID - 1} left to right, rows 0 to "
        f"{GRID - 1} top to bottom. The snake's body, head first: {body_text}. It is currently moving "
        f"{state.dir}. The food is at ({state.food[0]},{state.food[1]}). Moving into a wall or into the "
        f"snake's own body is not offered below.\n\n"
        f"The legal directions, with what each one would do:\n\n{chr(10).join(lines)}\n\n"
        f"food_closer is whether that move shortens the distance to the food. open_area is how many empty "
        f"cells are reachable from the new head position (more is safer: a smaller open area can trap the "
        f"snake later even if this move itself is safe). runway is how many empty cells lie straight ahead "
        f"in that direction before a wall or the snake's own body."
    )
    questions = [
        {"id": f"d{i}", "kind": "boolean", "prompt": f"Is moving {c.dir} the best of the directions listed above?"}
        for i, c in enumerate(cands)
    ]
    return context, questions, cands


def best_direction_heuristic(cands: list[Candidate]) -> str:
    """Not an external label -- Snake has none. A documented stand-in "gold" for this change's own offline
    student-admission check only: rank by `open_area` first (the demo's own README calls it "more is safer"),
    `food_closer` second, `runway` third. Used only to measure the student/this heuristic's agreement in
    learn1's own verification, never as a claim about how good Prismyra's own Snake play is."""
    return max(cands, key=lambda c: (c.open_area, c.food_closer, c.runway)).dir


def random_state(rng: random.Random, min_body: int = 1, max_body: int = 6) -> SnakeState:
    """A random, self-consistent board: a straight snake of random length and heading placed so every body
    cell is in bounds, and food placed anywhere empty. Not every state a real game would reach, but every one
    is legal (no self-overlap, no out-of-bounds), which is all the context template and the candidate
    computation above need to be exercised realistically."""
    length = rng.randint(min_body, max_body)
    heading = rng.choice(list(DELTA))
    dx, dy = DELTA[heading]
    # Head position chosen so the whole body (extending backwards from the head, opposite `heading`) fits.
    back_dx, back_dy = -dx * (length - 1), -dy * (length - 1)
    hx = rng.randint(max(0, -back_dx), min(GRID - 1, GRID - 1 - back_dx))
    hy = rng.randint(max(0, -back_dy), min(GRID - 1, GRID - 1 - back_dy))
    body = [(hx + dx * (-i), hy + dy * (-i)) for i in range(length)]
    occupied = set(body)
    while True:
        food = (rng.randint(0, GRID - 1), rng.randint(0, GRID - 1))
        if food not in occupied:
            break
    return SnakeState(body=body, food=food, dir=heading)


def sample_states_with_up_legal(rng: random.Random, n: int, max_tries: int = 50_000) -> list[SnakeState]:
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        s = random_state(rng)
        if "up" in legal_options(s):
            out.append(s)
    return out
