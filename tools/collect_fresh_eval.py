"""learn4: a Snake evaluation slice that has never been looked at before the admission bar was lowered.

DISTILL-RL-DESIGN-v2.md section 3's own condition 2 (held-out agreement) and condition 1 (gold accuracy) were
re-judged against a new, lower bar (0.90/0.90, prismyra.learn.spec.DEFAULT_MIN_MATCH_RATE/
DEFAULT_MIN_ACCURACY_RATIO) *after* learn3's own run (RUN-learn3.md) had already seen the old eval/held-out
numbers at the old bar (0.97). Re-checking the same old numbers against a bar chosen after seeing them would
be judging on a fixture the judgment itself was tuned to -- this script instead draws a slice of Snake boards
excluded, by exact context text, from every board learn2/learn3 already drew (`/work/learn2/pools/snake_*.jsonl`
on the shared PVC, the actual files those two runs trained and evaluated on -- the ground truth of "already
used", not a replayed random-number stream), so the re-judgment in `tools/rejudge_fresh_eval.py` sees problems
neither the model's own admission history nor this change's own development ever touched.

A *fresh seed* alone would not have been enough: RUN-learn2.md section 2.1 already records that drawing 20,000
boards from this task's finite state space (a 12x12 grid, body length 1-6) produces some exact-text duplicates
even within one slice, so two different seeds' draws can coincide too. The guarantee here is an explicit
exclusion filter against the real, on-disk contents of every slice learn2/learn3 ever touched, checked while
drawing (not asserted afterwards and hoped for).

Usage (on a GPU box, with the real model loaded, same shape as collect_pools.py):
    PRISMYRA_TEST_MODEL=littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l python3 collect_fresh_eval.py \
      --out /work/learn4/snake_fresh_eval.jsonl --exclude-from /work/learn2/pools --n 1100 --workers 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snake_port import best_direction_heuristic, build_request, legal_options, random_state

# A new seed is not what makes this slice disjoint (see the module docstring) -- it only has to differ from
# collect_pools.py's own 20261011/20261012 so this run's own draw order is not a coincidental prefix of theirs.
FRESH_SEED = 20261011_04  # "_04": the 4th run against this task's Snake tag


def load_excluded_contexts(pools_dir: Path) -> set[str]:
    """Every context any of learn2's three Snake pool files actually holds -- train_pool (pre-dedup: the
    file collect_pools.py wrote, not the post-hoc-deduped subset train_sweep.py trains on; excluding a
    superset of what training/evaluation touched is strictly safer, never a mistake), held_out, and eval."""
    excluded: set[str] = set()
    names = ["snake_train_pool.jsonl", "snake_held_out.jsonl", "snake_eval.jsonl"]
    for name in names:
        path = pools_dir / name
        if not path.is_file():
            raise SystemExit(f"expected {path} to exist (learn2's own pool file)")
        n_before = len(excluded)
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                excluded.add(json.loads(stripped)["context"])
        print(f"  {name}: +{len(excluded) - n_before} contexts (running total {len(excluded)})", flush=True)
    return excluded


def sample_fresh_states(rng: random.Random, n: int, excluded: set[str], max_tries: int = 2_000_000):
    """`snake_port.sample_states_with_up_legal`'s own loop, with one more rejection condition: a board whose
    exact context text is already in `excluded` (a previous run's slice) or has already been accepted in
    this run's own output -- not just "up" being a legal move."""
    out = []
    seen_this_run: set[str] = set()
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        s = random_state(rng)
        if "up" not in legal_options(s):
            continue
        context, _, _ = build_request(s)
        if context in excluded or context in seen_this_run:
            continue
        seen_this_run.add(context)
        out.append(s)
    return out, tries


def ask_concurrent(client, items: list[dict], workers: int) -> list[dict]:
    results: list[dict | None] = [None] * len(items)

    def one(i: int) -> None:
        item = items[i]
        resp = client.post("/ask", json={"context": item["context"], "questions": item["questions"]})
        if resp.status_code != 200:
            raise RuntimeError(f"item {i}: {resp.status_code} {resp.text[:300]}")
        results[i] = resp.json()

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, i) for i in range(len(items))]
        for f in futs:
            f.result()
    elapsed = time.monotonic() - t0
    rate = len(items) / elapsed if elapsed > 0 else float("inf")
    print(
        f"  [ask_concurrent] {len(items)} items in {elapsed:.1f}s ({rate:.1f} items/s, workers={workers})", flush=True
    )
    return results  # type: ignore[return-value]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-from", required=True, help="directory holding learn2's snake_*.jsonl pool files")
    p.add_argument("--n", type=int, default=1100, help="fresh slice size (the task asks for >= 1000)")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--max-queue", type=int, default=4096)
    p.add_argument("--seed", type=int, default=FRESH_SEED)
    args = p.parse_args()

    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    print(f"loading excluded contexts from {args.exclude_from} ...", flush=True)
    excluded = load_excluded_contexts(Path(args.exclude_from))

    rng = random.Random(args.seed)
    print(f"drawing {args.n} fresh states excluded against {len(excluded)} previously-used contexts ...", flush=True)
    t0 = time.monotonic()
    fresh_states, tries = sample_fresh_states(rng, args.n, excluded)
    if len(fresh_states) < args.n:
        raise SystemExit(f"only found {len(fresh_states)}/{args.n} fresh states in {tries} tries")
    print(f"  drew {len(fresh_states)} states in {tries} tries, {time.monotonic() - t0:.1f}s", flush=True)

    fresh_requests = [build_request(s) for s in fresh_states]
    # Re-verify explicitly rather than trust the loop above silently: every context here must be new.
    fresh_contexts = [ctx for ctx, _, _ in fresh_requests]
    assert len(set(fresh_contexts)) == len(fresh_contexts), "duplicate within the fresh slice itself"
    assert not (set(fresh_contexts) & excluded), "a fresh context collides with an excluded one"
    print(f"  confirmed 0/{args.n} collisions with the {len(excluded)} excluded contexts", flush=True)

    from fastapi.testclient import TestClient

    from prismyra.server import create_app

    print(f"loading {model} (batcher=True, learn_spec=None) ...", flush=True)
    t_load = time.monotonic()
    app = create_app(model, max_queue=args.max_queue, batcher=True, learn_spec=None)
    client = TestClient(app)
    print(f"  loaded in {time.monotonic() - t_load:.1f}s", flush=True)

    items = []
    meta = []
    for context, questions, cands in fresh_requests:
        up_idx = next(i for i, c in enumerate(cands) if c.dir == "up")
        gold_dir = best_direction_heuristic(cands)
        items.append({"context": context, "questions": questions})
        meta.append({"up_qid": f"d{up_idx}", "gold": "yes" if gold_dir == "up" else "no"})

    responses = ask_concurrent(client, items, workers=args.workers)
    rows = []
    for item, m, resp in zip(items, meta, responses, strict=True):
        probs = resp["answers"][m["up_qid"]]["probabilities"]
        # Same shape as learn2's own pool files (tools/collect_pools.py's collect_snake_slice) -- one slice
        # that doubles as both the held-out agreement check and the gold-accuracy check in
        # tools/rejudge_fresh_eval.py, instead of two separately-collected files.
        rows.append({"context": item["context"], "probabilities": probs, "gold": m["gold"]})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} fresh eval rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
