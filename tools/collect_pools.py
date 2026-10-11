"""learn2: collect large, disjoint pools of real Prismyra experience for two registered tags
(snake-move-up-v1, tweet-sentiment-v1), fast, via concurrent /ask calls so prismyra.schedule.Batcher
coalesces many documents into few passes (DISTILL-RL-DESIGN-v2.md section 3's student needs scale;
RUN-learn1.md section 3.4 only ever collected 800).

Three disjoint slices per tag, carved out of ONE continuing draw so sizes never overlap:
  - train_pool: as many as the largest learning-curve point needs (subsets of a prefix are themselves
    disjoint-free -- the curve re-uses the same pool, just takes a longer prefix)
  - held_out: fixed size, never used for training, used only to measure the student/Prismyra agreement
    lower bound at every curve point (fixed so the curve's x-axis is the only thing that moves)
  - eval: fixed size, gold-labelled (dataset's own label for tweets; a documented heuristic proxy for
    Snake, same one RUN-learn1.md used), used for the "beats Prismyra" condition

Output: one JSONL file per slice per tag, each line {"context", "probabilities", "gold"?}.
Not shipped as part of the package -- a one-off collector for this change's own verification, the
same way tools/collect_and_fit.py (learn1) and tools/audit_sm120.py are.

Usage (on a GPU box, with the real model loaded):
    PRISMYRA_TEST_MODEL=littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l python3 collect_pools.py \
      --out-dir /work/learn2/pools --tweet-parquet /work/learn2/data/tweet_sentiment_test.parquet \
      --snake-train 20000 --snake-held-out 3000 --snake-eval 1000 \
      --tweet-train 9784 --tweet-held-out 1500 --tweet-eval 1000 \
      --workers 48
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
from snake_port import best_direction_heuristic, build_request, sample_states_with_up_legal

TWEET_LABELS = ["negative", "neutral", "positive"]
TWEET_QUESTION = "What sentiment does this tweet express?"
SNAKE_QUESTION = "Is moving up the best of the directions listed above?"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def ask_concurrent(client, items: list[dict], workers: int) -> list[dict]:
    """POST /ask for every item in `items` (each `{"context", "questions"}`) using a thread pool, so
    Starlette dispatches each call to its own worker thread and prismyra.schedule.Batcher (server
    started with batcher=True) sees them arrive close together and coalesces them into shared passes
    -- the "bulk answer" lever this script exists to use, since /ask itself is one-context-at-a-time.
    Returns the parsed JSON body in the same order as `items`.
    """
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


def collect_snake_slice(client, rng: random.Random, n: int, workers: int) -> list[dict]:
    states = sample_states_with_up_legal(rng, n)
    items = []
    meta = []
    for s in states:
        context, questions, cands = build_request(s)
        up_idx = next(i for i, c in enumerate(cands) if c.dir == "up")
        gold_dir = best_direction_heuristic(cands)
        items.append({"context": context, "questions": questions})
        meta.append({"up_qid": f"d{up_idx}", "gold": "yes" if gold_dir == "up" else "no"})
    responses = ask_concurrent(client, items, workers=workers)
    rows = []
    for item, m, resp in zip(items, meta, responses, strict=True):
        probs = resp["answers"][m["up_qid"]]["probabilities"]
        rows.append({"context": item["context"], "probabilities": probs, "gold": m["gold"]})
    return rows


def collect_tweet_slice(client, texts: list[str], labels: list[int], workers: int) -> list[dict]:
    items = [
        {
            "context": t,
            "questions": [{"id": "sentiment", "prompt": TWEET_QUESTION, "kind": "choice", "choices": TWEET_LABELS}],
        }
        for t in texts
    ]
    responses = ask_concurrent(client, items, workers=workers)
    rows = []
    for t, lab, resp in zip(texts, labels, responses, strict=True):
        probs = resp["answers"]["sentiment"]["probabilities"]
        rows.append({"context": t, "probabilities": probs, "gold": TWEET_LABELS[int(lab)]})
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tweet-parquet", required=True)
    p.add_argument("--snake-train", type=int, default=20000)
    p.add_argument("--snake-held-out", type=int, default=3000)
    p.add_argument("--snake-eval", type=int, default=1000)
    p.add_argument("--tweet-train", type=int, default=9784)
    p.add_argument("--tweet-held-out", type=int, default=1500)
    p.add_argument("--tweet-eval", type=int, default=1000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--max-queue", type=int, default=4096)
    p.add_argument("--only", choices=["snake", "tweet", "both"], default="both")
    args = p.parse_args()

    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient

    from prismyra.server import create_app

    print(f"loading {model} (batcher=True, learn_spec=None -- this script does its own bookkeeping) ...", flush=True)
    t_load = time.monotonic()
    app = create_app(model, max_queue=args.max_queue, batcher=True, learn_spec=None)
    client = TestClient(app)
    print(f"  loaded in {time.monotonic() - t_load:.1f}s", flush=True)

    if args.only in ("snake", "both"):
        rng = random.Random(args.seed)
        print(
            f"=== snake: train={args.snake_train} held_out={args.snake_held_out} eval={args.snake_eval} ===", flush=True
        )
        train_rows = collect_snake_slice(client, rng, args.snake_train, args.workers)
        write_jsonl(out_dir / "snake_train_pool.jsonl", train_rows)
        held_rows = collect_snake_slice(client, rng, args.snake_held_out, args.workers)
        write_jsonl(out_dir / "snake_held_out.jsonl", held_rows)
        eval_rows = collect_snake_slice(client, rng, args.snake_eval, args.workers)
        write_jsonl(out_dir / "snake_eval.jsonl", eval_rows)
        print(f"  wrote {len(train_rows)}+{len(held_rows)}+{len(eval_rows)} snake rows to {out_dir}", flush=True)

    if args.only in ("tweet", "both"):
        import pyarrow.parquet as pq

        table = pq.read_table(args.tweet_parquet)
        texts = table.column("text").to_pylist()
        labels = table.column("label").to_pylist()
        n_total = args.tweet_train + args.tweet_held_out + args.tweet_eval
        if n_total > len(texts):
            raise SystemExit(f"need {n_total} tweet rows but only {len(texts)} available in {args.tweet_parquet}")
        idx = list(range(len(texts)))
        random.Random(args.seed + 1).shuffle(idx)
        i0 = args.tweet_train
        i1 = i0 + args.tweet_held_out
        i2 = i1 + args.tweet_eval
        train_idx, held_idx, eval_idx = idx[:i0], idx[i0:i1], idx[i1:i2]
        print(
            f"=== tweet: train={len(train_idx)} held_out={len(held_idx)} eval={len(eval_idx)} (pool={len(texts)}) ===",
            flush=True,
        )
        train_rows = collect_tweet_slice(
            client, [texts[i] for i in train_idx], [labels[i] for i in train_idx], args.workers
        )
        write_jsonl(out_dir / "tweet_train_pool.jsonl", train_rows)
        held_rows = collect_tweet_slice(
            client, [texts[i] for i in held_idx], [labels[i] for i in held_idx], args.workers
        )
        write_jsonl(out_dir / "tweet_held_out.jsonl", held_rows)
        eval_rows = collect_tweet_slice(
            client, [texts[i] for i in eval_idx], [labels[i] for i in eval_idx], args.workers
        )
        write_jsonl(out_dir / "tweet_eval.jsonl", eval_rows)
        print(f"  wrote {len(train_rows)}+{len(held_rows)}+{len(eval_rows)} tweet rows to {out_dir}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
