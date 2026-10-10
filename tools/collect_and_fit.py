"""learn1's real-machine check for stage 2 (DISTILL-RL-DESIGN-v2.md section 3): collect real experience from
the real model under two registered tags -- one short classification (tweet sentiment, a licensed, held-out
split another project's own training never touched), one structured input (a Snake board, ported from the
`examples`-style game demo in `demo4/`) -- then fit and evaluate an offline student for each.

Not shipped as part of the package; a one-off script for this change's own verification, the same way
`tools/audit_sm120.py` is.

Train and eval populations are disjoint by construction, not by hoping two random samples do not collide:
train goes through the real `/ask` endpoint (tagged, logged to disk by `prismyra.learn`, exactly the path a
real deployment would use) and eval goes straight through `engine.ask` on the *same* loaded model (bypassing
the server and the log entirely), on index ranges that never overlap the train sample's.

Usage: run on a box with the real model and a GPU.

    PRISMYRA_TEST_MODEL=littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l python3 tools/collect_and_fit.py \
      --tweet-parquet /work/learn1/data/tweet_sentiment_test.parquet --out-dir /work/learn1/student_run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snake_port import best_direction_heuristic, build_request, sample_states_with_up_legal

TWEET_LABELS = ["negative", "neutral", "positive"]
TWEET_QUESTION = "What sentiment does this tweet express?"
SNAKE_QUESTION = "Is moving up the best of the directions listed above?"


def write_spec(spec_path: Path) -> None:
    entries = [
        {
            "task": "tweet-sentiment-v1",
            "question": TWEET_QUESTION,
            "options": TWEET_LABELS,
            "kind": "choice",
            "normalize": "strip+lower",
            "retain_days": 30,
            "keep_hidden": False,
            "eval_set": "tweet_sentiment_eval.jsonl",
        },
        {
            "task": "snake-move-up-v1",
            "question": SNAKE_QUESTION,
            "options": ["no", "yes"],
            "kind": "boolean",
            "normalize": "none",
            "retain_days": 30,
            "keep_hidden": False,
            "eval_set": "snake_eval.jsonl",
        },
    ]
    spec_path.write_text(json.dumps(entries, indent=2))


def collect_tweets(client, engine, parquet_path: str, n_train: int, n_eval: int, out_dir: Path, seed: int) -> None:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    train_idx, eval_idx = idx[:n_train], idx[n_train : n_train + n_eval]

    print(f"[tweet] logging {len(train_idx)} train rows through /ask (tagged) ...", flush=True)
    for i in train_idx:
        row = df.iloc[i]
        resp = client.post(
            "/ask",
            json={
                "context": str(row["text"]),
                "questions": [{"id": "sentiment", "prompt": TWEET_QUESTION, "kind": "choice", "choices": TWEET_LABELS}],
            },
        )
        assert resp.status_code == 200, resp.text

    print(f"[tweet] building {len(eval_idx)} eval rows via engine.ask directly (never logged) ...", flush=True)
    from prismyra import Choice

    eval_rows = []
    for i in eval_idx:
        row = df.iloc[i]
        q = Choice(id="sentiment", prompt=TWEET_QUESTION, choices=TWEET_LABELS)
        result = engine.ask(str(row["text"]), [q])
        gold = TWEET_LABELS[int(row["label"])]
        eval_rows.append(
            {
                "context": str(row["text"]),
                "gold": gold,
                "options": TWEET_LABELS,
                "prismyra_probabilities": dict(result["sentiment"].probabilities),
            }
        )
    (out_dir / "tweet_sentiment_eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")


def collect_snake(client, engine, n_train: int, n_eval: int, out_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    train_states = sample_states_with_up_legal(rng, n_train)
    eval_states = sample_states_with_up_legal(rng, n_eval)  # same rng, continues past the train draws: disjoint

    print(f"[snake] logging {len(train_states)} train states through /ask (tagged) ...", flush=True)
    for s in train_states:
        context, questions, _ = build_request(s)
        resp = client.post("/ask", json={"context": context, "questions": questions})
        assert resp.status_code == 200, resp.text

    print(f"[snake] building {len(eval_states)} eval states via engine.ask directly (never logged) ...", flush=True)
    from prismyra import Boolean

    eval_rows = []
    for s in eval_states:
        context, questions, cands = build_request(s)
        up_q = next(q for q in questions if q["prompt"] == SNAKE_QUESTION)
        gold_dir = best_direction_heuristic(cands)
        gold = "yes" if gold_dir == "up" else "no"
        result = engine.ask(context, [Boolean(id=up_q["id"], prompt=up_q["prompt"])])
        eval_rows.append(
            {
                "context": context,
                "gold": gold,
                "options": ["no", "yes"],
                "prismyra_probabilities": dict(result[up_q["id"]].probabilities),
            }
        )
    (out_dir / "snake_eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tweet-parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "learn.json"
    write_spec(spec_path)

    from fastapi.testclient import TestClient

    from prismyra.server import create_app

    print(f"loading {model} ...", flush=True)
    app = create_app(model, max_queue=64, learn_spec=str(spec_path), learn_log_dir=str(out_dir / "experience"))
    client = TestClient(app)
    engine = app.state.engine

    collect_tweets(client, engine, args.tweet_parquet, args.n_train, args.n_eval, out_dir, args.seed)
    collect_snake(client, engine, args.n_train, args.n_eval, out_dir, args.seed + 1)

    print("learn stats:", client.get("/stats").json()["learn"], flush=True)

    import prismyra
    from prismyra.learn.fit import run

    for task in ("tweet-sentiment-v1", "snake-move-up-v1"):
        print(f"\n=== fitting {task} ===", flush=True)
        artifact = run(
            spec_path=spec_path,
            task=task,
            log_dir=str(out_dir / "experience"),
            package_version=prismyra.__version__,
            backbone=model,
            dims=256,
            held_out_fraction=0.25,
            min_match_rate=0.7,
            accuracy_margin=0.0,
            num_threads=4,
            seed=0,
            num_boost_round=100,
            min_data_in_leaf=5,
        )
        out_path = out_dir / f"{task}.json"
        out_path.write_text(json.dumps(artifact.to_json(), indent=2))
        print(f"{task}: admitted={artifact.admitted} reasons={artifact.reasons}")
        print(f"  metrics: {json.dumps(artifact.metrics, indent=2)[:2000]}")
        print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
