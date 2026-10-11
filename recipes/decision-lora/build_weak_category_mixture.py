"""Weak-category mixture: four licence-clear families for two judgment types the base adapter had
never seen a training example of -- sentiment and causal/multi-step reasoning. Rendered the same way
as build_families.py's rows (state + instruction + lettered options), mixed into an existing training
pool at a small, fixed fraction rather than appended at full size.

Why a small fraction and not the full four families: an earlier attempt mixed these families in at
~7% of the combined pool with no further change, trained a LoRA, and measured it against a held-out
set the base model is not allowed to regress on (a safety-style set with sentiment- and borderline-
confidence items of its own, out of scope for this recipe and not published). That run improved the
two new judgment types but also measured a drop on several items of the held-out set, concentrated on
the items closest in topic (sentiment) or confidence margin (borderline, regardless of topic) to the
new rows -- not a leak (the sets share no row), but ordinary interference from the only change between
the old pool and the new one: every row the old pool trains with has a self-distillation target (the
model's own prior-revision output, so the step is pulled toward "stay yourself" as well as "fit the
label"); the newly mixed rows had none, so their gradient had nothing pulling it back. Halving the
mixed-in fraction, with no other change, removed the regression and kept the improvement on the two
new judgment types; see the recipe's PR for the measured numbers both ways.

Licences (dataset card, checked at the HF dataset page):
  zeroshot/twitter-financial-news-sentiment  MIT
  cardiffnlp/tweet_eval, config "sentiment"  CC BY 3.0 Unported (the "offensive" config is a different,
                                              unrelated split of the same dataset repo -- not reused here)
  pkavumba/balanced-copa                     CC BY 4.0
  allenai/social_i_qa                        CC BY 4.0

Usage:
  export DATA_DIR=data EVAL_ITEMS=items/race150.json:items/boolq400.json:...
  python build_weak_category_mixture.py BASE_POOL.json --fraction 0.04 --out mixed.json
"""

import argparse
import collections
import json
import os
import random

from datasets import load_dataset

DATA = os.environ.get("DATA_DIR", "data")
EVAL = [p for p in os.environ.get("EVAL_ITEMS", "").split(":") if p]

SEED = 20261010


def ch(ctx, q, opts, gold, fam):
    return {"context": ctx, "kind": "choice", "question": q, "options": opts, "gold": gold, "family": fam}


def build_financial_sentiment(rng):
    opts = [
        "bearish: the text expresses a negative view",
        "bullish: the text expresses a positive view",
        "neutral: the text expresses neither",
    ]
    rows = list(load_dataset("zeroshot/twitter-financial-news-sentiment", split="train"))
    rng.shuffle(rows)
    return [ch(r["text"], "Which option best describes the sentiment of this text?", opts, r["label"], "kd_finsent") for r in rows]


def build_tweet_sentiment(rng):
    opts = ["negative: the tweet expresses a negative sentiment", "neutral: the tweet expresses neither", "positive: the tweet expresses a positive sentiment"]
    rows = list(load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train"))
    rng.shuffle(rows)
    return [ch(r["text"], "Which option best describes the sentiment of this tweet?", opts, r["label"], "kd_tweetsent") for r in rows]


def build_balanced_copa(rng):
    rows = list(load_dataset("pkavumba/balanced-copa", split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows:
        word = "cause" if r["question"] == "cause" else "effect"
        q = f"What is the most plausible {word} of the premise?"
        out.append(ch(r["premise"], q, [r["choice1"], r["choice2"]], r["label"], "kd_copa"))
    return out


def build_social_iqa(rng):
    rows = list(load_dataset("allenai/social_i_qa", split="train"))
    rng.shuffle(rows)
    out = []
    for r in rows:
        opts = [r["answerA"], r["answerB"], r["answerC"]]
        gold = int(r["label"]) - 1  # 1-indexed in the source
        out.append(ch(r["context"], r["question"], opts, gold, "kd_siqa"))
    return out


def dedup_against_eval(rows):
    """Exact-match guard against the files this recipe is told to never overlap with. The internal
    round that produced this recipe also ran a 3-stage check (exact, 13-gram, short-string containment)
    against the project's full training/calibration/lock corpus and found zero drops for these four
    families (they are new HF sources the project had not touched before) -- this exact-match pass is
    the part of that check that travels with the recipe without the rest of the corpus."""
    seen = set()
    for p in EVAL:
        if not os.path.exists(p):
            continue
        try:
            items = json.load(open(p))
        except (json.JSONDecodeError, UnicodeDecodeError):
            items = [json.loads(l) for l in open(p)]
        for x in items:
            if "context" in x:
                seen.add(x["context"])
    before = len(rows)
    rows = [r for r in rows if r["context"] not in seen]
    if before != len(rows):
        print(f"dedup: dropped {before - len(rows)} of {before} rows (exact context match against EVAL_ITEMS)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_pool", help="existing training pool (JSON list) to mix the new families into")
    ap.add_argument("--fraction", type=float, default=0.04, help="target fraction of the combined pool the new families should be (default 0.04; 0.0738 regressed a held-out safety set, see module docstring)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(SEED)
    base = json.load(open(args.base_pool))

    by_fam = {}
    for name, builder in [
        ("kd_finsent", build_financial_sentiment),
        ("kd_tweetsent", build_tweet_sentiment),
        ("kd_copa", build_balanced_copa),
        ("kd_siqa", build_social_iqa),
    ]:
        rows = dedup_against_eval(builder(random.Random(SEED)))
        by_fam[name] = rows
        print(name, "available after dedup:", len(rows))

    # target_new / (len(base) + target_new) == args.fraction  =>  target_new == fraction * len(base) / (1 - fraction)
    target_new = round(args.fraction * len(base) / (1 - args.fraction))
    per_family = target_new // 4
    sampled = []
    for name, rows in by_fam.items():
        take = min(per_family, len(rows))
        sampled.extend(rows[:take])
        print(name, "took", take)

    combined = base + sampled  # base rows keep their original index so a row-indexed teacher/distillation
    # target file for the base pool still lines up by position
    json.dump(combined, open(args.out, "w"))
    report = {
        "base_rows": len(base),
        "new_rows": len(sampled),
        "new_by_family": dict(collections.Counter(r["family"] for r in sampled)),
        "total_rows": len(combined),
        "new_fraction_pct": round(100 * len(sampled) / len(combined), 2),
    }
    print(report)


if __name__ == "__main__":
    main()
