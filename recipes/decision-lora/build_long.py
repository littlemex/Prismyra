"""Long-context training set (v2): RACE-train and BoolQ-train items buried in 4k-12k tokens of other TRAIN text.

Disjoint from train_v1's RACE articles (keys[:1500] there, the same seeded shuffle here) and from every eval item.
At most two questions per article: a long row costs its whole prefill, so questions are spread over more contexts.
"""

import hashlib
import json
import os
import random

#: Where the rows are written, and the evaluation items every row is deduplicated against (json item files and jsonl
#: System One rows). Both come from the environment so the recipe carries no machine's paths.
DATA = os.environ.get("DATA_DIR", "data")
EVAL = [p for p in os.environ.get("EVAL_ITEMS", "").split(":") if p]
from datasets import load_dataset

rng = random.Random(20260924)
L = "ABCDEFGH"
seen = set()
out = []
for path in [p for p in EVAL if p.endswith(".json")]:
    for it in json.load(open(path))["items"]:
        seen.add(hashlib.sha1(it["context"][:2000].encode()).hexdigest())
        for q in it["questions"]:
            seen.add(hashlib.sha1(q["question"].strip().lower().encode()).hexdigest())
ok = lambda s: hashlib.sha1(s.strip().lower().encode()).hexdigest() not in seen
race = load_dataset("ehovy/race", "all", split="train")
arts = {}
for r in race:
    arts.setdefault(r["article"], []).append(r)
keys = list(arts)
rng.shuffle(keys)  # identical to build_train.py's shuffle: keys[:1500] are v1's
pool = keys[4000:]


def bury(core):
    target = rng.randint(16000, 48000)
    before, after, n = [], [], len(core)
    for o in rng.sample(pool, 80):
        if n >= target:
            break
        (before if rng.random() < 0.5 else after).append(o)
        n += len(o)
    return "\n\n".join(before + [core] + after)


for a in keys[1500:2900]:
    ctx = bury(a)
    for r in rng.sample(arts[a], min(2, len(arts[a]))):
        if ok(r["question"]):
            out.append(
                {
                    "context": ctx,
                    "kind": "choice",
                    "question": r["question"].strip(),
                    "options": r["options"],
                    "gold": L.index(r["answer"]),
                    "family": "race_buried",
                }
            )
bq = list(load_dataset("google/boolq", split="train"))
rng.shuffle(bq)
for r in bq[3000:3400]:
    if ok(r["question"]):
        out.append(
            {
                "context": bury(r["passage"]),
                "kind": "boolean",
                "question": r["question"].strip() + "?",
                "gold": int(r["answer"]),
                "family": "boolq_buried",
            }
        )
for a in keys[2900:3100]:
    r = rng.choice(arts[a])
    if ok(r["question"]):
        out.append(
            {
                "context": a,
                "kind": "choice",
                "question": r["question"].strip(),
                "options": r["options"],
                "gold": L.index(r["answer"]),
                "family": "race",
            }
        )
rng.shuffle(out)
json.dump(out, open(os.path.join(DATA, "train_long_v2.json"), "w"))
import collections

print(len(out), collections.Counter(x["family"] for x in out))
