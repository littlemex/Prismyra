"""The 7,018-row subset of the mixture that was trained (v1): per-family caps, seeded."""

import collections
import json
import os
import random

#: Where the rows are written, and the evaluation items every row is deduplicated against (json item files and jsonl
#: System One rows). Both come from the environment so the recipe carries no machine's paths.
DATA = os.environ.get("DATA_DIR", "data")
EVAL = [p for p in os.environ.get("EVAL_ITEMS", "").split(":") if p]
d = json.load(open(os.path.join(DATA, "train_v1.json")))
rng = random.Random(7)
want = {
    "race": 1500,
    "race_buried": 1318,
    "boolq": 800,
    "mmlu": 800,
    "sciq": 500,
    "qnli": 500,
    "tweet_offensive": 500,
    "synth_logic": 500,
    "synth_signoff": 300,
    "synth_timeliness": 300,
}
by = collections.defaultdict(list)
for x in d:
    by[x["family"]].append(x)
out = [x for f, n in want.items() for x in rng.sample(by[f], min(n, len(by[f])))]
rng.shuffle(out)
json.dump(out, open(os.path.join(DATA, "train_v1_7k.json"), "w"))
print(len(out), collections.Counter(x["family"] for x in out))
