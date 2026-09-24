"""v3 extra families (train splits), rendered like the Kev suite rows (state + instruction + labelled options).

Held out as before: emotion, PAWS, exception-clause policies -- and, to keep "unseen family" meaningful, no sentiment
and no paraphrase task of any source is added either.
"""

import json
import os
import random

#: Where the rows are written, and the evaluation items every row is deduplicated against (json item files and jsonl
#: System One rows). Both come from the environment so the recipe carries no machine's paths.
DATA = os.environ.get("DATA_DIR", "data")
EVAL = [p for p in os.environ.get("EVAL_ITEMS", "").split(":") if p]
from datasets import load_dataset

rng = random.Random(20260925)
out = []


def ch(ctx, q, opts, gold, fam):
    out.append({"context": ctx, "kind": "choice", "question": q, "options": opts, "gold": gold, "family": fam})


def take(ds, n):
    ds = list(ds)
    rng.shuffle(ds)
    return ds[:n]


for r in take(load_dataset("fancyzhx/ag_news", split="train"), 400):
    ch(
        r["text"],
        "Which topic is this news item about?",
        ["World", "Sports", "Business", "Sci/Tech"],
        r["label"],
        "ag_news",
    )
for r in take(load_dataset("fancyzhx/dbpedia_14", split="train"), 300):
    names = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork",
    ]
    ch(r["title"] + ". " + r["content"], "What kind of entity does this text describe?", names, r["label"], "dbpedia")
for r in take(load_dataset("nyu-mll/glue", "mnli", split="train"), 400):
    ch(
        json.dumps({"premise": r["premise"], "hypothesis": r["hypothesis"]}, ensure_ascii=False, sort_keys=True),
        "How does the hypothesis relate to the premise?",
        [
            "entailment: The premise guarantees the hypothesis",
            "neutral: The premise neither guarantees nor rules out the hypothesis",
            "contradiction: The premise rules out the hypothesis",
        ],
        r["label"],
        "mnli",
    )
for r in take(load_dataset("tau/commonsense_qa", split="train"), 300):
    ch(
        json.dumps({"question": r["question"]}, ensure_ascii=False),
        "Which option best answers the question?",
        [f"{l.lower()}: {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"])],
        r["choices"]["label"].index(r["answerKey"]),
        "csqa",
    )
for r in take(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"), 300):
    if r["answerKey"] in r["choices"]["label"]:
        ch(
            json.dumps({"question": r["question"]}, ensure_ascii=False),
            "Which option correctly answers the question?",
            [f"{l.lower()}: {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"])],
            r["choices"]["label"].index(r["answerKey"]),
            "arc",
        )
for r in take(load_dataset("allenai/openbookqa", "main", split="train"), 300):
    ch(
        json.dumps({"question": r["question_stem"]}, ensure_ascii=False),
        "Which option answers the science question?",
        [f"{l.lower()}: {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"])],
        r["choices"]["label"].index(r["answerKey"]),
        "obqa",
    )
for r in take(load_dataset("Rowan/hellaswag", split="train"), 300):
    ch(r["ctx"], "Which continuation is the most plausible?", r["endings"], int(r["label"]), "hellaswag")
rng.shuffle(out)
json.dump(out, open(os.path.join(DATA, "train_div_v3.json"), "w"))
import collections

print(len(out), collections.Counter(x["family"] for x in out))
v1 = json.load(open(os.path.join(DATA, "train_v1_7k.json")))
lg = json.load(open(os.path.join(DATA, "train_long_v2.json")))[:1800]
allv = v1 + lg + out
rng.shuffle(allv)
json.dump(allv, open(os.path.join(DATA, "train_v3.json"), "w"))
print("v3", len(allv))
