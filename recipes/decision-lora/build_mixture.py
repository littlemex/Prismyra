"""Training data for the Prismyra decision LoRA. Train splits only; three families are held out entirely.

Every example is rendered exactly as the evaluation renders it (e1/e6 items: context + one lettered Choice, or a
Boolean), so what is learned is the read-out Prismyra performs. Held out (never generated, never loaded):
emotion, PAWS, and any policy with an exception clause -- the "unseen family" slice of the Kev suite.
"""

import datetime
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
out = []
seen = set()
# never train on anything that appears in an evaluation item
for path in EVAL:
    if path.endswith(".jsonl"):  # System One rows: the state is what must not recur
        for l in open(path):
            r = json.loads(l)
            seen.add(hashlib.sha1(json.dumps(r["state"], sort_keys=True).lower().encode()).hexdigest())
        continue
    for it in json.load(open(path))["items"]:
        for q in it["questions"]:
            seen.add(hashlib.sha1(q["question"].strip().lower().encode()).hexdigest())


def ok(key):
    h = hashlib.sha1(key.strip().lower().encode()).hexdigest()
    return h not in seen


def choice(ctx, question, options, gold, fam):
    out.append(
        {"context": ctx, "kind": "choice", "question": question, "options": options, "gold": gold, "family": fam}
    )


def boolean(ctx, question, gold, fam):
    out.append({"context": ctx, "kind": "boolean", "question": question, "gold": int(bool(gold)), "family": fam})


# RACE train (middle + high), short and buried in other TRAIN articles
race = load_dataset("ehovy/race", "all", split="train")
arts = {}
for r in race:
    arts.setdefault(r["article"], []).append(r)
keys = list(arts)
rng.shuffle(keys)
for a in keys[:1100]:
    for r in arts[a]:
        if ok(r["question"]):
            choice(a, r["question"].strip(), r["options"], L.index(r["answer"]), "race")
for a in keys[1100:1500]:  # buried: 2k..12k tokens (approximated as 4 chars per token) of other train articles
    target = rng.randint(8000, 48000)
    before, after, n = [], [], len(a)
    for o in rng.sample(keys[1500:], 60):
        if n >= target:
            break
        (before if rng.random() < 0.5 else after).append(o)
        n += len(o)
    ctx = "\n\n".join(before + [a] + after)
    for r in arts[a]:
        if ok(r["question"]):
            choice(ctx, r["question"].strip(), r["options"], L.index(r["answer"]), "race_buried")
# BoolQ train
for r in rng.sample(list(load_dataset("google/boolq", split="train")), 2000):
    if ok(r["question"]):
        boolean(r["passage"], r["question"].strip() + "?", r["answer"], "boolq")
# Kev-suite families that ARE trained on (train splits); rendered like the suite
for r in rng.sample(list(load_dataset("cais/mmlu", "all", split="auxiliary_train")), 2000):
    st = json.dumps({"question": r["question"]}, ensure_ascii=False, sort_keys=True)
    if ok(st):
        choice(
            st,
            "Which option correctly answers the question?",
            [f"{k}: {v}" for k, v in zip("abcd", r["choices"])],
            r["answer"],
            "mmlu",
        )
for r in rng.sample(list(load_dataset("allenai/sciq", split="train")), 1500):
    opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
    perm = list(range(4))
    rng.shuffle(perm)
    st = json.dumps({"question": r["question"]}, ensure_ascii=False, sort_keys=True)
    if ok(st):
        choice(
            st,
            "Which option answers the science question?",
            [f"{k}: {opts[p]}" for k, p in zip("abcd", perm)],
            perm.index(0),
            "sciq",
        )
for r in rng.sample(list(load_dataset("nyu-mll/glue", "qnli", split="train")), 1500):
    if ok(r["sentence"]):
        boolean(
            r["sentence"],
            f'Does the sentence contain the answer to this question: "{r["question"]}"',
            r["label"] == 0,
            "qnli",
        )
for r in rng.sample(list(load_dataset("cardiffnlp/tweet_eval", "offensive", split="train")), 1500):
    if ok(r["text"]):
        boolean(r["text"], "Is this post offensive?", r["label"] == 1, "tweet_offensive")

# Synthetic policies, written independently of any other generator (different vocabulary and templates)
names = [
    "Aiko",
    "Bruno",
    "Chen",
    "Dara",
    "Emil",
    "Farah",
    "Goro",
    "Hana",
    "Ivan",
    "Jun",
    "Kira",
    "Luis",
    "Mina",
    "Nils",
    "Omar",
    "Priya",
]
objects = ["purchase order", "expense claim", "contract change", "access request", "vendor payment", "leave request"]
for _ in range(900):  # sign-off (boolean)
    who, other = rng.sample(names, 2)
    obj = rng.choice(objects)
    acct = rng.randint(10, 99)
    amt = rng.randint(1, 50) * 100
    limit = rng.choice([500, 1000, 2000, 3000])
    signer = rng.choice([who, other])
    over = amt > limit
    rule = f"A {obj} for unit {acct} is approved only if it is signed by the unit's designated approver, and amounts above ${limit} also need a second signature."
    second = rng.random() < 0.5
    case = f"{who} is the designated approver for unit {acct}. The {obj} is for ${amt}. It was signed by {signer}." + (
        " A second signature is attached." if second else " No second signature is attached."
    )
    boolean(
        json.dumps({"rule": rule, "case": case}, sort_keys=True),
        f"Is the {obj} approved?",
        signer == who and (not over or second),
        "synth_signoff",
    )
for _ in range(900):  # timeliness (ordered levels, asked as a lettered choice like the suite's Score)
    due = datetime.date(2026, rng.randint(1, 12), rng.randint(1, 28))
    grace = rng.choice([3, 5, 10, 14])
    delta = rng.randint(-10, 25)
    got = due + datetime.timedelta(days=delta)
    lvl = 0 if delta <= 0 else (1 if delta <= grace else 2)
    rule = f"An application that arrives on or before its closing date is on time. One that arrives up to {grace} days after the closing date is accepted with a penalty. Anything later is rejected."
    case = f"{rng.choice(names)}'s application arrived on {got:%B %-d, %Y}. The closing date was {due:%B %-d, %Y}."
    choice(
        json.dumps({"rule": rule, "case": case}, sort_keys=True),
        "How should this application be treated?",
        ["On time", "Accepted with a penalty", "Rejected"],
        lvl,
        "synth_timeliness",
    )


def cond(vals):
    k = rng.choice(list(vals))
    v = vals[k]
    op = rng.choice(["above", "at most", "exactly"])
    t = v + rng.choice([-3, -1, 0, 1, 3])
    return (f"the {k} is {op} {t}", {"above": v > t, "at most": v <= t, "exactly": v == t}[op])


for _ in range(1400):  # combinations and negations of conditions (no exception clauses: that family is held out)
    vals = {k: rng.randint(1, 60) for k in rng.sample(["score", "weight", "priority", "age", "count", "distance"], 4)}
    parts = [cond(vals) for _ in range(rng.choice([2, 3]))]
    mode = rng.choice(["all", "any", "not-any", "not-all"])
    truth = {
        "all": all(p[1] for p in parts),
        "any": any(p[1] for p in parts),
        "not-any": not any(p[1] for p in parts),
        "not-all": not all(p[1] for p in parts),
    }[mode]
    phr = {
        "all": "every one of these holds",
        "any": "at least one of these holds",
        "not-any": "none of these holds",
        "not-all": "not all of these hold",
    }[mode]
    rule = f"The request passes when {phr}: " + "; ".join(p[0] for p in parts) + ". Reference numbers do not matter."
    case = " ".join(f"The {k} is {v}." for k, v in vals.items()) + f" The reference number is {rng.randint(100, 999)}."
    opts = ["pass: the rule is satisfied", "fail: the rule is not satisfied"]
    rng.shuffle(opts)
    choice(
        json.dumps({"rule": rule, "case": case}, sort_keys=True),
        "Apply the rule to this case.",
        opts,
        [o.startswith("pass") for o in opts].index(truth),
        "synth_logic",
    )
rng.shuffle(out)
json.dump(out, open(os.path.join(DATA, "train_v1.json"), "w"))
import collections

print(len(out), collections.Counter(x["family"] for x in out))
