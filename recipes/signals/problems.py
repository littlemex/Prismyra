"""Problems with checkable answers for self_solve labels. Families are kept fine-grained (MMLU-Pro category, BBH task)
so evaluation can be done within a family and with families held out."""

import json, random, sys
from datasets import load_dataset

rng = random.Random(2026)
out = []


def add(fam, sub, prompt, answer, kind):
    out.append({"family": fam, "sub": sub, "prompt": prompt, "answer": answer, "kind": kind})


L = "ABCDEFGHIJ"
mp = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
rng.shuffle(mp)
for r in mp[:1200]:
    opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"]))
    add(
        "mmlu_pro",
        r["category"],
        f"{r['question']}\n{opts}\n\nThink briefly, then end with 'Answer: X' where X is the letter.",
        r["answer"],
        "letter",
    )
gs = list(load_dataset("openai/gsm8k", "main", split="test"))
rng.shuffle(gs)
for r in gs[:800]:
    add(
        "gsm8k",
        "gsm8k",
        r["question"] + "\n\nSolve it, then end with 'Answer: N' where N is the final number.",
        r["answer"].split("####")[-1].strip().replace(",", ""),
        "number",
    )
for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
    add(
        "math", r["subject"], r["problem"] + "\n\nSolve it and put the final answer in \\boxed{}.", r["answer"], "boxed"
    )
arc = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
rng.shuffle(arc)
for r in arc[:600]:
    opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
    add(
        "arc",
        "arc_challenge",
        f"{r['question']}\n{opts}\n\nThink briefly, then end with 'Answer: X' where X is the label.",
        r["answerKey"],
        "letter",
    )
nq = list(load_dataset("google-research-datasets/nq_open", split="validation"))
rng.shuffle(nq)
for r in nq[:600]:
    add(
        "nq",
        "nq_open",
        r["question"] + "?\n\nAnswer with a short phrase, ending with 'Answer: ...'.",
        r["answer"],
        "alias",
    )
for task in [
    "boolean_expressions",
    "date_understanding",
    "logical_deduction_five_objects",
    "navigate",
    "object_counting",
    "word_sorting",
    "multistep_arithmetic_two",
    "tracking_shuffled_objects_five_objects",
]:
    rows = list(load_dataset("lukaemon/bbh", task, split="test"))
    rng.shuffle(rows)
    for r in rows[:75]:
        add(
            "bbh",
            task,
            r["input"] + "\n\nThink briefly, then end with 'Answer: ...' giving only the final answer.",
            r["target"],
            "exact",
        )
for i, x in enumerate(out):
    x["id"] = i
json.dump(out, open(sys.argv[1], "w"))
import collections

print(len(out), collections.Counter(x["family"] for x in out))
