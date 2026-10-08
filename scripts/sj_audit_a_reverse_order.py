"""Experiment 4(a) (SYNTHESIS-v2.md, SearchJev mechanism (a)): does reversing a choice question's option
order change the model's answer, measured by CONTENT identity (not by letter), on the production
multiple-choice sets?

SearchJev (https://github.com/EvoScientist/SearchJev) trains with option order shuffled as an
augmentation against the model learning a positional habit instead of reading content. The diagnostic
this project's pre-registration calls for is the other direction: does the CURRENT model (never trained
with that augmentation) already answer by content regardless of presentation order? If so, the
augmentation would not be fixing a problem this model has, and is deleted from consideration without ever
training it.

`c` := the fraction of questions whose answer, read by the CONTENT of the chosen option (not by which
letter it landed on), differs between the normal-order and the fully reversed-order presentation of the
same question's options. A content-faithful model's answer should not depend on where its options sit, so
`c` close to 0 is the expected, "no problem" outcome; `c <= 2%` is this experiment's pre-registered bar for
deleting the mechanism from consideration. (A model that only ever picked position 0 regardless of content
would show `c` close to 100%, the opposite, alarming end of this scale -- `c` is defined as the DISAGREEMENT
rate precisely so the deletion rule's direction ("small c -> delete, nothing to fix") reads naturally.)

Scope (recorded, not hidden): boolean questions have no content order to reverse (two fixed options, "no"
then "yes") and are excluded. Of the five production sets, race150 (579 Q), race40_bury7k (157 Q) and
race40_bury10k (157 Q) are entirely choice-kind and included in full; Kev's choice-kind rows (300 of its
600) are included; BoolQ (400, boolean-only) is excluded entirely -- the mechanism this diagnoses has
nothing to say about a two-fixed-option set, so including it would not change the measurement, only its
cost.

Usage: sj_audit_a_reverse_order.py OUT.json
"""
import os, sys, json

os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/work/next/scripts")

import torch
from prismyra import Prismyra, Choice

MODEL = os.environ.get("SJ_MODEL", "/work/next/models/p35-l36-kd_a8_2xrows")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/tok/results/sj_a_reverse_order.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = "ABCDEFGH"


def race_style_docs(path):
    man = json.load(open(path))
    docs = []
    for it in man["items"]:
        rows = [q for q in it["questions"] if q.get("kind", "choice") == "choice"]
        if rows:
            docs.append({"context": it["context"], "rows": rows})
    return docs


def kev_choice_docs(limit=None):
    rows = json.load(open("/work/data/dev_kevside.json"))
    rows = [r for r in rows if r["kind"] == "choice"]
    if limit:
        rows = rows[:limit]
    # Kev is one row per context (no natural grouping) -- one doc per row.
    return [{"context": r["context"], "rows": [r]} for r in rows]


DATASETS = {
    "race150": lambda: race_style_docs("/work/items/race150.json"),
    "race40_bury7k": lambda: race_style_docs("/work/items/race40_bury7k.json"),
    "race40_bury10k": lambda: race_style_docs("/work/items/race40_bury10k.json"),
    "kev_choice": lambda: kev_choice_docs(),
}

eng = Prismyra(MODEL, require_kernels=True)

results = {}
with torch.inference_mode():
    for name, loader in DATASETS.items():
        docs = loader()
        n_q = sum(len(d["rows"]) for d in docs)
        print(f"{name}: {len(docs)} documents, {n_q} choice questions", flush=True)
        n_total = 0
        n_changed = 0
        changed_examples = []
        for di, doc in enumerate(docs):
            rows = doc["rows"]
            normal_qs = [
                Choice(
                    id=f"q{i}",
                    prompt=r["question"] + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(r["options"])),
                    choices=list(L[: len(r["options"])]),
                )
                for i, r in enumerate(rows)
            ]
            reversed_qs = [
                Choice(
                    id=f"q{i}",
                    prompt=r["question"] + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(list(reversed(r["options"])))),
                    choices=list(L[: len(r["options"])]),
                )
                for i, r in enumerate(rows)
            ]
            normal_out = eng.ask(doc["context"], normal_qs)
            reversed_out = eng.ask(doc["context"], reversed_qs)
            for i, r in enumerate(rows):
                opts = r["options"]
                normal_choice_content = opts[L.index(normal_out[f"q{i}"].option)]
                reversed_choice_content = list(reversed(opts))[L.index(reversed_out[f"q{i}"].option)]
                n_total += 1
                if normal_choice_content != reversed_choice_content:
                    n_changed += 1
                    if len(changed_examples) < 10:
                        changed_examples.append({
                            "dataset": name, "doc_i": di, "q_i": i,
                            "normal_choice": normal_choice_content, "reversed_choice": reversed_choice_content,
                        })
            if di % 50 == 0:
                print(f"  {name} doc {di}/{len(docs)}: n_total={n_total} n_changed={n_changed}", flush=True)
        c = n_changed / n_total if n_total else None
        results[name] = {"n_questions": n_total, "n_changed": n_changed, "c_disagreement_rate": c}
        print(f"{name}: c = {c:.4f} ({n_changed}/{n_total})", flush=True)

n_total_all = sum(r["n_questions"] for r in results.values())
n_changed_all = sum(r["n_changed"] for r in results.values())
c_overall = n_changed_all / n_total_all if n_total_all else None
summary = {
    "model": MODEL,
    "by_dataset": results,
    "overall": {"n_questions": n_total_all, "n_changed": n_changed_all, "c_disagreement_rate": c_overall},
    "prereg_verdict": (
        "DELETE_mechanism_a_no_order_sensitivity" if c_overall is not None and c_overall <= 0.02
        else "KEEP_FOR_CONSIDERATION_order_sensitivity_found" if c_overall is not None
        else "no_data"
    ),
}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items()}, indent=1, default=str), flush=True)
print("wrote", OUT, flush=True)
