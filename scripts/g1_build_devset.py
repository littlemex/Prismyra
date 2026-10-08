"""Experiment 4, G1 (P7 gate): build the ~2,000-question non-lock dev set from the four named themes
(RACE-style, BoolQ-style, bury-long, Kev-permitted-family), entirely from this project's existing
production evaluation files -- none of lock9/lock10/lock11/lock12 are touched, and these files are the
same ones the published model card's own Evaluation section already treats as held-out.

Scope recorded (cost/time, not hidden): bury7k/bury10k (each a long document, several questions per
document) are capped at their first 50 documents each rather than the full 40-document... wait, note:
race40_bury{7k,10k}.json only have 40 documents (157 questions) each in total already, so no cap is
needed there -- taken in full. The total below (579 + 400 + 157 + 157 + 600 = 1,893) is short of exactly
2,000 because that is the total size of the held-out files this project already has; inflating it with a
different, non-held-out source would cost more than the shortfall is worth for a learning-free sieve.

Output: a flat JSON list, each row {context, kind, question, options, gold, family, src}.

Usage: g1_build_devset.py OUT.json
"""
import json, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/tok/results/g1_devset.json"
rows = []


def add_choice_style(path, family_prefix):
    man = json.load(open(path))
    for it in man["items"]:
        for q in it["questions"]:
            rows.append({
                "context": it["context"], "kind": q["kind"], "question": q["question"],
                "options": q["options"], "gold": q["gold_index"],
                "family": f"{family_prefix}", "src": f"{path}#{it['item']}",
            })


add_choice_style("/work/items/race150.json", "race")
add_choice_style("/work/items/boolq400.json", "boolq")
add_choice_style("/work/items/race40_bury7k.json", "bury7k")
add_choice_style("/work/items/race40_bury10k.json", "bury10k")

kev = json.load(open("/work/data/dev_kevside.json"))
for i, r in enumerate(kev):
    rows.append({
        "context": r["context"], "kind": r["kind"], "question": r["question"],
        "options": r["options"] if r["kind"] == "choice" else ["no", "yes"],
        "gold": r["gold"], "family": f"kev_{r.get('family', '?')}", "src": f"dev_kevside#{i}",
    })

print(f"built {len(rows)} rows", flush=True)
from collections import Counter
fam_prefix = Counter(r["family"].split("_")[0] for r in rows)
print("by theme:", dict(fam_prefix), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
