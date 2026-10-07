"""A3 (acc): build the same 50-long-document row set A5 used (bury7k/bury10k longest items + JevBench
long rows), in the plain context/kind/question/options/gold schema n2_fit.py's render() expects, so
the H12-vs-experts-only KL comparison is on the EXACT same documents as A5's precision measurement."""
import json, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/next/data/a3_long50.json"
N_DOCS = int(sys.argv[2]) if len(sys.argv) > 2 else 50

docs = []
for path in ["/work/items/race40_bury7k.json", "/work/items/race40_bury10k.json"]:
    man = json.load(open(path))
    for it in sorted(man["items"], key=lambda x: -len(x.get("context", ""))):
        sub = it["questions"][0]
        docs.append({"context": it["context"], "kind": sub["kind"], "question": sub["question"],
                     "options": sub["options"], "gold": sub["gold_index"], "family": "a3_bury",
                     "src": f"{path.split('/')[-1]}#{it['item']}"})
jevsel = json.load(open("/work/next/data/jevsel.json"))
for x in sorted(jevsel, key=lambda r: -len(r.get("context", ""))):
    # n2_fit.py's render() needs "question"/"options" (plain), not jevsel's pre-rendered "prompt"/"n_opt".
    if "prompt" in x:
        continue  # skip pre-rendered rows here; A3 reuses only the bury7k/10k docs which are already plain
    d = dict(x); d["src"] = "jevsel#" + str(x.get("jev_id", "?"))
    docs.append(d)
docs = docs[:N_DOCS]
json.dump(docs, open(OUT, "w"))
print(f"wrote {len(docs)} rows to {OUT}; length range chars: {min(len(d['context']) for d in docs)}..{max(len(d['context']) for d in docs)}")
