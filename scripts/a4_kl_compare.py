"""A4 (acc): compare the shipped w2 scale (already measured as A3's "experts-only" baseline KL=0.0512,
since that run also used the shipped nvfp4_experts_36l.safetensors with no --fq_dense) against the two
rescaled-w2 alternatives (error-minimizing search, 4/6 adaptive), via no-train KL to the same teacher,
on the same 50 long documents."""
import json, math

teacher_doc = json.load(open("/work/next/results/a5_tril_precision_full50.json"))["per_document"]
lens = json.load(open("/work/next/data/a3_long50.json"))
shipped = {r["i"]: r for r in [json.loads(l) for l in open("/work/next/results/a3_expertsonly.jsonl")]}
search = {r["i"]: r for r in [json.loads(l) for l in open("/work/next/results/a4_w2search.jsonl")]}
fourover6 = {r["i"]: r for r in [json.loads(l) for l in open("/work/next/results/a4_w2fourover6.jsonl")]}
teach = {r["i"]: r for r in teacher_doc}

def kl(p, q):
    return sum(pi * math.log(max(pi, 1e-12) / max(qi, 1e-12)) for pi, qi in zip(p, q))

rows = []
for i in range(len(lens)):
    t = teach.get(i); s = shipped.get(i); se = search.get(i); fo = fourover6.get(i)
    if any(x is None or x.get("p") is None for x in (t, s, se, fo)):
        continue
    rows.append({"i": i, "chars": len(lens[i]["context"]),
                 "kl_shipped": kl(t["p"], s["p"]), "kl_search": kl(t["p"], se["p"]), "kl_fourover6": kl(t["p"], fo["p"])})

n = len(rows)
m_s = sum(r["kl_shipped"] for r in rows) / n
m_se = sum(r["kl_search"] for r in rows) / n
m_fo = sum(r["kl_fourover6"] for r in rows) / n
n_search_better = sum(1 for r in rows if r["kl_search"] < r["kl_shipped"])
n_fo_better = sum(1 for r in rows if r["kl_fourover6"] < r["kl_shipped"])
print(f"n documents compared: {n}")
print(f"mean KL(teacher, shipped w2)      = {m_s:.5f}")
print(f"mean KL(teacher, search w2)       = {m_se:.5f}  (better on {n_search_better}/{n} docs)")
print(f"mean KL(teacher, four-over-six w2)= {m_fo:.5f}  (better on {n_fo_better}/{n} docs)")
json.dump(rows, open("/work/next/results/a4_kl_compare.json", "w"), indent=1)
print("wrote /work/next/results/a4_kl_compare.json")
