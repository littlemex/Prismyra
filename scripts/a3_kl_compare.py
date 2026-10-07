"""A3 (acc): compute KL(teacher, H12) and KL(teacher, experts-only) per document, bucketed by length
octile, over the 50 long documents. Teacher = FP8-36l's own real-engine probabilities on the exact same
50 docs, reused from A5's a5_tril_precision_full50.json run (same model, no LoRA, no quantization)."""
import json, math, sys

H12 = [json.loads(l) for l in open("/work/next/results/a3_h12.jsonl")]
EO = [json.loads(l) for l in open("/work/next/results/a3_expertsonly.jsonl")]
teacher_doc = json.load(open("/work/next/results/a5_tril_precision_full50.json"))["per_document"]
lens = json.load(open("/work/next/data/a3_long50.json"))

def kl(p, q):
    return sum(pi * math.log(max(pi, 1e-12) / max(qi, 1e-12)) for pi, qi in zip(p, q))

h12_by_i = {r["i"]: r for r in H12}
eo_by_i = {r["i"]: r for r in EO}
teach_by_i = {r["i"]: r for r in teacher_doc}

rows = []
for i in range(len(lens)):
    t = teach_by_i.get(i); h = h12_by_i.get(i); e = eo_by_i.get(i)
    if t is None or h is None or e is None or h.get("p") is None or e.get("p") is None:
        continue
    tp = t["p"]; hp = h["p"]; ep = e["p"]
    rows.append({"i": i, "chars": len(lens[i]["context"]), "gold": lens[i]["gold"],
                 "kl_h12": kl(tp, hp), "kl_experts_only": kl(tp, ep),
                 "teacher_gold_p": tp[lens[i]["gold"]], "h12_gold_p": hp[lens[i]["gold"]], "eo_gold_p": ep[lens[i]["gold"]]})

rows.sort(key=lambda r: r["chars"])
n = len(rows)
print(f"n documents compared: {n}")
OCT = 8
for k in range(OCT):
    lo, hi = int(k * n / OCT), int((k + 1) * n / OCT)
    chunk = rows[lo:hi]
    if not chunk:
        continue
    mh = sum(r["kl_h12"] for r in chunk) / len(chunk)
    me = sum(r["kl_experts_only"] for r in chunk) / len(chunk)
    print(f"octile {k+1}/{OCT} (n={len(chunk)}, chars {chunk[0]['chars']}..{chunk[-1]['chars']}): "
          f"mean KL(teacher,H12)={mh:.5f}  mean KL(teacher,experts_only)={me:.5f}  "
          f"experts_only worse by {me - mh:+.5f}")

overall_h = sum(r["kl_h12"] for r in rows) / n
overall_e = sum(r["kl_experts_only"] for r in rows) / n
n_eo_worse = sum(1 for r in rows if r["kl_experts_only"] > r["kl_h12"])
print(f"\noverall mean KL(teacher,H12)={overall_h:.5f}  mean KL(teacher,experts_only)={overall_e:.5f}")
print(f"experts_only worse than H12 on {n_eo_worse}/{n} documents")
json.dump(rows, open("/work/next/results/a3_kl_compare.json", "w"), indent=1)
print("wrote /work/next/results/a3_kl_compare.json")
