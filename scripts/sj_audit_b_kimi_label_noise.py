"""Experiment 4(b) (SYNTHESIS-v2.md, SearchJev mechanism (b)/(c)): is the frontier-LLM (Kimi K3) label
noisy on the rows where it overruled the training pool's own label?

`/work/next/kd/train_L1.json` (31,418 rows -- the exact size of the previous revision's own training
pool per the published card) carries both `pool_gold` (the dataset's own original label) and `gold` (what
was actually trained on -- Kimi's answer, on the rows where it disagreed with the pool). Rows with
`gold != pool_gold` are the ones Kimi overruled: 1,548 of them.

Method: ask three of the four checkpoints named in the pre-registration (`a6b`, `a8`, `soup3` -- all FP8,
all loadable on this担当's resource, `tput`/L40S; `L1`, the NVFP4 checkpoint, needs the Blackwell card in
Spain, outside this担当's assigned resource -- BRIEF-COMMON.md rule 5 -- and is excluded, recorded here
rather than silently skipped) each overruled row's question, with Kimi's own chosen option in its
declared position. Average the three models' own probability on Kimi's label (`p_gold`). Rows where this
ensemble average is below 0.05 are rows the ensemble -- independently of whatever any one of them was
itself trained on -- thinks Kimi's label is very unlikely to be right. For a sample of up to 100 such
rows, check automatically whether the ensemble's own strongest alternative instead agrees with the pool's
original label (`pool_gold`): if the ensemble isn't just "confused" but specifically and consistently
prefers the pool's own answer, that is itself evidence the overrule was a mislabel rather than noise in
the ensemble's own judgement.

Pre-registered rule: agreement >= 95% (ensemble's own top alternative equals pool_gold, on the p_gold<0.05
sample) -> delete the flat 0.5-weight-for-every-Kimi-row mechanism (c) from consideration: the noise this
mechanism is a blanket defence against turns out to be a small, identifiable minority (the low-p_gold
tail), not something spread evenly across every Kimi-sourced row, so a flat discount on all of them is the
wrong shape of fix. Also used by mechanism (b)'s own question (is Kimi's K3 labelling noisy at all): if
the ensemble is right to doubt Kimi on these rows, that's noise found, and feeds back into whether (c)'s
blanket discount makes sense for the OTHER ~98%+ of Kimi rows that are not in this low-confidence tail
(answer: no, a mechanism that cannot tell this row from a correctly-overruled one is not targeted at the
actual noise).

Usage: sj_audit_b_kimi_label_noise.py OUT.json
"""
import os, sys, json, random

os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from prismyra import Prismyra, Boolean, Choice

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/tok/results/sj_b_kimi_label_noise.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = "ABCDEFGHIJKLMNOP"

MODELS = {
    "a6b": "/work/next/models/p35-l36-kd_a6b_rank64a128",
    "a8": "/work/next/models/p35-l36-kd_a8_2xrows",
    "soup3": "/work/next/models/p35-l36-kd_a6b_soup3",
}
SAMPLE_N = int(os.environ.get("SJ_B_SAMPLE_N", "500"))  # of the 1,548 kimi-overruled rows

rows = json.load(open("/work/next/kd/train_L1.json"))
overruled = [r for r in rows if "pool_gold" in r and r["gold"] != r["pool_gold"]]
print(f"train_L1.json: {len(rows)} rows, {len(overruled)} kimi-overruled (gold != pool_gold)", flush=True)
random.Random(0).shuffle(overruled)
subset = overruled[:SAMPLE_N]


def question_for(r):
    if r["kind"] == "boolean":
        return Boolean(id="q", prompt=r["question"])
    return Choice(
        id="q",
        prompt=r["question"] + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(r["options"])),
        choices=list(L[: len(r["options"])]),
    )


def gold_index(r):
    return int(r["gold"]) if r["kind"] == "boolean" else r["gold"]


def probs_for(r, answer):
    if r["kind"] == "boolean":
        return [answer.probabilities["no"], answer.probabilities["yes"]]
    return [answer.probabilities[L[j]] for j in range(len(r["options"]))]


per_model_probs = {name: [None] * len(subset) for name in MODELS}
for name, path in MODELS.items():
    eng = Prismyra(path, require_kernels=True)
    with torch.inference_mode():
        for i, r in enumerate(subset):
            q = question_for(r)
            out = eng.ask(r["context"], [q])
            per_model_probs[name][i] = probs_for(r, out["q"])
            if i % 100 == 0:
                print(f"  {name}: row {i}/{len(subset)}", flush=True)
    del eng
    torch.cuda.empty_cache()
    print(f"{name} done", flush=True)

results = []
for i, r in enumerate(subset):
    gi = gold_index(r)
    ensemble = [sum(per_model_probs[name][i][k] for name in MODELS) / len(MODELS) for k in range(len(per_model_probs["a6b"][i]))]
    p_gold = ensemble[gi]
    ensemble_top = max(range(len(ensemble)), key=lambda k: ensemble[k])
    results.append({
        "i": i, "family": r.get("family"), "kind": r["kind"],
        "question": r["question"][:200], "options": r.get("options"),
        "pool_gold": r["pool_gold"], "kimi_gold": r["gold"],
        "ensemble_p_gold": p_gold, "ensemble_top_index": ensemble_top,
        "ensemble_top_matches_pool_gold": ensemble_top == r["pool_gold"],
        "per_model_p_gold": {name: per_model_probs[name][i][gi] for name in MODELS},
    })

low_p = [r for r in results if r["ensemble_p_gold"] < 0.05]
inspect_n = min(100, len(low_p))
inspected = low_p[:inspect_n]
n_match_pool = sum(1 for r in inspected if r["ensemble_top_matches_pool_gold"])
agreement = n_match_pool / inspect_n if inspect_n else None

summary = {
    "n_total_rows": len(rows),
    "n_kimi_overruled_total": len(overruled),
    "n_sampled": len(subset),
    "n_low_p_gold_lt_0.05": len(low_p),
    "n_inspected": inspect_n,
    "n_ensemble_top_matches_pool_gold": n_match_pool,
    "agreement_rate_with_pool_gold": agreement,
    "prereg_verdict": (
        "DELETE_mechanism_c_flat_weight" if agreement is not None and agreement >= 0.95
        else "KEEP_mechanism_c_for_consideration" if agreement is not None
        else "no_low_p_gold_rows_found"
    ),
    "inspected_rows": inspected,
    "all_results_summary": {"n": len(results), "mean_ensemble_p_gold": sum(r["ensemble_p_gold"] for r in results) / len(results)},
}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "inspected_rows"}, indent=1, default=str), flush=True)
print("wrote", OUT, flush=True)
