"""A4 (acc) deliverable 2: does the shipped NVFP4 activation scale (set once from calib128.json's
amax, see /work/next/tokyo/scripts/calib.py and /work/next/results/calib36.json) clip/saturate on
REAL long documents (bury7k/bury10k/JevBench long rows) that are longer or differently distributed
than the calibration set?

Same hooking method as calib.py (register_forward_pre_hook on every FusedExperts module to track the
observed amax of the routed-experts' w13 input over all tokens, and the down-projection w2 input amax
per expert), but run over up to 50 long documents instead of calib128's 128 rows, and compare the
observed amax against calib36.json's recorded amax (which already has a 1.25x headroom factor baked
into the serving-time global scale G = 448*6/(1.25*amax) -- see n2_fit.py/run_qat12.sh's `nv["a13"]`).
Clipping happens whenever a real activation exceeds `1.25 * calib_amax`; this script reports, per MoE
layer, the ratio (observed_amax / (1.25*calib_amax)) so a value above 1.0 means real documents clip.
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "/work/next/scripts")
sys.path.insert(0, "/work/prismyra")

import torch, torch.nn.functional as F
from prismyra import Prismyra
import prismyra.kernels.qwen3_moe as K
import common

M = os.environ.get("A4_MODEL", "/work/models/p35-l36_kd025_7k_cap32k")
N_DOCS = int(os.environ.get("A4_N_DOCS", "50"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/next/results/a4_saturation_check.json"
CALIB36 = json.load(open("/work/next/results/calib36.json"))

docs = []
for path in ["/work/items/race40_bury7k.json", "/work/items/race40_bury10k.json"]:
    man = json.load(open(path))
    for it in sorted(man["items"], key=lambda x: -len(x.get("context", ""))):
        sub = it["questions"][0]
        docs.append({"context": it["context"], "kind": sub["kind"], "question": sub["question"],
                     "options": sub["options"], "gold": sub["gold_index"], "src": f"{os.path.basename(path)}#{it['item']}"})
jevsel = json.load(open("/work/next/data/jevsel.json"))
for x in sorted(jevsel, key=lambda r: -len(r.get("context", ""))):
    d = dict(x); d["src"] = "jevsel#" + str(x.get("jev_id", "?"))
    docs.append(d)
docs = docs[:N_DOCS]
print(f"using {len(docs)} documents; length range chars: {min(len(d['context']) for d in docs)}..{max(len(d['context']) for d in docs)}", flush=True)

eng = Prismyra(M, group=32); eng._check_fits = lambda *a, **k: None
names = {m: n for n, m in eng.backbone.named_modules()}
experts = [m for m in eng.backbone.modules() if isinstance(m, K.FusedExperts)]
E = experts[0].w1.shape[0]
stat = {names[m]: {"w13": 0.0, "w2": torch.zeros(E), "tok": torch.zeros(E, dtype=torch.long), "n_docs_seen": 0} for m in experts}

def deq(w, s):
    Ex, O, I = w.shape
    return (w.float().view(Ex, O // 128, 128, I // 128, 128) * s.float().view(Ex, O // 128, 1, I // 128, 1)).view(Ex, O, I).to(torch.bfloat16)

def moe_hook(m, args):
    from vllm.model_executor.layers.fused_moe import fused_topk
    x = args[0].reshape(-1, args[0].shape[-1]); st = stat[names[m]]
    st["w13"] = max(st["w13"], float(x.abs().amax()))
    logits, _, _ = m.gate(x); _, ids = fused_topk(x, logits, m.top_k, renormalize=True)[:2]
    W1 = deq(m.w1, m.quant.w1_scale); I = W1.shape[1] // 2
    for e in ids.unique().tolist():
        tok = (ids == e).any(-1); xe = x[tok]
        h = xe @ W1[e].T; a = F.silu(h[:, :I]) * h[:, I:]
        st["w2"][e] = max(float(st["w2"][e]), float(a.abs().amax())); st["tok"][e] += int(tok.sum())
    del W1
for m in experts: m.register_forward_pre_hook(moe_hook)

t0 = time.time()
with torch.inference_mode():
    for i, d in enumerate(docs):
        q = common.question(d)
        eng.ask(d["context"], [q])
        if i % 5 == 0:
            print(i, len(docs), f"{time.time()-t0:.0f}s", flush=True)

report = {}
worst = []
for layer_name, st in stat.items():
    key = layer_name if layer_name.startswith("language_model.") else "language_model." + layer_name
    calib = CALIB36["moe"].get(key) or CALIB36["moe"].get(layer_name)
    if calib is None:
        continue
    thresh13 = 1.25 * calib["w13_in_amax"]
    ratio13 = st["w13"] / thresh13 if thresh13 > 0 else None
    thresh2 = [1.25 * v for v in calib["w2_in_amax"]]
    ratio2 = [(st["w2"][e].item() / thresh2[e]) if thresh2[e] > 0 else None for e in range(len(thresh2))]
    row = {"observed_w13_amax": st["w13"], "calib_w13_amax": calib["w13_in_amax"], "w13_clip_ratio": ratio13,
           "max_w2_clip_ratio": max([r for r in ratio2 if r is not None], default=None),
           "n_experts_over_1x_w2": sum(1 for r in ratio2 if r is not None and r > 1.0)}
    report[layer_name] = row
    worst.append((layer_name, ratio13, row["max_w2_clip_ratio"]))

worst.sort(key=lambda x: -(max(x[1] or 0, x[2] or 0)))
summary = {"n_docs": len(docs), "n_layers": len(report), "worst_5_layers": worst[:5],
           "n_layers_w13_clipping": sum(1 for _, r in report.items() if r["w13_clip_ratio"] and r["w13_clip_ratio"] > 1.0),
           "n_layers_any_w2_clipping": sum(1 for _, r in report.items() if r["max_w2_clip_ratio"] and r["max_w2_clip_ratio"] > 1.0),
           "per_layer": report}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=1, default=str), flush=True)
print("wrote", OUT, flush=True)
