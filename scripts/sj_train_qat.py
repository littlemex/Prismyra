"""[sj_train_qat.py: SearchJev-cheap-filter candidates on top of train_qat.py, the exact script that trained
a6b/a7/a8 (see /Users/akazawt/tmp/smr/redesign/IDEAS-CEILING.md 6.4.11, 6.4.16).

This file is train_qat.py (/work/next/scripts/train_qat.py) UNCHANGED except for three additive, off-by-default
flags, each implementing one of the two SearchJev mechanisms the learning-free sieve (RUN-tok.md experiment 4)
found "worth trying" (c=4.36% for (a), reversed-order argmax flip, above the 2% stand-down line; 80.9% agreement
for (b), below the 95% stand-down line):

  --shuffle_opts        (a) option-order augmentation. For kind=="choice" rows only (kind=="boolean" rows have no
                         "options" list to permute -- Prismyra's own experiment-4(a) scoping decision, RUN-tok.md,
                         is reused here for the same reason). Each row's options are permuted ONCE, in-place, right
                         after the data file is loaded and BEFORE any filtering/sharding, with a dedicated
                         random.Random(SHUFFLE_SEED + row_index) instance that never touches the `random` module's
                         global state (so the existing per-epoch `random.shuffle(data)` call, and its interaction
                         with --seed, is bit-for-bit unaffected by turning this flag on or off). gold is remapped
                         along with the permutation. No family in train_L1_2x.json encodes an ordered scale (the
                         13+3 training pools are commonsense_qa/aqua_rat/mmlu_aux/cosmos_qa/sciq/openbookqa/dream/
                         quail/mnli/mrpc/qqp/arc/QuALITY/hotpot_comparison/bb_logical_deduction plus race/boolq/
                         mmlu/qnli/tweet_offensive/typed_*/synth_* -- checked by hand against train_L1_2x.json's
                         family list, 2026-10-08, no Likert/ordinal-scale family found), so nothing is excluded by
                         the "no ordered scale" carve-out at this time; the carve-out is still coded as an explicit
                         per-row skip (ex.get("ordered_scale")) in case a future data pool needs it.

  --kimi_weight W        (b) down-weight Kimi-overwritten labels. train_L1_2x.json rows that carry a `pool_gold`
                         field different from `gold` are rows where the labelling pass accepted Kimi K3's answer
                         over the source dataset's own answer. The CE term (the only term that reads `gold`) is
                         scaled by W (default 1.0, i.e. off) for those rows ONLY; the KD/cosine/vocab-KL terms,
                         which are self-distillation targets taken from the FP8-36l teacher's own forward pass and
                         never read `gold`, are left untouched regardless of this flag -- down-weighting a term that
                         does not depend on the disputed label would not implement "down-weight the label".

  --w_brier W            additive calibration term. Adds W * Brier(softmax(logits), one_hot(gold)) to the existing
                         w_ce*CE + w_kd*KL + w_h*cos + w_v*KV loss (does not replace any existing term, unlike
                         train_qat.py's own --loss brier choice, which replaces CE outright). Default 0.0 (off).

Everything else -- model loading, FP8/NVFP4 QAT forward swaps, LoRA module, render(), data filtering/sharding,
optimiser/schedule, checkpoint format -- is byte-for-byte train_qat.py, so a run with all three flags off
reproduces a8 exactly given the same --data/--teacher_pt/--out (not re-verified here beyond code diff; the
flags are additive and gated by `if a.shuffle_opts`/`if a.kimi_weight != 1.0`/`if a.w_brier`, so the untouched
code paths are identical to train_qat.py when all three are at their off-defaults).
"""
import argparse, collections, json, math, os, random, re, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import transformers.integrations.finegrained_fp8 as fp8
from transformers import AutoModel, AutoTokenizer, AutoConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
ap.add_argument("--data", default="/work/data/train_v1.json"); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--max_tokens", type=int, default=12500)
ap.add_argument("--loss", default="log", choices=["log", "brier"]); ap.add_argument("--teacher", default=""); ap.add_argument("--kd", type=float, default=0.5); ap.add_argument("--init", default=""); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--first_layer", type=int, default=0); ap.add_argument("--log_every", type=int, default=25)
ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,shared_expert.gate_proj,shared_expert.up_proj,shared_expert.down_proj")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nvfp4", default="")      # QAT: prepared NVFP4 experts (prismyra.kernels.nvfp4); model dir must leave FP8 experts out
ap.add_argument("--calib", default="/work/next/results/calib36.json")
ap.add_argument("--teacher_pt", default="")  # FP8 36l signals: option probs, final hidden, top-64 vocab logits
ap.add_argument("--w_ce", type=float, default=0.25); ap.add_argument("--w_kd", type=float, default=0.75)
ap.add_argument("--w_h", type=float, default=1.0); ap.add_argument("--w_v", type=float, default=0.5)
ap.add_argument("--fq_dense", action="store_true")
ap.add_argument("--fq_keep", default="")
ap.add_argument("--max_chars", type=int, default=0)
# --- sj additions (SearchJev cheap-filter candidates, RUN-sj.md pre-registration) ---
ap.add_argument("--shuffle_opts", action="store_true", help="(a) permute choice-row option order once at load time")
ap.add_argument("--shuffle_seed", type=int, default=20261008, help="independent RNG stream for --shuffle_opts, never touches random module global state")
ap.add_argument("--kimi_weight", type=float, default=1.0, help="(b) CE-term weight for rows where gold != pool_gold (Kimi overwrote the source label); 1.0 = off")
ap.add_argument("--w_brier", type=float, default=0.0, help="additive W * Brier(softmax(logits), one_hot(gold)) term; 0.0 = off")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
BLOCK = 128
import torch.distributed as dist
WORLD = int(os.environ.get("WORLD_SIZE", "1")); RANK = int(os.environ.get("RANK", "0")); DEV = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
if WORLD > 1: dist.init_process_group("nccl"); torch.cuda.set_device(DEV)

def dequant(weight, scale_inv, block_size=None):
    bs = block_size or (BLOCK, BLOCK)
    w = weight.to(torch.bfloat16) if weight.element_size() > 1 else weight.float()
    if weight.element_size() == 1:
        s = scale_inv.float().repeat_interleave(bs[0], 0).repeat_interleave(bs[1], 1)[: w.shape[0], : w.shape[1]]
        w = (w * s).to(torch.bfloat16)
    return w

def dequant_blocks(w, s, bs=(BLOCK, BLOCK)):
    """Block-FP8 to bf16 by broadcasting each block's scale; any leading dims (experts) are kept."""
    *lead, O, I = w.shape
    if O % bs[0] or I % bs[1]:
        return torch.stack([dequant(w[e], s[e], bs) for e in range(w.shape[0])]) if lead else dequant(w, s, bs)
    out = torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)
    wv = w.view(-1, O // bs[0], bs[0], I // bs[1], bs[1]); sv = s.view(-1, O // bs[0], 1, I // bs[1], 1)
    ov = out.view(-1, O // bs[0], bs[0], I // bs[1], bs[1]); per = max(1, (1 << 24) // (O * I))
    if per >= 1 and O * I <= (1 << 24):
        for e in range(0, wv.shape[0], per):
            ov[e:e + per] = (wv[e:e + per].float() * sv[e:e + per].float()).to(torch.bfloat16)
    else:
        step = max(1, (1 << 24) // (bs[0] * I))
        for e in range(wv.shape[0]):
            for r in range(0, O // bs[0], step):
                ov[e, r:r + step] = (wv[e, r:r + step].float() * sv[e, r:r + step].float()).to(torch.bfloat16)
    return out

def fp8_linear_train(input, weight, weight_scale_inv, block_size=None, activation_scale=None, bias=None, allow_deepgemm=True, **kw):
    w = weight if weight.element_size() > 1 else dequant_blocks(weight, weight_scale_inv, block_size or (BLOCK, BLOCK))
    return F.linear(input, w, bias)
fp8.fp8_linear = fp8_linear_train

from types import SimpleNamespace
import transformers.integrations.moe as moe
def experts_forward_train(self, hidden_states, top_k_index, top_k_weights):
    bs = tuple(self.block_size or (BLOCK, BLOCK))
    view = SimpleNamespace(num_experts=self.num_experts, has_gate=self.has_gate, has_bias=False, is_transposed=False,
                           gate_up_proj=dequant_blocks(self.gate_up_proj, self.gate_up_proj_scale_inv, bs),
                           down_proj=dequant_blocks(self.down_proj, self.down_proj_scale_inv, bs),
                           _apply_gate=self._apply_gate, act_fn=self.act_fn)
    return moe.grouped_mm_experts_forward(view, hidden_states, top_k_index, top_k_weights)
fp8.FP8Experts.forward = experts_forward_train

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
sys.path.insert(0, "/work/next/scripts")
from nvfp4_deq import deq_nv
class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q): return q
    @staticmethod
    def backward(ctx, g): return g, None
def fq_act(x, G):
    with torch.no_grad():
        xf = x.float(); shp = xf.shape; b = xf.view(-1, shp[-1] // 16, 16)
        s = (b.abs().amax(-1, keepdim=True) / 6.0 * G).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        eff = s / G
        y = torch.where(eff > 0, b / eff, torch.zeros_like(b)).clamp(-6, 6)
        idx = torch.bucketize(y.abs(), E2M1_MID.to(x.device))
        q = (E2M1.to(x.device)[idx] * y.sign() * eff).view(shp).to(x.dtype)
    return _STE.apply(x, q)
def experts_forward_qat(self, hidden_states, top_k_index, top_k_weights):
    nv = self._nv
    W1 = deq_nv(nv["w1"], nv["s1"], nv["g1"]); W2 = deq_nv(nv["w2"], nv["s2"], nv["g2"])
    num_tokens, k = hidden_states.shape[0], top_k_index.shape[-1]
    ids = top_k_index.reshape(-1); w = top_k_weights.reshape(-1)
    ids_g, perm = torch.sort(ids)
    xs = fq_act(hidden_states, nv["a13"])[perm // k]
    offs = torch.cumsum(torch.histc(ids_g.int(), bins=W1.shape[0], min=0, max=W1.shape[0] - 1), 0, dtype=torch.int32)
    h = moe._grouped_linear(xs, W1, offs)
    g_, u_ = h.chunk(2, -1); a_ = F.silu(g_) * u_
    o = moe._grouped_linear(fq_act(a_, nv["a2"]), W2, offs) * w[perm].unsqueeze(-1).to(h.dtype)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return o[inv].view(num_tokens, k, -1).sum(1)
if a.nvfp4:
    fp8.FP8Experts.forward = experts_forward_qat

class LoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base; self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
    def forward(self, x):
        if getattr(self, "ga", None) is not None:
            with torch.no_grad():
                W = dequant(self.base.weight, self.base.weight_scale_inv)
            Wm = W + (self.B @ self.A * self.scale).to(W.dtype)
            Gw = (448.0 * 6.0 / Wm.detach().abs().amax().float().clamp_min(1e-12)).item()
            return F.linear(fq_act(x, self.ga), fq_act(Wm, Gw))
        return self.base(x) + (F.linear(F.linear(x.to(torch.float32), self.A), self.B) * self.scale).to(x.dtype)

tok = AutoTokenizer.from_pretrained(a.model)
if a.nvfp4:
    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map=DEV, experts_implementation="eager")
    for m_ in model.modules():
        if type(m_).__name__ == "FP8Experts":
            for p_ in ("gate_up_proj", "gate_up_proj_scale_inv", "down_proj", "down_proj_scale_inv"):
                if hasattr(m_, p_): delattr(m_, p_)
    torch.cuda.empty_cache()
    from safetensors.torch import load_file
    side = load_file(a.nvfp4); cal = json.load(open(a.calib))["moe"]
    for n_, m_ in model.named_modules():
        if type(m_).__name__ == "FP8Experts":
            L_i = n_.split("layers.")[1].split(".")[0]; c_ = cal[f"language_model.layers.{L_i}.mlp"]
            m_._nv = {k_: side[f"{L_i}.{k_}"].to(DEV) for k_ in ("w1", "s1", "g1", "w2", "s2", "g2")}
            m_._nv["a13"] = 448.0 * 6.0 / (1.25 * c_["w13_in_amax"]); m_._nv["a2"] = 448.0 * 6.0 / (1.25 * max(c_["w2_in_amax"]))
            m_.num_experts = m_._nv["w1"].shape[0]
    del side; torch.cuda.empty_cache()
else:
    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map=DEV, experts_implementation="eager")
for p in model.parameters(): p.requires_grad_(False)
if hasattr(model, "visual"):
    del model.visual; torch.cuda.empty_cache()
lm = model.language_model if hasattr(model, "language_model") else model
targets = a.targets.split(","); wrapped = []
for name, mod in list(lm.named_modules()):
    parts = name.split(".")
    if "layers" not in parts or parts.index("layers") + 1 >= len(parts): continue
    li = int(parts[parts.index("layers") + 1])
    if li < a.first_layer or ".experts." in name or name.endswith(".experts"): continue
    for t in targets:
        if name.endswith(t) and isinstance(mod, nn.Linear):
            parent = lm.get_submodule(name.rsplit(".", 1)[0]); child = name.rsplit(".", 1)[1]
            setattr(parent, child, LoRA(mod, a.rank, a.alpha)); wrapped.append(name)
            if a.fq_dense and not (a.fq_keep and re.search(a.fq_keep, name)):
                dk = json.load(open(a.calib))["dense_in_amax"]; key = "language_model." + name if not name.startswith("language_model.") else name
                if key not in dk: key = key.replace(".self_attn.", ".self_attn.inner.")
                getattr(parent, child).ga = 448.0 * 6.0 / (1.25 * dk[key]) if key in dk else None
                if key not in dk: print("no calibration for", key, flush=True)
            break
params = [p for n, p in lm.named_parameters() if n.endswith(".A") or n.endswith(".B")]
if a.init:
    init = torch.load(a.init, map_location="cpu"); src = init["lora"] if "lora" in init else init
    with torch.no_grad():
        for n, p in lm.named_parameters():
            if n in src: p.copy_(src[n].to(p.device, p.dtype))
print("wrapped", len(wrapped), "trainable", sum(p.numel() for p in params), flush=True)
lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
lm.config.use_cache = False

sys.path.insert(0, "/work/prismyra")
from prismyra.schema import Choice, Boolean
from prismyra import readout
cfg = AutoConfig.from_pretrained(a.model); hidden = getattr(cfg, "text_config", cfg).hidden_size
U = readout.load_unembedding(a.model, hidden, DEV, torch.bfloat16)
L_ = "ABCDEFGHIJKLMNOP"

def render(ex):
    if ex["kind"] == "boolean":
        q = Boolean(id="q", prompt=ex["question"])
    else:
        listed = "\n".join(f"{L_[j]}. {o}" for j, o in enumerate(ex["options"]))
        q = Choice(id="q", prompt=f"{ex['question']}\n{listed}", choices=list(L_[: len(ex["options"])]))
    plan = readout.plan(q, tok)
    ctx = tok(ex["context"], add_special_tokens=True)["input_ids"]
    suf = tok("\n" + plan.text, add_special_tokens=False)["input_ids"]
    opts = list(q.options)
    gold_opt = ("yes" if ex["gold"] else "no") if ex["kind"] == "boolean" else L_[ex["gold"]]
    gold = opts.index(gold_opt)
    return ctx + suf, plan.token_ids, gold

data = json.load(open(a.data))
if a.teacher_pt:
    T = torch.load(a.teacher_pt)
    for i_, t_ in T.items(): data[int(i_)]["T"] = t_
    del T
if a.teacher:
    for l in open(a.teacher):
        r = json.loads(l); data[r["i"]]["teacher"] = r["p"]
if a.n: data = data[: a.n]

# --- sj (a): option-order augmentation, applied once at load time, BEFORE any filtering/sharding, with an
# RNG stream (`shuf_rng`) completely separate from the `random` module's global state used for epoch
# shuffling and (if --teacher is used instead of --teacher_pt) nothing else here reads `random` before this
# point, so turning --shuffle_opts on/off cannot perturb the per-epoch `random.shuffle(data)` sequence below.
n_shuffled = n_skipped_scale = 0
if a.shuffle_opts:
    for row_idx, ex in enumerate(data):
        if ex.get("kind") != "choice":
            continue  # boolean rows have no "options" list to permute (same scoping as RUN-tok.md experiment 4(a))
        if ex.get("ordered_scale"):
            n_skipped_scale += 1
            continue  # carve-out for a future ordered-scale family; none exists in train_L1_2x.json today
        opts = ex["options"]
        if len(opts) < 2:
            continue
        rng = random.Random(a.shuffle_seed + row_idx)
        perm = list(range(len(opts)))
        rng.shuffle(perm)
        ex["options"] = [opts[j] for j in perm]
        ex["gold"] = perm.index(ex["gold"])
        n_shuffled += 1
    if RANK == 0:
        print(f"--shuffle_opts: permuted {n_shuffled} choice rows, skipped {n_skipped_scale} ordered-scale rows "
              f"(of {len(data)} total)", flush=True)

# --- sj (b): per-row CE weight for Kimi-overwritten labels (gold != pool_gold). Computed here, once, from the
# fields build_kd2x.py/build_kdpool.py already write, so it survives the max_chars/max_tokens filters and the
# rank sharding below untouched (it travels with the row dict).
n_kimi_down = 0
if a.kimi_weight != 1.0:
    for ex in data:
        pg = ex.get("pool_gold")
        ex["_ce_w"] = a.kimi_weight if (pg is not None and ex["gold"] != pg) else 1.0
        if ex["_ce_w"] != 1.0: n_kimi_down += 1
    if RANK == 0:
        print(f"--kimi_weight {a.kimi_weight}: down-weighted {n_kimi_down} of {len(data)} rows "
              f"(gold overwritten relative to pool_gold)", flush=True)

if a.max_chars:
    before = len(data)
    excluded = [x for x in data if len(x["context"]) + len(x["question"]) > a.max_chars]
    data = [x for x in data if len(x["context"]) + len(x["question"]) <= a.max_chars]
    if RANK == 0:
        print(f"max_chars={a.max_chars}: excluded {len(excluded)} of {before} rows; "
              f"by family: {dict(collections.Counter(x['family'] for x in excluded))}", flush=True)
before_tok = len(data)
data = [x for x in data if len(render(x)[0]) <= a.max_tokens]
if RANK == 0:
    print(f"max_tokens={a.max_tokens}: dropped {before_tok - len(data)} of {before_tok} rows before sharding", flush=True)
usable = (len(data) // (WORLD * a.accum)) * (WORLD * a.accum)
if usable < len(data) and RANK == 0:
    print(f"trimming {len(data) - usable} of {len(data)} rows so every rank's shard is exactly "
          f"{usable // WORLD} rows ({usable // WORLD // a.accum} accum-groups each)", flush=True)
data = data[:usable]
data = data[RANK::WORLD]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
steps_total = a.epochs * len(data) // a.accum; step = 0; t0 = time.time(); run_loss = []; skipped = 0
sched = lambda s: min(1.0, s / 20) * max(0.1, 1 - s / max(1, steps_total))
lm.train()
for ep in range(a.epochs):
    random.shuffle(data)
    for i, ex in enumerate(data):
        ids, opt_ids, gold = render(ex)
        assert len(ids) <= a.max_tokens, "long rows are dropped before sharding; a skip here would desynchronise ranks"
        x = torch.tensor([ids], device=DEV)
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0, -1]
        logits = (h.float() @ U[torch.tensor(sum(opt_ids, []) if isinstance(opt_ids[0], list) else opt_ids, device=DEV)].float().t())
        if a.loss == "brier":
            pr = torch.softmax(logits, -1); y = F.one_hot(torch.tensor(gold, device=DEV), pr.shape[-1]).float()
            loss = ((pr - y) ** 2).sum() / a.accum
        else:
            ce_w = ex.get("_ce_w", 1.0)
            loss = ce_w * F.cross_entropy(logits[None], torch.tensor([gold], device=DEV))
            if "T" in ex:
                T_ = ex["T"]; tp = T_["p"].to(DEV).float().clamp_min(1e-8)
                kl = (tp * (tp.log() - F.log_softmax(logits, -1))).sum()
                cos = 1 - F.cosine_similarity(h.float(), T_["h"].to(DEV).float(), 0)
                vi = T_["ti"].to(DEV).long(); vt = torch.softmax(T_["tv"].to(DEV).float(), -1)
                vs = F.log_softmax(h.float() @ U[vi].float().t(), -1)
                kv = (vt * (vt.log() - vs)).sum()
                loss = a.w_ce * loss + a.w_kd * kl + a.w_h * cos + a.w_v * kv
            elif "teacher" in ex:
                tp = torch.tensor(ex["teacher"], device=DEV).clamp_min(1e-8)
                kl = (tp * (tp.log() - F.log_softmax(logits, -1))).sum()
                loss = (1 - a.kd) * loss + a.kd * kl
            if a.w_brier:
                pr = torch.softmax(logits, -1); y = F.one_hot(torch.tensor(gold, device=DEV), pr.shape[-1]).float()
                loss = loss + a.w_brier * ((pr - y) ** 2).sum()
            loss = loss / a.accum
        loss.backward(); run_loss.append(loss.item() * a.accum)
        if (i + 1) % a.accum == 0:
            for g in opt.param_groups: g["lr"] = a.lr * sched(step)
            if WORLD > 1:
                for p_ in params:
                    if p_.grad is None: p_.grad = torch.zeros_like(p_)
                    dist.all_reduce(p_.grad); p_.grad /= WORLD
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % a.log_every == 0 and RANK == 0:
                el = time.time() - t0
                print(f"step {step}/{steps_total} loss {sum(run_loss[-a.accum*a.log_every:])/len(run_loss[-a.accum*a.log_every:]):.4f} "
                      f"ex/s {(i+1+ep*len(data))/el:.2f} mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB skipped {skipped}", flush=True)
            if step % 100 == 0 and RANK == 0:
                torch.save({n: p.detach().cpu() for n, p in lm.named_parameters() if n.endswith(".A") or n.endswith(".B")}, a.out + ".partial")
if WORLD > 1:
    torch.cuda.synchronize(); dist.barrier()
if RANK != 0: sys.exit(0)
state = {n: p.detach().cpu() for n, p in lm.named_parameters() if n.endswith(".A") or n.endswith(".B")}
torch.save({"lora": state, "rank": a.rank, "alpha": a.alpha, "targets": targets, "wrapped": wrapped,
            "args": vars(a)}, a.out)
print("saved", a.out, "steps", step, "skipped", skipped, "minutes", round((time.time() - t0) / 60, 1), flush=True)
