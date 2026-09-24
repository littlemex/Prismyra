"""LoRA on the FP8 Qwen3.6-35B-A3B that Prismyra serves, trained on exactly the read-out Prismyra performs.

Usage (one node, N GPUs):

    torchrun --nproc_per_node N train.py --data train.json --out lora.pt

`train.json` is a list of rows as `build_data.py` writes them. Afterwards fold the adapter in with `merge.py`.

The loss is the log score of the declared options' letters at the branch's last token -- the probability Prismyra
returns -- so the objective is calibrated decisions, not generation. The routed experts are frozen and never adapted,
so the checkpoint merged afterwards keeps the shape Prismyra's MoE kernels were measured on.

FP8 weights stay FP8 in memory. For training only, transformers' block-FP8 matmul is swapped for "dequantise the block
and multiply in bf16", which autograd can differentiate; the weights themselves never receive a gradient.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import transformers.integrations.finegrained_fp8 as fp8
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--max_tokens", type=int, default=12500)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--first_layer", type=int, default=0)
ap.add_argument("--log_every", type=int, default=25)
ap.add_argument(
    "--targets",
    default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,shared_expert.gate_proj,shared_expert.up_proj,shared_expert.down_proj",
)
a = ap.parse_args()
torch.manual_seed(0)
random.seed(0)
BLOCK = 128
import torch.distributed as dist

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
DEV = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
if WORLD > 1:
    dist.init_process_group("nccl")
    torch.cuda.set_device(DEV)


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
    # Written into a bf16 buffer a slab of block-rows at a time: a whole expert layer in fp32 is 2 GiB of transient,
    # which is what ran a 44 GiB card out of memory with the FP8 weights (35 GiB) resident.
    out = torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)
    wv = w.view(-1, O // bs[0], bs[0], I // bs[1], bs[1])
    sv = s.view(-1, O // bs[0], 1, I // bs[1], 1)
    ov = out.view(-1, O // bs[0], bs[0], I // bs[1], bs[1])
    per = max(1, (1 << 24) // (O * I))
    if per >= 1 and O * I <= (1 << 24):
        for e in range(0, wv.shape[0], per):  # several experts per slab
            ov[e : e + per] = (wv[e : e + per].float() * sv[e : e + per].float()).to(torch.bfloat16)
    else:
        step = max(1, (1 << 24) // (bs[0] * I))
        for e in range(wv.shape[0]):
            for r in range(0, O // bs[0], step):
                ov[e, r : r + step] = (wv[e, r : r + step].float() * sv[e, r : r + step].float()).to(torch.bfloat16)
    return out


def fp8_linear_train(
    input, weight, weight_scale_inv, block_size=None, activation_scale=None, bias=None, allow_deepgemm=True, **kw
):
    w = weight if weight.element_size() > 1 else dequant_blocks(weight, weight_scale_inv, block_size or (BLOCK, BLOCK))
    return F.linear(input, w, bias)


fp8.fp8_linear = fp8_linear_train  # used by FP8Linear.forward

from types import SimpleNamespace

from transformers.integrations import moe


def experts_forward_train(self, hidden_states, top_k_index, top_k_weights):
    """All experts of the layer dequantised once to bf16, then one grouped matmul per projection (differentiable).
    The per-expert Python loop it replaces spent ~16 s per example launching kernels."""
    bs = tuple(self.block_size or (BLOCK, BLOCK))
    view = SimpleNamespace(
        num_experts=self.num_experts,
        has_gate=self.has_gate,
        has_bias=False,
        is_transposed=False,
        gate_up_proj=dequant_blocks(self.gate_up_proj, self.gate_up_proj_scale_inv, bs),
        down_proj=dequant_blocks(self.down_proj, self.down_proj_scale_inv, bs),
        _apply_gate=self._apply_gate,
        act_fn=self.act_fn,
    )
    return moe.grouped_mm_experts_forward(view, hidden_states, top_k_index, top_k_weights)


fp8.FP8Experts.forward = experts_forward_train


class LoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(
            torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features)
        )
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.to(torch.float32), self.A), self.B) * self.scale).to(x.dtype)


tok = AutoTokenizer.from_pretrained(a.model)
# eager experts: the grouped-mm path needs a separate kernel package and would bypass the differentiable matmul above
model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map=DEV, experts_implementation="eager")
for p in model.parameters():
    p.requires_grad_(False)
if hasattr(model, "visual"):  # the vision tower is never called here; its memory is better spent on activations
    del model.visual
    torch.cuda.empty_cache()
lm = model.language_model if hasattr(model, "language_model") else model
targets = a.targets.split(",")
wrapped = []
for name, mod in list(lm.named_modules()):
    parts = name.split(".")
    if "layers" not in parts or parts.index("layers") + 1 >= len(parts):
        continue
    li = int(parts[parts.index("layers") + 1])
    if li < a.first_layer or ".experts." in name or name.endswith(".experts"):
        continue
    for t in targets:
        if name.endswith(t) and isinstance(mod, nn.Linear):
            parent = lm.get_submodule(name.rsplit(".", 1)[0])
            child = name.rsplit(".", 1)[1]
            setattr(parent, child, LoRA(mod, a.rank, a.alpha))
            wrapped.append(name)
            break
params = [p for n, p in lm.named_parameters() if n.endswith(".A") or n.endswith(".B")]
print("wrapped", len(wrapped), "trainable", sum(p.numel() for p in params), flush=True)
lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
lm.config.use_cache = False

from prismyra import readout
from prismyra.schema import Boolean, Choice

cfg = AutoConfig.from_pretrained(a.model)
hidden = getattr(cfg, "text_config", cfg).hidden_size
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
    # plan.token_ids follow question.options (readout.option_token_ids), so the gold index is taken in that order
    opts = list(q.options)
    gold_opt = ("yes" if ex["gold"] else "no") if ex["kind"] == "boolean" else L_[ex["gold"]]
    gold = opts.index(gold_opt)
    return ctx + suf, plan.token_ids, gold


data = json.load(open(a.data))
if a.n:
    data = data[: a.n]
data = data[RANK::WORLD]  # each rank a disjoint shard; gradients are averaged below
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
steps_total = a.epochs * len(data) // a.accum
step = 0
t0 = time.time()
run_loss = []
skipped = 0
sched = lambda s: min(1.0, s / 20) * max(0.1, 1 - s / max(1, steps_total))
lm.train()
for ep in range(a.epochs):
    random.shuffle(data)
    for i, ex in enumerate(data):
        ids, opt_ids, gold = render(ex)
        if len(ids) > a.max_tokens:
            skipped += 1
            continue
        x = torch.tensor([ids], device=DEV)
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0, -1]
        logits = (
            h.float()
            @ U[torch.tensor(sum(opt_ids, []) if isinstance(opt_ids[0], list) else opt_ids, device=DEV)].float().t()
        )
        loss = F.cross_entropy(logits[None], torch.tensor([gold], device=DEV)) / a.accum
        loss.backward()
        run_loss.append(loss.item() * a.accum)
        if (i + 1) % a.accum == 0:
            for g in opt.param_groups:
                g["lr"] = a.lr * sched(step)
            if WORLD > 1:
                for p_ in params:
                    if p_.grad is None:
                        p_.grad = torch.zeros_like(p_)
                    dist.all_reduce(p_.grad)
                    p_.grad /= WORLD
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0 and RANK == 0:
                el = time.time() - t0
                print(
                    f"step {step}/{steps_total} loss {sum(run_loss[-a.accum * a.log_every :]) / len(run_loss[-a.accum * a.log_every :]):.4f} "
                    f"ex/s {(i + 1 + ep * len(data)) / el:.2f} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB skipped {skipped}",
                    flush=True,
                )
            if step % 200 == 0 and RANK == 0:
                torch.save(
                    {n: p.detach().cpu() for n, p in lm.named_parameters() if n.endswith(".A") or n.endswith(".B")},
                    a.out + ".partial",
                )
if RANK != 0:
    sys.exit(0)
state = {n: p.detach().cpu() for n, p in lm.named_parameters() if n.endswith(".A") or n.endswith(".B")}
torch.save(
    {"lora": state, "rank": a.rank, "alpha": a.alpha, "targets": targets, "wrapped": wrapped, "args": vars(a)}, a.out
)
print("saved", a.out, "steps", step, "skipped", skipped, "minutes", round((time.time() - t0) / 60, 1), flush=True)
