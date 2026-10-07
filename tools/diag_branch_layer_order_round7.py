"""inv5, round 7 (continued): `diag_prefill_divergence_round7.py` proved prefill is clean (document 0's GDN
state and attention KV are bit-identical, alone vs 7 companions, before any question). `diag_branch_attn_io_
round7.py` then found the BRANCH pass's attention inputs (q/k/v) already differ by the time the first attention
layer in the branch runs -- so whatever causes the qn=1 residual lives in the branch pass, upstream of its own
first attention call. This model's layer 0 is GDN (fp8spd4's finding, RUN-inv.md 13 section), so the branch's
own GDN/conv/norm layers run before its first attention call and are the natural next suspect.

Hooks every GDN (`chunk_gated_delta_rule`), conv1d (`causal_depthwise_conv1d`) and norm (`FastRMSNorm`/
`FusedGatedRMSNorm`) call *in temporal (layer) order* for document 0's own branch row only, across the whole
`ask()`-vs-`open_batch` comparison (prefill + branch together, so the call index tells which layer and which
phase unambiguously), and reports the first one whose output for document 0's row differs.

Usage: PRISMYRA_MODEL=<repo> python3 diag_branch_layer_order_round7.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402
from prismyra import varlen  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
GROUP_SIZE = int(os.environ.get("DIAG_GROUP_SIZE", "8"))

CONTEXTS = [
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
    "seller when the item is faulty and by the buyer otherwise.",
    "Gift cards never expire and are not redeemable for cash. A lost gift card is replaced only with the original "
    "purchase receipt and a matching photo ID.",
    "Membership renews automatically each year unless cancelled thirty days before the renewal date. A refund is "
    "given only for the unused portion of the current year.",
    "Warranty claims require the original receipt and are void if the device was opened by anyone other than an "
    "authorised technician.",
    "Delivery windows are estimates, not guarantees, and a delay of under five business days does not qualify for "
    "a shipping refund.",
    "Price matching applies only to identical items sold directly by a competitor, not to marketplace sellers or "
    "clearance stock.",
    "A subscription paused for more than ninety days is treated as cancelled and restarting it requires a new "
    "sign-up at the current price.",
    "Store credit never expires but is forfeited if the account it was issued to is closed for inactivity.",
]
ONE_QUESTION = Boolean(id="target", prompt="Is a refund limited to unopened items?")
COMPANION_QUESTION = Boolean(id="companion", prompt="Does this clause mention a time limit?")

SCENARIO = {"name": None, "phase": "prefill"}
RECORDS: dict[str, list[dict]] = {}


def _slice_for_target(t: torch.Tensor, target_rows: int) -> torch.Tensor | None:
    if not torch.is_tensor(t) or t.dim() == 0:
        return None
    boundaries = varlen.current()
    if boundaries is not None and SCENARIO["phase"] == "prefill":
        n = boundaries.lengths[0]
        for dim, size in enumerate(t.shape):
            if size == boundaries.tokens:
                idx = [slice(None)] * t.dim()
                idx[dim] = slice(0, n)
                return t[tuple(idx)]
        return None
    if t.shape[0] < target_rows:
        return None
    return t[:target_rows]


def _record(name: str, target_rows: int, output) -> None:
    scenario = SCENARIO["name"]
    if scenario is None:
        return
    tensors = output if isinstance(output, (tuple, list)) else (output,)
    sliced = [_slice_for_target(t, target_rows) for t in tensors]
    sliced = [s for s in sliced if s is not None]
    if not sliced:
        return
    RECORDS[scenario].append({
        "name": name, "phase": SCENARIO["phase"],
        "tensors": [s.detach().to("cpu", torch.float32).clone() for s in sliced],
    })


def _wrap_module_forward(module, name: str, target_rows: int) -> None:
    original = module.forward

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        _record(name, target_rows, out)
        return out

    module.forward = wrapped


def _wrap_function(obj, attr: str, name: str, target_rows: int):
    original = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        _record(name, target_rows, out)
        return out

    setattr(obj, attr, wrapped)


def install_hooks(engine: Prismyra, target_rows: int) -> None:
    hook_types = ("FastRMSNorm", "FusedGatedRMSNorm", "Qwen3_5MoeGatedDeltaNet", "FlashAttention")
    layer_idx_by_mod: dict[int, int] = {}
    for mod_name, module in engine.backbone.named_modules():
        type_name = type(module).__name__
        if type_name not in hook_types:
            continue
        # mod_name like "language_model.layers.3.linear_attn" -- pull the layer number out so the report sorts
        # by depth, not by hook-installation order.
        parts = mod_name.split(".")
        layer_num = next((p for p in parts if p.isdigit()), "?")
        _wrap_module_forward(module, f"L{layer_num}.{type_name}", target_rows)

    import vllm.third_party.flash_linear_attention.ops.chunk as _chunk
    _wrap_function(_chunk, "chunk_gated_delta_rule", "fn:chunk_gated_delta_rule", target_rows)
    from prismyra.kernels import conv as _conv
    _wrap_function(_conv, "causal_depthwise_conv1d", "fn:causal_depthwise_conv1d", target_rows)


def run_scenario(engine: Prismyra, name: str, group_size: int, target_rows: int) -> None:
    RECORDS[name] = []
    SCENARIO["name"] = name
    SCENARIO["phase"] = "prefill"
    docs = CONTEXTS[:group_size]
    with engine.open_batch(docs) as batch:
        SCENARIO["phase"] = "branch"
        per_doc = [[ONE_QUESTION]] + [[COMPANION_QUESTION] for _ in docs[1:]]
        results = batch.ask(per_doc)
        want = results[0]
    SCENARIO["name"] = None
    print(f"{name}: group_size={group_size}, {len(RECORDS[name])} hook events, "
          f"final p={want['target'].probabilities}", flush=True)


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    import prismyra.engine as _engine_mod
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, "dispatcher not registered -- see round-7 13.5 note"
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    install_hooks(engine, target_rows=1)
    run_scenario(engine, "alone", 1, target_rows=1)
    run_scenario(engine, f"group{GROUP_SIZE}", GROUP_SIZE, target_rows=1)

    a, b = RECORDS["alone"], RECORDS[f"group{GROUP_SIZE}"]
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"WARNING: event count differs ({len(a)} vs {len(b)}) -- comparing the first {n} only")
    first = None
    for i in range(n):
        ea, eb = a[i], b[i]
        if ea["name"] != eb["name"] or ea["phase"] != eb["phase"]:
            print(f"  #{i}: name/phase misaligned ({ea['name']}/{ea['phase']} vs {eb['name']}/{eb['phase']}) "
                  f"-- stopping alignment here")
            break
        for j, (ta, tb) in enumerate(zip(ea["tensors"], eb["tensors"])):
            if ta.shape != tb.shape:
                print(f"  #{i} {ea['name']} ({ea['phase']})[{j}]: shape differs {tuple(ta.shape)} vs {tuple(tb.shape)}")
                continue
            if not torch.equal(ta, tb):
                d = (ta - tb).abs().max().item()
                print(f"  #{i} {ea['name']} ({ea['phase']})[{j}]: MISMATCH max|diff|={d:.3e}"
                      f"{'  <-- FIRST DIVERGENCE' if first is None else ''}")
                if first is None:
                    first = (i, ea["name"], ea["phase"], d)
    if first is None:
        print("no mismatch found in any hooked GDN/norm/attention call (bit-identical through everything hooked)")
    else:
        print(f"\nFIRST DIVERGENT EVENT: #{first[0]} {first[1]} (phase={first[2]}) max|diff|={first[3]:.3e}")


if __name__ == "__main__":
    main()
