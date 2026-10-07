"""inv5, round 7 (continued): `diag_branch_layer_order_round7.py` found the FIRST divergence in the whole
alone-vs-group8 comparison at `L0.FusedGatedRMSNorm` during the BRANCH pass (max|diff|=3.052e-05, bf16-ULP
scale), with everything in prefill and the start of the branch (through GDN's own chunk call) hidden inside the
`Qwen3_5MoeGatedDeltaNet` module-level hook -- that hook only reports the *whole* layer's output, so it cannot
say which of the five sub-kernels `chunk_gated_delta_rule_fwd` calls (`chunk_local_cumsum`, `chunk_scaled_dot_
kkt_fwd`, `solve_tril`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o`) is the first to
diverge for the branch's own (short, cu_seqlens-packed) call. This hooks each of those six by name inside
`vllm.third_party.flash_linear_attention.ops.chunk`, for document 0's own row/token range only, layer 0 only
(the smallest reproducible unit of the round-7 finding).

Usage: PRISMYRA_MODEL=<repo> python3 diag_gdn_branch_subkernel_round7.py
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

SCENARIO = {"name": None, "phase": "prefill", "branch_gdn_layer_count": 0}
RECORDS: dict[str, list[dict]] = {}
SUB_KERNELS = (
    "chunk_local_cumsum", "chunk_scaled_dot_kkt_fwd", "solve_tril", "recompute_w_u_fwd",
    "chunk_gated_delta_rule_fwd_h", "chunk_fwd_o",
)


def _record(name: str, output) -> None:
    scenario = SCENARIO["name"]
    if scenario is None or SCENARIO["phase"] != "branch":
        return
    if name == "chunk_local_cumsum":
        # Always the first of the 6 sub-kernels called inside `chunk_gated_delta_rule_fwd`, so this marks a new
        # GDN layer's branch call starting. Layer 0 is call-count 1.
        SCENARIO["branch_gdn_layer_count"] += 1
    if SCENARIO["branch_gdn_layer_count"] != 1:
        return
    tensors = output if isinstance(output, (tuple, list)) else (output,)
    sliced = []
    for t in tensors:
        if not torch.is_tensor(t) or t.dim() == 0:
            continue
        # Document 0's own branch tokens are the leading rows/tokens of whatever axis matches the flat varlen
        # token count -- same slicing convention as the round-5/6/7 scripts, generalised to any tensor rank.
        boundaries = varlen.current()
        if boundaries is None:
            continue
        n = boundaries.lengths[0]
        matched = False
        for dim, size in enumerate(t.shape):
            if size == boundaries.tokens:
                idx = [slice(None)] * t.dim()
                idx[dim] = slice(0, n)
                sliced.append(t[tuple(idx)].detach().to("cpu", torch.float32).clone())
                matched = True
                break
        if not matched and t.shape[0] >= n:
            sliced.append(t[:n].detach().to("cpu", torch.float32).clone())
    if sliced:
        RECORDS[scenario].append({"name": name, "tensors": sliced})


def install_hooks(engine: Prismyra) -> None:
    import vllm.third_party.flash_linear_attention.ops.chunk as _chunk_mod

    for fn_name in SUB_KERNELS:
        original = getattr(_chunk_mod, fn_name)

        def make_wrapped(original=original, fn_name=fn_name):
            def wrapped(*args, **kwargs):
                out = original(*args, **kwargs)
                _record(fn_name, out)
                return out
            return wrapped

        setattr(_chunk_mod, fn_name, make_wrapped())


def run_scenario(engine: Prismyra, name: str, group_size: int) -> None:
    RECORDS[name] = []
    SCENARIO["name"] = name
    SCENARIO["phase"] = "prefill"
    SCENARIO["branch_gdn_layer_count"] = 0
    docs = CONTEXTS[:group_size]
    with engine.open_batch(docs) as batch:
        SCENARIO["phase"] = "branch"
        per_doc = [[ONE_QUESTION]] + [[COMPANION_QUESTION] for _ in docs[1:]]
        results = batch.ask(per_doc)
        want = results[0]
    SCENARIO["name"] = None
    print(f"{name}: group_size={group_size}, {len(RECORDS[name])} sub-kernel events (layer 0 branch only), "
          f"final p={want['target'].probabilities}", flush=True)


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    import prismyra.engine as _engine_mod
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, "dispatcher not registered -- see round-7 13.5 note"
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    install_hooks(engine)
    run_scenario(engine, "alone", 1)
    run_scenario(engine, f"group{GROUP_SIZE}", GROUP_SIZE)

    a, b = RECORDS["alone"], RECORDS[f"group{GROUP_SIZE}"]
    n = min(len(a), len(b))
    first = None
    for i in range(n):
        ea, eb = a[i], b[i]
        if ea["name"] != eb["name"]:
            print(f"  #{i}: name misaligned ({ea['name']} vs {eb['name']}) -- stopping")
            break
        for j, (ta, tb) in enumerate(zip(ea["tensors"], eb["tensors"])):
            if ta.shape != tb.shape:
                print(f"  #{i} {ea['name']}[{j}]: shape differs {tuple(ta.shape)} vs {tuple(tb.shape)}")
                continue
            eq = torch.equal(ta, tb)
            if not eq:
                d = (ta - tb).abs().max().item()
                marker = "  <-- FIRST DIVERGENCE" if first is None else ""
                print(f"  #{i} {ea['name']}[{j}]: MISMATCH max|diff|={d:.3e}{marker}")
                if first is None:
                    first = (i, ea["name"], j, d)
            else:
                print(f"  #{i} {ea['name']}[{j}]: identical")
    if first is None:
        print("no mismatch found among the 6 GDN sub-kernels for layer 0's branch call "
              "(all bit-identical -- the divergence must be in something these hooks don't cover, e.g. the "
              "projections feeding `chunk_gated_delta_rule_fwd` or the fused-norm-gate step itself)")
    else:
        print(f"\nFIRST DIVERGENT SUB-KERNEL: #{first[0]} {first[1]} tensor#{first[2]} max|diff|={first[3]:.3e}")


if __name__ == "__main__":
    main()
