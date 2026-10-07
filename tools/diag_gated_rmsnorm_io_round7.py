"""inv5, round 7, chair's next step (a): `diag_branch_layer_order_round7.py` found the first divergence in the
whole alone-vs-group8 comparison at `L0.FusedGatedRMSNorm`'s OUTPUT during the branch pass (max|diff|=3.052e-05).
This hooks `FusedGatedRMSNorm.forward`'s own ARGUMENTS (hidden_states = the GDN core's raw output, gate = z) for
document 0's row, layer 0, branch phase only, to tell apart:

  - inputs identical, output differs -> the kernel itself (`prismyra/kernels/qwen3_moe.py`'s own
    `_gated_rms_norm` Triton kernel, docs/KERNELS.md 118-124) is not row-count invariant, despite looking
    row-independent by construction (grid=(x.shape[0],), one program per row, BLOCK/num_warps fixed by `width`
    alone -- no `@triton.autotune`, nothing keyed on M). Would be surprising and worth confirming on the pin
    table / looking for anything that reads neighbouring rows.
  - inputs already differ -> the GDN core (`chunk_gated_delta_rule`'s varlen branch call) is where the
    residual enters, and the chair's step (b) (an isolated varlen repro with the dispatcher asserted active)
    is the next move.

Usage: PRISMYRA_MODEL=<repo> python3 diag_gated_rmsnorm_io_round7.py
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

SCENARIO = {"name": None, "phase": "prefill", "branch_calls": 0}
RECORDS: dict[str, dict] = {}


def install_hook(engine: Prismyra) -> None:
    target = None
    for mod_name, module in engine.backbone.named_modules():
        if type(module).__name__ != "FusedGatedRMSNorm":
            continue
        layer_num = next((p for p in mod_name.split(".") if p.isdigit()), None)
        if layer_num == "0":
            target = module
            break
    assert target is not None, "could not find layer 0's FusedGatedRMSNorm module"

    original = target.forward

    def _slice_doc0(t: torch.Tensor, n: int, total_tokens: int) -> torch.Tensor:
        for dim, size in enumerate(t.shape):
            if size == total_tokens:
                idx = [slice(None)] * t.dim()
                idx[dim] = slice(0, n)
                return t[tuple(idx)]
        return t[:n]  # fallback: assume dim 0 is the flat token axis

    def wrapped(hidden_states, gate=None):
        out = original(hidden_states, gate)
        scenario = SCENARIO["name"]
        if scenario is not None and SCENARIO["phase"] == "branch":
            SCENARIO["branch_calls"] += 1
            if SCENARIO["branch_calls"] == 1:
                boundaries = varlen.current()
                n = boundaries.lengths[0] if boundaries is not None else 1
                total = boundaries.tokens if boundaries is not None else hidden_states.shape[0]
                RECORDS[scenario] = {
                    "hidden_states": _slice_doc0(hidden_states, n, total).detach().to("cpu", torch.float32).clone(),
                    "gate": (_slice_doc0(gate, n, total).detach().to("cpu", torch.float32).clone()
                             if gate is not None else None),
                    "out": _slice_doc0(out, n, total).detach().to("cpu", torch.float32).clone(),
                }
        return out

    target.forward = wrapped


def run_scenario(engine: Prismyra, name: str, group_size: int) -> None:
    SCENARIO["name"] = name
    SCENARIO["phase"] = "prefill"
    SCENARIO["branch_calls"] = 0
    docs = CONTEXTS[:group_size]
    with engine.open_batch(docs) as batch:
        SCENARIO["phase"] = "branch"
        per_doc = [[ONE_QUESTION]] + [[COMPANION_QUESTION] for _ in docs[1:]]
        results = batch.ask(per_doc)
        want = results[0]
    SCENARIO["name"] = None
    print(f"{name}: group_size={group_size}, final p={want['target'].probabilities}", flush=True)


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    import prismyra.engine as _engine_mod
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, "dispatcher not registered -- see round-7 13.5 note"
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    install_hook(engine)
    run_scenario(engine, "alone", 1)
    run_scenario(engine, f"group{GROUP_SIZE}", GROUP_SIZE)

    a, b = RECORDS["alone"], RECORDS[f"group{GROUP_SIZE}"]
    for key in ("hidden_states", "gate", "out"):
        ta, tb = a[key], b[key]
        if ta is None or tb is None:
            print(f"{key}: one side is None")
            continue
        if ta.shape != tb.shape:
            print(f"{key}: shape differs {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        eq = torch.equal(ta, tb)
        if eq:
            print(f"{key}: IDENTICAL")
        else:
            d = (ta - tb).abs().max().item()
            print(f"{key}: DIFFERS max|diff|={d:.3e}")

    inputs_ok = torch.equal(a["hidden_states"], b["hidden_states"]) and (
        a["gate"] is None or torch.equal(a["gate"], b["gate"])
    )
    print()
    if inputs_ok:
        print("VERDICT: FusedGatedRMSNorm's own inputs (GDN core output, gate z) are identical -- the kernel "
              "itself is where the row-count dependence enters. Check its grid/BLOCK/num_warps choice and the "
              "pin table (docs/KERNELS.md 118-124).")
    else:
        print("VERDICT: FusedGatedRMSNorm's inputs already differ -- the GDN core (chunk_gated_delta_rule's "
              "varlen branch call) is where the residual enters. Proceed to the isolated varlen repro (chair's "
              "step b).")


if __name__ == "__main__":
    main()
