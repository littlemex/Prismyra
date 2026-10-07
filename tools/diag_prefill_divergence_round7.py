"""inv5, round 7: isolate whether the qn=1 residual audit_sm120.py/diag_round6_matrix.py still shows (26/364,
worst move 0.093305 with unified_attention, same count as FA2 -- see RUN-inv.md round 6) comes from the
*branch-read* (the step round 6 already changed the attention kernel for, with no effect on the failure count) or
from *prefill itself* -- `Prismyra.open_batch` concatenates every companion document's tokens into one flat
`torch.cat` and runs the *whole* 36-layer backbone once over the combined sequence (`prismyra/engine.py`
`open_batch`, the `self.backbone(input_ids=ids, ...)` call), so a document's own stored keys/values and GDN
recurrent state are written by a kernel launch whose total token count depends on its companions, before any
question is ever asked. `ask()` (the ground truth) never shares that launch with anything.

This script never asks a question. It opens the same single document (CONTEXTS[0]) through `open_batch` alone
(group_size=1, audit_sm120's own non-failing case) and through `open_batch` with real companions (group_size=8,
one of the 26 failing cells), and diffs the *admitted* cache state for document 0 between the two -- before
`batch.ask()` is ever called. If this differs, the residual is a prefill-time, not a branch-read-time, effect.

Captures, for document 0 only, right after the prefill forward and before any fork/question:
  - every GDN layer's `recurrent_states`/`conv_states` (via `fork.snapshot`/`fork.pick`, the same mechanism
    `open_batch` itself uses internally -- intercepted rather than reimplemented, so this cannot drift from what
    the engine actually does).
  - every attention (`PagedForkLayer`) layer's real keys/values, read out of the shared page pool through that
    document's own `block_table`/`seqused` (not covered by `fork.snapshot`, which only clones per-row state and
    explicitly skips `keys`/`values` for layers that `_holds_attention`).

Usage: PRISMYRA_MODEL=<repo> python3 diag_prefill_divergence_round7.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra, fork  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

# Same 8 inline documents as diag_round6_matrix.py / diag_qn1_residual_round5.py, so this round's finding sits on
# exactly the same fixture the 26/364 count was measured on.
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

CAPTURED: dict[str, dict] = {}
LAST_CACHE: list = []


def install_hooks() -> None:
    original_snapshot = fork.snapshot

    def wrapped_snapshot(cache):
        snap = original_snapshot(cache)
        LAST_CACHE.append(cache)
        CAPTURED["snapshot"] = snap
        return snap

    fork.snapshot = wrapped_snapshot
    import prismyra.engine as _engine_mod

    _engine_mod.snapshot = wrapped_snapshot


def attention_kv_for_row(cache, row: int) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Document `row`'s real keys/values, read out of each attention layer's shared page pool through its own
    `block_table`/`seqused` -- the data `fork.snapshot` does not capture (it only clones per-row recurrent state;
    attention layers are addressed by page, not by row, see `fork._holds_attention`)."""
    out = []
    for i, layer in enumerate(cache.layers):
        if not getattr(layer, "holds_attention", False):
            continue
        table = layer.block_table[row]
        n = int(layer.seqused[row].item())
        BLOCK = 16
        pages_needed = -(-n // BLOCK)
        ks, vs = [], []
        remaining = n
        for p in range(pages_needed):
            page = int(table[p].item())
            take = min(BLOCK, remaining)
            ks.append(layer.keys[page, :take])
            vs.append(layer.values[page, :take])
            remaining -= take
        k = torch.cat(ks, dim=0) if ks else layer.keys[:0]
        v = torch.cat(vs, dim=0) if vs else layer.values[:0]
        out.append((i, k.detach().to("cpu", torch.float32).clone(), v.detach().to("cpu", torch.float32).clone()))
    return out


def run_scenario(engine: Prismyra, name: str, group_size: int) -> None:
    docs = CONTEXTS[:group_size]
    LAST_CACHE.clear()
    with engine.open_batch(docs) as batch:
        cache = LAST_CACHE[-1]
        doc0_snapshot = fork.pick(CAPTURED["snapshot"], 0)
        doc0_kv = attention_kv_for_row(cache, 0)
    CAPTURED[name] = {"gdn": doc0_snapshot, "kv": doc0_kv}
    print(f"{name}: group_size={group_size}, captured {len(doc0_snapshot)} cache layers, "
          f"{len(doc0_kv)} attention layers' kv", flush=True)


def diff_gdn(a: dict, b: dict) -> tuple[int, int, float]:
    checked = 0
    mismatches = 0
    worst = 0.0
    layer_ids = sorted(set(a) | set(b))
    first = None
    for i in layer_ids:
        ea, eb = a.get(i, {}), b.get(i, {})
        for attr in ("recurrent_states", "conv_states"):
            da, db = ea.get(attr, {}), eb.get(attr, {})
            for key in sorted(set(da) | set(db)):
                ta, tb = da.get(key), db.get(key)
                if ta is None or tb is None:
                    continue
                if ta.shape != tb.shape:
                    print(f"  layer {i} {attr}[{key}]: shape differs {tuple(ta.shape)} vs {tuple(tb.shape)}")
                    continue
                checked += 1
                ta32 = ta.detach().to("cpu", torch.float32)
                tb32 = tb.detach().to("cpu", torch.float32)
                if not torch.equal(ta32, tb32):
                    mismatches += 1
                    d = (ta32 - tb32).abs().max().item()
                    worst = max(worst, d)
                    if first is None:
                        first = (i, attr, key, d)
                    print(f"  layer {i} {attr}[{key}]: MISMATCH max|diff|={d:.3e}")
    if first:
        print(f"  FIRST GDN-STATE DIVERGENCE: layer {first[0]} {first[1]}[{first[2]}] max|diff|={first[3]:.3e}")
    return checked, mismatches, worst


def diff_kv(a: list, b: list) -> tuple[int, int, float]:
    checked = 0
    mismatches = 0
    worst = 0.0
    first = None
    a_by_layer = {i: (k, v) for i, k, v in a}
    b_by_layer = {i: (k, v) for i, k, v in b}
    for i in sorted(set(a_by_layer) | set(b_by_layer)):
        if i not in a_by_layer or i not in b_by_layer:
            print(f"  attention layer {i}: present in only one scenario")
            continue
        ka, va = a_by_layer[i]
        kb, vb = b_by_layer[i]
        if ka.shape != kb.shape:
            print(f"  attention layer {i}: shape differs {tuple(ka.shape)} vs {tuple(kb.shape)} (context padding?)")
            continue
        checked += 1
        k_eq = torch.equal(ka, kb)
        v_eq = torch.equal(va, vb)
        if not (k_eq and v_eq):
            mismatches += 1
            d = max((ka - kb).abs().max().item(), (va - vb).abs().max().item())
            worst = max(worst, d)
            if first is None:
                first = (i, d)
            print(f"  attention layer {i}: KV MISMATCH max|diff|={d:.3e} (keys_eq={k_eq}, values_eq={v_eq})")
    if first:
        print(f"  FIRST ATTENTION-KV DIVERGENCE: layer {first[0]} max|diff|={first[1]:.3e}")
    return checked, mismatches, worst


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    import prismyra.engine as _engine_mod
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, (
        "the batch-invariant dispatcher is not registered -- a diagnostic run under this condition would blame "
        "the wrong op for a difference that is really just unprotected cuBLAS/F.linear (inv5, round 7, 13.5: a "
        "whole minimal-MoE 'repro' turned out to be exactly this mistake)"
    )
    install_hooks()
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    run_scenario(engine, "alone", 1)
    run_scenario(engine, "group8", 8)

    print("\n=== GDN recurrent/conv state, document 0, alone vs group8, before any question ===")
    g_checked, g_mismatch, g_worst = diff_gdn(CAPTURED["alone"]["gdn"], CAPTURED["group8"]["gdn"])
    print(f"GDN: {g_checked} tensors checked, {g_mismatch} mismatched, worst max|diff|={g_worst:.3e}")

    print("\n=== Attention keys/values, document 0, alone vs group8, before any question ===")
    k_checked, k_mismatch, k_worst = diff_kv(CAPTURED["alone"]["kv"], CAPTURED["group8"]["kv"])
    print(f"KV: {k_checked} layers checked, {k_mismatch} mismatched, worst max|diff|={k_worst:.3e}")

    print("\n=== Verdict ===")
    if g_mismatch == 0 and k_mismatch == 0:
        print("Prefill-time cache state for document 0 is bit-identical whether it is read alone or with 7 "
              "companions. The qn=1 residual, if it exists for this fixture, is NOT a prefill-time effect; it "
              "must come from the branch/question-answering pass itself.")
    else:
        print("Prefill-time cache state for document 0 ALREADY DIFFERS before any question is asked. The qn=1 "
              "residual is (at least partly) a PREFILL-TIME effect: `open_batch` concatenates every companion's "
              "tokens into one flat pass through the whole backbone, and the per-document state that pass writes "
              "depends on who else shares that pass.")


if __name__ == "__main__":
    main()
