"""inv5, round 7 (part 2): for the qn=1 residual (`tools/diag_round6_matrix.py`: 364 checks, 26 non-exact, all
at question-count=1, worst move 0.093305 with `unified_attention`), decide between two possibilities at the
*branch-read* attention call itself (`FlashAttention.forward`'s `block_table`/`seqused_k` path, the one round 6
(commit `35928d3`) already switched from FA2 to `unified_attention`):

  (A) the call's own INPUTS (this document's query, the keys/values its `block_table`/`seqused_k` actually name,
      `max_seqlen_k`/capacity) are bit-identical whether this document answers alone or alongside companions, and
      the kernel's OUTPUT still differs -- true kernel-internal row-count/companion dependence (the schedule the
      kernel chooses depends on how many other rows share this exact launch, e.g. a decode-shaped split-KV
      heuristic keyed on `num_seqs`).
  (B) the inputs themselves already differ (something upstream constructs a different query/key/value/capacity
      for the same document depending on who else is in the batch) -- in which case the kernel is not the bug,
      whatever builds its call is.

Monkeypatches `vllm.v1.attention.ops.triton_unified_attention.unified_attention` (the import is local inside
`FlashAttention.forward`, resolved at call time, so patching the module attribute before any forward pass is
enough) to record every call's `q`/`k`/`v`/`seqused_k`/`max_seqlen_k`/`block_table` and its `out`, sliced to the
one row this script's target document occupies in each scenario (row 0 in both -- `CONTEXTS[0]` is listed first
in each `open_batch` call).

Usage: PRISMYRA_MODEL=<repo> python3 diag_branch_attn_io_round7.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402

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

TARGET_ROW = 0
SCENARIO = {"name": None}
CALLS: dict[str, list[dict]] = {}


def install_hook() -> None:
    import vllm.v1.attention.ops.triton_unified_attention as _ua_mod

    original = _ua_mod.unified_attention

    def wrapped(**kwargs):
        out = kwargs["out"]
        before = out.clone()
        original(**kwargs)
        name = SCENARIO["name"]
        if name is not None:
            rows = kwargs["q"].shape[0]
            # qn=1: one row per document in this call, so TARGET_ROW indexes the document directly.
            if TARGET_ROW < rows:
                seqused_k = kwargs["seqused_k"]
                block_table = kwargs["block_table"]
                n = int(seqused_k[TARGET_ROW].item())
                table_row = block_table[TARGET_ROW, : -(-n // 16)].detach().to("cpu").clone()
                record = {
                    "rows_in_call": rows,
                    "q": kwargs["q"][TARGET_ROW].detach().to("cpu", torch.float32).clone(),
                    "seqused_k": n,
                    "max_seqlen_k": kwargs["max_seqlen_k"],
                    "block_table_row": table_row,
                    "out_before": before[TARGET_ROW].detach().to("cpu", torch.float32).clone(),
                    "out_after": out[TARGET_ROW].detach().to("cpu", torch.float32).clone(),
                    # The actual K/V bytes this row's own pages hold -- read through the *same* pool and table
                    # the kernel itself was just given, not re-derived.
                    "k_pool_slice": _gather_pool(kwargs["k"], table_row, n),
                    "v_pool_slice": _gather_pool(kwargs["v"], table_row, n),
                }
                CALLS.setdefault(name, []).append(record)

    _ua_mod.unified_attention = wrapped


def _gather_pool(pool: torch.Tensor, table_row: torch.Tensor, n: int) -> torch.Tensor:
    BLOCK = 16
    parts = []
    remaining = n
    for p in table_row.tolist():
        take = min(BLOCK, remaining)
        if take <= 0:
            break
        parts.append(pool[p, :take].detach().to("cpu", torch.float32).clone())
        remaining -= take
    return torch.cat(parts, dim=0) if parts else pool[:0].detach().to("cpu", torch.float32).clone()


def run_scenario(engine: Prismyra, name: str, group_size: int) -> None:
    SCENARIO["name"] = name
    CALLS[name] = []
    docs = CONTEXTS[:group_size]
    with engine.open_batch(docs) as batch:
        per_doc = [[ONE_QUESTION]] + [[dataclasses.replace(COMPANION_QUESTION, id=f"c{i}")] for i in range(1, group_size)]
        results = batch.ask(per_doc)
        want = results[0]
    SCENARIO["name"] = None
    print(f"{name}: group_size={group_size}, captured {len(CALLS[name])} unified_attention call(s) touching row "
          f"{TARGET_ROW}, final answer={want['target'].option!r} p={want['target'].probabilities}", flush=True)


def compare(a_name: str, b_name: str) -> None:
    a_calls, b_calls = CALLS[a_name], CALLS[b_name]
    print(f"\n=== {a_name} ({len(a_calls)} calls) vs {b_name} ({len(b_calls)} calls), document row {TARGET_ROW} ===")
    n = min(len(a_calls), len(b_calls))
    if len(a_calls) != len(b_calls):
        print(f"  WARNING: call count differs ({len(a_calls)} vs {len(b_calls)}) -- comparing the first {n} only")
    for i in range(n):
        a, b = a_calls[i], b_calls[i]
        print(f"  call#{i}: rows_in_call {a['rows_in_call']} vs {b['rows_in_call']}, "
              f"max_seqlen_k(capacity) {a['max_seqlen_k']} vs {b['max_seqlen_k']}, "
              f"seqused_k {a['seqused_k']} vs {b['seqused_k']}")
        for key in ("q", "k_pool_slice", "v_pool_slice", "out_before"):
            ta, tb = a[key], b[key]
            if ta.shape != tb.shape:
                print(f"    {key}: shape differs {tuple(ta.shape)} vs {tuple(tb.shape)}")
                continue
            eq = torch.equal(ta, tb)
            tag = "input" if key != "out_before" else "out_before(garbage buffer, informational only)"
            if eq:
                print(f"    {key} ({tag}): IDENTICAL")
            else:
                d = (ta - tb).abs().max().item()
                print(f"    {key} ({tag}): DIFFERS max|diff|={d:.3e}")
        oa, ob = a["out_after"], b["out_after"]
        if oa.shape == ob.shape:
            eq = torch.equal(oa, ob)
            d = 0.0 if eq else (oa - ob).abs().max().item()
            print(f"    out_after (kernel's own output for this row): {'IDENTICAL' if eq else f'DIFFERS max|diff|={d:.3e}'}")
        verdict_inputs_identical = all(
            a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in ("q", "k_pool_slice", "v_pool_slice")
        ) and a["seqused_k"] == b["seqused_k"]
        if verdict_inputs_identical and oa.shape == ob.shape and not torch.equal(oa, ob):
            print("    VERDICT: inputs identical, output differs -- the kernel itself is not row-count/"
                  "companion-count invariant for this call.")
        elif not verdict_inputs_identical:
            print("    VERDICT: inputs already differ before the kernel runs -- look upstream of attention.")


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    import prismyra.engine as _engine_mod
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, (
        "the batch-invariant dispatcher is not registered -- see diag_prefill_divergence_round7.py's note "
        "(inv5, round 7, 13.5) on why this assert exists"
    )
    install_hook()
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    run_scenario(engine, "alone", 1)
    run_scenario(engine, f"group{GROUP_SIZE}", GROUP_SIZE)
    compare("alone", f"group{GROUP_SIZE}")


if __name__ == "__main__":
    main()
