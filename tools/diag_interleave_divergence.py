"""fp8spd4: find the first layer/op where `interleaved_fork=True` diverges from the two-pass baseline for a
question count that fails torch.equal (sm_120: n in {2,3,16}; RUN-fp8spd.md round4). Modeled on inv's
`diag_layer_op_divergence_round4.py` (RUN-inv.md 9.1 row 12) -- per-module forward hooks keyed by (layer_idx,
kind), diffed call-by-call between two scenarios run in the *same* process so the only thing that differs is
which code path produced the tensors.

Key idea that makes the two scenarios comparable despite running in a different global order (two_pass calls
every layer's context forward, then every layer's branch forward; interleaved calls one layer's context then
that same layer's branch, 36 times): hook each layer's own `self_attn`/`linear_attn`/`mlp` submodule by
identity. The *first* call recorded against a given layer in either scenario is that layer's context forward,
the *second* is its branch forward -- true in both scenarios because each layer object only ever gets a context
call and a branch call once per `ask()`. So comparing "call #1 of layer i" to "call #1 of layer i" (and #2 to
#2) across scenarios is apples-to-apples without needing to track interleaving order.

For `mlp`, two_pass calls it once per layer on the context's rows alone and once per layer on the branch's rows
alone (two calls, like self_attn/linear_attn); interleaved calls it once per layer on the concatenated rows (one
call). The concatenated call's output is split by this script before comparison, so there are still two
comparable slices (ctx-rows, branch-rows) per layer for `mlp` too.

Usage: PRISMYRA_TEST_MODEL=... python3 diag_interleave_divergence.py [n_questions]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Choice, Prismyra  # noqa: E402

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

CONTEXT = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
    "seller when the item is faulty and by the buyer otherwise. Gift cards never expire and are not redeemable "
    "for cash. A lost gift card is replaced only with the original purchase receipt and a matching photo ID. "
    "Loyalty points accrue at one point per dollar spent and expire after twenty-four months of inactivity. "
    "Price-matching is honored within fourteen days of purchase for identical items sold by the same retailer."
)

ALL_QUESTIONS = [
    Boolean(id="faulty", prompt="Does the seller pay return shipping on a faulty item?"),
    Choice(
        id="opened",
        prompt="What happens to an opened item?\nA. Refunded\nB. Exchanged\nC. Kept\nD. Discarded",
        choices=["A", "B", "C", "D"],
    ),
    Boolean(id="giftcard", prompt="Do gift cards expire?"),
    Boolean(id="points", prompt="Do loyalty points expire after inactivity?"),
] * 8  # enough to cover n up to 32 by cycling ids with a suffix below
QUESTIONS = []
for i in range(N_QUESTIONS):
    base = ALL_QUESTIONS[i % 4]
    q = type(base)(**{**base.__dict__, "id": f"{base.id}{i}"})
    QUESTIONS.append(q)

SCENARIO = {"name": None}
RECORDS: dict[str, dict[str, list]] = {"two_pass": {}, "interleaved": {}}
COUNTERS: dict[str, int] = {}


def _record(name: str, tensors) -> None:
    scenario = SCENARIO["name"]
    if scenario is None:
        return
    RECORDS[scenario].setdefault(name, []).append(
        [t.detach().to("cpu", torch.float32).clone() if torch.is_tensor(t) else t for t in tensors]
    )


def _wrap(module, name: str) -> None:
    original = module.forward

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        _record(name, tensors)
        return out

    module.forward = wrapped


def install_hooks(engine: Prismyra) -> None:
    text_model = engine.backbone.language_model if hasattr(engine.backbone, "language_model") else engine.backbone
    for i, layer in enumerate(text_model.layers):
        decoder = getattr(engine.config, "text_config", engine.config)
        kind = list(decoder.layer_types)[i]
        if kind == "linear_attention":
            _wrap(layer.linear_attn, f"L{i:02d}.linear_attn")
        else:
            _wrap(layer.self_attn, f"L{i:02d}.self_attn")
        _wrap(layer.mlp, f"L{i:02d}.mlp")


def main() -> None:
    engine = Prismyra(MODEL, require_kernels=True)
    install_hooks(engine)

    SCENARIO["name"] = "two_pass"
    engine.interleaved_fork = False
    res_two_pass = engine.ask(CONTEXT, QUESTIONS)
    SCENARIO["name"] = None

    SCENARIO["name"] = "interleaved"
    engine.interleaved_fork = True
    res_interleaved = engine.ask(CONTEXT, QUESTIONS)
    SCENARIO["name"] = None

    def _flat_probs(result):
        out = []
        for qid in sorted(result.answers):
            out.extend(result.answers[qid].probabilities.values())
        return torch.tensor(out, dtype=torch.float64)

    probs_tp = _flat_probs(res_two_pass)
    probs_il = _flat_probs(res_interleaved)
    probs_equal = torch.equal(probs_tp, probs_il)
    probs_maxdiff = (probs_tp - probs_il).abs().max().item() if not probs_equal else 0.0

    print(f"n_questions={N_QUESTIONS} model={MODEL}")
    print(f"final-answer torch.equal: {probs_equal} (maxdiff={probs_maxdiff:.6e})")

    names = sorted(set(RECORDS["two_pass"]) | set(RECORDS["interleaved"]))
    first_mismatch = None
    for name in names:
        tp_calls = RECORDS["two_pass"].get(name, [])
        il_calls = RECORDS["interleaved"].get(name, [])
        is_mlp = name.endswith(".mlp")
        if is_mlp:
            # two_pass: 2 calls (ctx-rows-only, branch-rows-only). interleaved: 1 call (concatenated) -- split it.
            if len(il_calls) == 1 and len(tp_calls) == 2:
                combined = il_calls[0][0]
                ctx_rows = tp_calls[0][0].reshape(-1, combined.shape[-1]).shape[0]
                branch_rows = tp_calls[1][0].reshape(-1, combined.shape[-1]).shape[0]
                flat = combined.reshape(-1, combined.shape[-1])
                if flat.shape[0] == ctx_rows + branch_rows:
                    il_split = [[flat[:ctx_rows]], [flat[ctx_rows:]]]
                else:
                    print(f"  {name}: SKIP (shape mismatch, ctx={ctx_rows} branch={branch_rows} flat={flat.shape[0]})")
                    continue
            else:
                print(f"  {name}: SKIP (unexpected call counts tp={len(tp_calls)} il={len(il_calls)})")
                continue
        else:
            il_split = il_calls
        pair_count = min(len(tp_calls), len(il_split))
        for call_idx in range(pair_count):
            tp_t = tp_calls[call_idx][0].reshape(-1)
            il_t = il_split[call_idx][0].reshape(-1)
            n = min(tp_t.numel(), il_t.numel())
            if tp_t.numel() != il_t.numel():
                print(f"  {name} call#{call_idx}: SHAPE DIFF tp={tp_calls[call_idx][0].shape} "
                      f"il={il_split[call_idx][0].shape}")
                continue
            eq = torch.equal(tp_t, il_t)
            maxdiff = (tp_t - il_t).abs().max().item() if not eq else 0.0
            label = "ctx" if call_idx == 0 else "branch"
            status = "OK" if eq else "MISMATCH"
            print(f"  {name} call#{call_idx}({label}): {status} maxdiff={maxdiff:.6e}")
            if not eq and first_mismatch is None:
                first_mismatch = (name, call_idx, label, maxdiff)

    print("---")
    if first_mismatch:
        print(f"FIRST MISMATCH: {first_mismatch[0]} call#{first_mismatch[1]} ({first_mismatch[2]}) "
              f"maxdiff={first_mismatch[3]:.6e}")
    else:
        print("No per-layer mismatch found (all hooked ops bit-identical) -- divergence must be in something "
              "not hooked here (embeddings, norms, rope, or the final readout).")


if __name__ == "__main__":
    main()
