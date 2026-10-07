"""inv3 task 1: `audit_sm120.py`'s own matrix (companions in {1,2,3,8}, questions-per-document in
{1,2,3,31,32,33}) still shows 72/4392 non-exact after round 4's `paged`-property fix (commit 51ab8ff) --
all 72 at question-count=1, none at group_size=1 (a lone document inside `open_batch`, no real companion). Round
4's own verification (`measure_residual_after_fix.py`) used a 2-question target fixture and never re-checked
qn=1, so this is this round's first job: find the first operation that diverges between "target (1 question)
answered alone" and "target (1 question) answered alongside N-1 companions who also ask 1 question each",
reusing round 4's forward-hook methodology (`diag_layer_op_divergence_round4.py`) at the one row count audit_sm120
flags.

Run on the device under PRISMYRA_TEST_MODEL (fp8-36l by default) or PRISMYRA_MODEL for the sm_120/nvfp4 path.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean  # noqa: E402
from prismyra import Prismyra  # noqa: E402
from prismyra import varlen  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL") or os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
GROUP_SIZE = int(os.environ.get("DIAG_GROUP_SIZE", "8"))  # one of audit_sm120's own non-exact buckets: 2, 3, 8

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

TARGET_ROWS = 1

SCENARIO = {"name": None}
COUNTERS: dict[str, int] = {}
RECORDS: dict[str, dict[str, list]] = {}


def _slice_for_tensor(t: torch.Tensor) -> torch.Tensor | None:
    if not torch.is_tensor(t) or t.dim() == 0:
        return None
    boundaries = varlen.current()
    if boundaries is not None:
        n = boundaries.lengths[0]
        for dim, size in enumerate(t.shape):
            if size == boundaries.tokens:
                idx = [slice(None)] * t.dim()
                idx[dim] = slice(0, n)
                return t[tuple(idx)]
        return None
    if t.shape[0] < TARGET_ROWS:
        return None
    return t[:TARGET_ROWS]


def _record(name: str, output) -> None:
    scenario = SCENARIO["name"]
    if scenario is None:
        return
    tensors = output if isinstance(output, (tuple, list)) else (output,)
    sliced = [_slice_for_tensor(t) for t in tensors]
    sliced = [s for s in sliced if s is not None]
    if not sliced:
        return
    idx = COUNTERS.get(name, 0)
    COUNTERS[name] = idx + 1
    RECORDS[scenario].setdefault(name, []).append([s.detach().to("cpu", torch.float32).clone() for s in sliced])


def _wrap_module_forward(module, name: str) -> None:
    original = module.forward

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        _record(name, out)
        return out

    module.forward = wrapped


def _wrap_function(obj, attr: str, name: str):
    original = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        _record(name, out)
        return out

    setattr(obj, attr, wrapped)
    return original


def install_hooks(engine: Prismyra) -> None:
    hook_types = (
        "FastRMSNorm", "FusedGatedRMSNorm", "Fp8Linear", "FusedExperts", "FusedExpertsFp4",
        "FlashAttention", "Qwen3_5MoeGatedDeltaNet", "Qwen3_5MoeRMSNorm",
    )
    counts: dict[str, int] = {}
    for mod_name, module in engine.backbone.named_modules():
        type_name = type(module).__name__
        if type_name not in hook_types:
            continue
        counts[type_name] = counts.get(type_name, 0) + 1
        _wrap_module_forward(module, f"module:{type_name}:{mod_name}")
    print(f"hooked module instances by type: {counts}", flush=True)

    import vllm.vllm_flash_attn as _fa
    _wrap_function(_fa, "flash_attn_varlen_func", "fn:flash_attn_varlen_func")

    import vllm.third_party.flash_linear_attention.ops.chunk as _chunk
    _wrap_function(_chunk, "chunk_gated_delta_rule", "fn:chunk_gated_delta_rule")

    from prismyra.kernels import conv as _conv
    _wrap_function(_conv, "causal_depthwise_conv1d", "fn:causal_depthwise_conv1d")

    import vllm.model_executor.layers.fused_moe as _fm
    _wrap_function(_fm, "fused_experts", "fn:fused_experts")

    import vllm.model_executor.layers.quantization.utils.fp8_utils as _fp8u
    _wrap_function(_fp8u, "w8a8_triton_block_scaled_mm", "fn:w8a8_triton_block_scaled_mm")


def run_scenario(engine: Prismyra, name: str) -> dict:
    SCENARIO["name"] = name
    COUNTERS.clear()
    RECORDS[name] = {}
    if name == "alone":
        want = engine.ask(CONTEXTS[0], [ONE_QUESTION])
    else:
        docs = CONTEXTS[:GROUP_SIZE]
        with engine.open_batch(docs) as batch:
            per_doc = [[ONE_QUESTION]] + [[COMPANION_QUESTION] for _ in docs[1:]]
            results = batch.ask(per_doc)
            want = results[0]
    SCENARIO["name"] = None
    return want


def diff(a_name: str, b_name: str):
    keys = sorted(set(RECORDS[a_name]) | set(RECORDS[b_name]))
    print(f"\n=== {a_name} vs {b_name}: {len(keys)} distinct op/module keys recorded ===")
    first = None
    for key in keys:
        a_calls = RECORDS[a_name].get(key, [])
        b_calls = RECORDS[b_name].get(key, [])
        if len(a_calls) != len(b_calls):
            print(f"  {key}: call count differs ({len(a_calls)} vs {len(b_calls)}) -- skipping (not aligned)")
            continue
        for i, (a_tensors, b_tensors) in enumerate(zip(a_calls, b_calls)):
            if len(a_tensors) != len(b_tensors):
                print(f"  {key}[{i}]: tensor count differs -- skipping")
                continue
            for j, (ta, tb) in enumerate(zip(a_tensors, b_tensors)):
                if ta.shape != tb.shape:
                    print(f"  {key}[{i}][{j}]: shape differs {tuple(ta.shape)} vs {tuple(tb.shape)} -- skipping")
                    continue
                eq = torch.equal(ta, tb)
                if not eq:
                    diff_val = (ta - tb).abs().max().item()
                    marker = ""
                    if first is None:
                        first = (key, i, j, diff_val)
                        marker = "  <-- FIRST DIVERGENCE"
                    print(f"  {key}[{i}][{j}]: MISMATCH max|diff|={diff_val:.3e}{marker}")
    if first is None:
        print(f"  no mismatch found in any hooked op/module between {a_name} and {b_name} "
              f"(bit-identical through everything this hooks)")
    else:
        print(f"\n  FIRST DIVERGENT OP: {first[0]} call#{first[1]} tensor#{first[2]} max|diff|={first[3]:.3e}")
    return first


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    install_hooks(engine)

    want = run_scenario(engine, "alone")
    grouped = run_scenario(engine, f"group{GROUP_SIZE}")

    for option, p in want["target"].probabilities.items():
        got = grouped["target"].probabilities[option]
        if got != p:
            print(f"probability moved: target {option}: alone={p!r} group{GROUP_SIZE}={got!r} diff={abs(got - p):.6e}")
        else:
            print(f"probability matched: target {option}: {p!r}")

    diff("alone", f"group{GROUP_SIZE}")


if __name__ == "__main__":
    main()
