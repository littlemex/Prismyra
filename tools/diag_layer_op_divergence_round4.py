"""inv2 round4: find the first operation that diverges between "target answered alone" and "target answered
with a companion", on the REAL model and the REAL test fixtures (test_gpu.py's
test_open_batch_matches_ask_bit_for_bit_whatever_the_companions_total_length), after round 3's per-document
padding fix. The isolated synthetic probes (probe_attn_rowcount.py, probe_gdn_rowcount.py,
probe_fused_moe_rowcount.py, probe_dense_fp8_rowcount.py) each showed torch.equal across row counts in
isolation -- so whatever is left either needs the real weights/real routing to show up, or needs several ops'
small per-call differences to compound into the measured 0.008346. This hooks every interesting submodule by
identity (forward hook, so it sees the module's real output regardless of what raw kernel it calls inside) and
every one of the four raw functions the isolated probes already checked (belt and suspenders -- a real pass
may hit a different code branch, e.g. the "key is None" paged branch vs. the non-paged one, than a synthetic
call does), then diffs call-by-call between the "alone" and "with companion" runs, slicing each tensor down to
the target document's own rows/tokens so a real difference in the companion's own content never shows up as a
false positive.

Run on the device under PRISMYRA_TEST_MODEL (defaults to the FP8/L40S checkpoint, matching test_gpu.py).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Choice, Prismyra  # noqa: E402
from prismyra import varlen  # noqa: E402

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

CONTEXT = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the seller "
    "when the item is faulty and by the buyer otherwise."
)
SECOND_CONTEXT = (
    "Gift cards never expire and are not redeemable for cash. A lost gift card is replaced only with the original "
    "purchase receipt and a matching photo ID."
)

ASKED = [
    Boolean(id="faulty", prompt="Does the seller pay return shipping on a faulty item?"),
    Choice(
        id="opened",
        prompt="What happens to an opened item?\nA. Refunded\nB. Exchanged\nC. Kept\nD. Discarded",
        choices=["A", "B", "C", "D"],
    ),
]
ABOUT_COMPANION = [Boolean(id="replaced", prompt="Is a lost gift card replaced on proof of purchase?")]
TARGET_ROWS = len(ASKED)  # 2 -- already a power-of-two bucket, so the target needs no padding of its own

# -------------------------------------------------------------------------------------------- recording machinery
SCENARIO = {"name": None}
COUNTERS: dict[str, int] = {}
RECORDS: dict[str, dict[str, list]] = {}
FIRST_MISMATCH = None


def _slice_for_tensor(t: torch.Tensor) -> torch.Tensor | None:
    """Cut a tensor down to the target document's own rows/tokens, token-major if a batched read is open,
    row-major (the pass's batch dimension) otherwise -- matching exactly which of the two code paths a real
    forward pass takes, decided by `varlen.current()` the same way the kernels themselves decide it."""
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
        return None  # this tensor's shape has nothing to do with the flat token run -- not comparable here
    # Branch pass (or any call outside a batched read): the pass's own batch/row dimension is dim 0 for every
    # kernel this file hooks (conv/attention/GDN/MoE/RMSNorm all keep the row axis first).
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
    RECORDS[scenario].setdefault(name, []).append(
        [s.detach().to("cpu", torch.float32).clone() for s in sliced]
    )


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

    # Raw functions the real code reaches outside any single nn.Module's own forward (belt and suspenders: the
    # isolated probes already checked these, this just confirms the real pass hits the same code with the same
    # result, under the real weights and the real routing).
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
        want = engine.ask(CONTEXT, ASKED)
    else:
        companion = SECOND_CONTEXT if name == "short" else SECOND_CONTEXT * 6
        with engine.open_batch([CONTEXT, companion]) as batch:
            want = batch.ask([ASKED, ABOUT_COMPANION])[0]
    SCENARIO["name"] = None
    return want


def diff(a_name: str, b_name: str) -> None:
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
        print(f"  no mismatch found in any hooked op/module between {a_name} and {b_name} (bit-identical through everything this hooks)")
    else:
        print(f"\n  FIRST DIVERGENT OP: {first[0]} call#{first[1]} tensor#{first[2]} max|diff|={first[3]:.3e}")


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False)
    install_hooks(engine)

    want = run_scenario(engine, "alone")
    short = run_scenario(engine, "short")

    for q in ASKED:
        for option, p in want[q.id].probabilities.items():
            got = short[q.id].probabilities[option]
            if got != p:
                print(f"probability moved: {q.id} {option}: alone={p!r} short={got!r} diff={abs(got - p):.6e}")

    diff("alone", "short")


if __name__ == "__main__":
    main()
