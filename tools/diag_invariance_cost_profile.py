"""inv3 S2 step 1: where does the +10% (solo) / -21% (open_batch) cost documented in docs/PERFORMANCE.md actually
sit? `engine._enable_batch_invariance()` registers vLLM's fixed-tile Triton kernel on `aten::mm`/`addmm`/`matmul`/
`linear` for the *whole process* -- every plain bf16 `nn.Linear`/`F.linear`/`@` call pays for it, not only the ones
that need it. This profiles a solo `ask()` and a 32-document `open_batch` pass twice -- once under today's shipped
registration ("full"), once with the global dispatcher override forced off but the two already-narrow fixes
(`_ROUTER_LINEAR` at `_route`'s one call site, `VLLM_BATCH_INVARIANT=1` for `fused_moe`'s own tile-size guard) left
on ("narrow") -- and reports CUDA self-time by op, so the next step (an exhaustive TorchDispatchMode enumeration,
`diag_dispatch_inventory.py`) has a cost ranking to prioritise, not just a candidate list.

Usage: PRISMYRA_MODEL=<repo> python3 diag_invariance_cost_profile.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402
import prismyra.engine as engine_mod  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
N_DOCS = 32

# inv4, round 6: `tasks.load("race", ...)` needs the `datasets` package, which cannot be installed alongside this
# image's pinned `transformers`/`tokenizers` (huggingface-hub<2.0) without breaking model loading outright
# (confirmed on both the Spain and Tokyo pods this round). This profiles timing only, not correctness, so 32
# synthetic documents of realistic length stand in for real RACE passages.
_BASE_DOC = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
    "seller when the item is faulty and by the buyer otherwise. Gift cards never expire and are not redeemable "
    "for cash. A lost gift card is replaced only with the original purchase receipt and a matching photo ID."
)
DOCS = [f"{_BASE_DOC} (document {i} of this profiling run, otherwise identical to its neighbours.)"
        for i in range(N_DOCS)]
QUESTION = Boolean(id="q", prompt="Is a refund limited to unopened items?")


def one_question(i):
    return [dataclasses.replace(QUESTION, id=f"q{i}")]


def run_solo(engine):
    engine.ask(DOCS[0], one_question(0))


def run_open_batch32(engine):
    with engine.open_batch(DOCS) as batch:
        batch.ask([one_question(i) for i in range(N_DOCS)])


def force_dispatcher_off():
    """Diagnostic only -- drops the module-level refcount straight to zero regardless of who claimed it, so the
    *global* dispatcher override comes off while the two narrow fixes (independent of it) stay on. Not something
    production code or a test should do; `engine.paged`'s own refcounted setter is the real API."""
    while engine_mod._BATCH_INVARIANT_REFCOUNT > 0:
        engine_mod._disable_batch_invariance()
    assert engine_mod._BATCH_INVARIANT_DISPATCH_LIB is None


def profile_scenario(engine, fn, label):
    fn(engine)  # warm up: autotune caches, CUDA graph pools, first-call allocations
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        fn(engine)
        torch.cuda.synchronize()
    attr = "self_device_time_total" if hasattr(prof.key_averages()[0] if prof.key_averages() else None, "self_device_time_total") else "self_cuda_time_total"
    print(f"\n=== {label} ===")
    sort_key = "self_cuda_time_total" if attr == "self_cuda_time_total" else "self_device_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
    print(table)
    total_cuda_us = sum(getattr(e, attr) for e in prof.key_averages())
    print(f"total self CUDA time: {total_cuda_us / 1000:.3f} ms")
    return total_cuda_us


engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)
print(f"dispatcher registered (refcount)={engine_mod._BATCH_INVARIANT_REFCOUNT}", flush=True)

full_solo = profile_scenario(engine, run_solo, "FULL (today's prod) -- solo ask()")
full_batch = profile_scenario(engine, run_open_batch32, "FULL (today's prod) -- open_batch(32 docs)")

force_dispatcher_off()
print(f"\nafter forcing off: dispatcher registered (refcount)={engine_mod._BATCH_INVARIANT_REFCOUNT}, "
      f"lib={engine_mod._BATCH_INVARIANT_DISPATCH_LIB}", flush=True)

narrow_solo = profile_scenario(engine, run_solo, "NARROW (router+moe-tiling only) -- solo ask()")
narrow_batch = profile_scenario(engine, run_open_batch32, "NARROW (router+moe-tiling only) -- open_batch(32 docs)")

print("\n=== summary (self CUDA time, ms) ===")
print(f"solo:  full={full_solo/1000:.3f}  narrow={narrow_solo/1000:.3f}  "
      f"delta={(full_solo-narrow_solo)/1000:.3f} ({(full_solo-narrow_solo)/narrow_solo*100:.1f}%)")
print(f"batch: full={full_batch/1000:.3f}  narrow={narrow_batch/1000:.3f}  "
      f"delta={(full_batch-narrow_batch)/1000:.3f} ({(full_batch-narrow_batch)/narrow_batch*100:.1f}%)")
