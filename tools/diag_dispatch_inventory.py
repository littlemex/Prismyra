"""inv (SYNTHESIS 0 / S2 step 2): enumerate every aten matmul-family call site hit during one real pass, and flag
which ones are row-count dependent -- i.e. the GEMM reduction algorithm the backend picks changes with the pass's
total row count M, which is exactly the class of bug this project calls "the companion effect"
(THROUGHPUT.md's root cause for the MoE router and, separately, the borrowed gated-delta kernel).

Unlike the existing diag_full_inventory_op_divergence.py (named-module forward hooks: misses anything that happens
*inside* a borrowed kernel that is not itself an nn.Module call, which is exactly the gated-delta-rule kernel's own
internal matmul -- the dossier's "no Python hook to fix narrowly" finding), this tool hooks at the torch_dispatch
level, below every Python call, so it sees every aten::mm/addmm/bmm/linear invocation regardless of whether the
calling code is prismyra's own, a borrowed third-party kernel's, or vLLM's.

Method: run the identical `open_batch([target, companion])` pass twice, once with a short companion and once with a
long one (content differs too, but diag_total_length_hypothesis.py already established only total row count M
matters, not companion content) -- same technique diag_full_inventory_op_divergence.py uses, applied one level
lower. For every logged call we record (call-site key, op name, operand shapes). A call site is "M-dependent" if it
appears in both runs with a different M (the dimension that equals the pass's total row count) -- that is a
necessary condition for the row-count-keyed-algorithm bug; it is not sufficient (plenty of ops legitimately take a
different-shaped input and are still bit-reproducible for a fixed input), so this script's output is a *candidate
list* for diag_full_inventory_op_divergence.py-style value diffing, not a final verdict by itself.

Usage: DIAG_MODE=raw|narrow|full python3 diag_dispatch_inventory.py
"""

import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

import tasks  # noqa: E402
from prismyra import Prismyra  # noqa: E402
from prismyra.kernels import qwen3_moe  # noqa: E402

MODEL = os.environ.get("DIAG_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
DIAG_MODE = os.environ.get("DIAG_MODE", "full")

import prismyra.engine as engine_mod  # noqa: E402

if DIAG_MODE == "raw":
    qwen3_moe._ROUTER_LINEAR = None
    engine_mod._enable_batch_invariance = lambda: None
elif DIAG_MODE == "narrow":
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    engine_mod._enable_batch_invariance = lambda: None
elif DIAG_MODE == "full":
    pass
else:
    raise ValueError(f"unknown DIAG_MODE={DIAG_MODE!r}")
print(f"DIAG_MODE={DIAG_MODE}", flush=True)

WATCH = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.linear.default if hasattr(torch.ops.aten, "linear") else None,
}
WATCH.discard(None)

#: file-stems we care about; everything else collapses to "<external>" so the key stays short and stable across runs.
INTERESTING_STEMS = ("prismyra", "flash_linear_attention", "fused_moe", "batch_invariant", "qwen3_5_moe", "qwen3_moe")


def _site_key() -> str:
    for frame in reversed(traceback.extract_stack()[:-2]):  # skip this frame + __torch_dispatch__ itself
        if any(stem in frame.filename for stem in INTERESTING_STEMS):
            return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
    # Nothing interesting on the stack (e.g. called from torch internals only) -- use the closest frame anyway.
    frame = traceback.extract_stack()[-3]
    return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"


class Inventory(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls: dict[str, list[tuple]] = defaultdict(list)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func in WATCH:
            shapes = tuple(a.shape for a in args if isinstance(a, torch.Tensor))
            self.calls[_site_key()].append(shapes)
        return out


def run_pass(engine, target, companion):
    inv = Inventory()
    with inv:
        with engine.open_batch([target.context, companion.context]) as batch:
            batch.ask([target.questions, companion.questions])
    return inv.calls


items = tasks.load("race", 80, split="validation", seed=0)
tok_len = lambda it, eng: len(eng.tokenizer(it.context, add_special_tokens=False)["input_ids"])  # noqa: E731

engine = Prismyra(MODEL, paged=True, graphs=False)
by_len = sorted(items, key=lambda it: tok_len(it, engine))
target = by_len[len(by_len) // 2]
short, long_ = by_len[0], by_len[-1]
if short is target:
    short = by_len[1]
if long_ is target:
    long_ = by_len[-2]
print(f"target={tok_len(target, engine)}tok short_companion={tok_len(short, engine)}tok long_companion={tok_len(long_, engine)}tok", flush=True)

calls_short = run_pass(engine, target, short)
calls_long = run_pass(engine, target, long_)

all_sites = sorted(set(calls_short) | set(calls_long))
print(f"\n{len(all_sites)} distinct call sites hit (op in {{mm,addmm,bmm,linear}}):\n")
candidates = []
for site in all_sites:
    s_shapes = calls_short.get(site, [])
    l_shapes = calls_long.get(site, [])
    s_set, l_set = set(s_shapes), set(l_shapes)
    m_dependent = bool(s_shapes) and bool(l_shapes) and s_set != l_set
    flag = "  <-- CANDIDATE (shape changed with companion length)" if m_dependent else ""
    print(f"  {site:<55} short_calls={len(s_shapes):<4} long_calls={len(l_shapes):<4} "
          f"short_shapes={sorted(s_set)[:2]} long_shapes={sorted(l_set)[:2]}{flag}")
    if m_dependent:
        candidates.append(site)

print(f"\n{len(candidates)} candidate row-count-dependent call sites: {candidates}")
