"""inv: what is `self.gate` inside FusedExpertsFp4, and does it reach the row-count-invariant router kernel?

`FusedExperts._route` (the FP8 path) wraps the router call as `linear(x, self.gate.weight)` with
`linear = _ROUTER_LINEAR or F.linear` -- a direct, row-count-invariant call. `FusedExpertsFp4.forward` (the
NVFP4-experts path) instead does `logits, _, _ = self.gate(x)`, calling the module's own `forward` whole. If that
module's forward is a plain `F.linear` under the hood, it is *not* going through `_ROUTER_LINEAR` at all -- a
candidate root cause for the sm_120 residual this session is trying to close.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PRISMYRA_EXPERTS", "nvfp4")

import torch  # noqa: E402
from prismyra import Prismyra  # noqa: E402
from prismyra.kernels import nvfp4 as nvfp4_mod  # noqa: E402

MODEL = os.environ["PRISMYRA_MODEL"]
engine = Prismyra(MODEL, paged=True, graphs=False, group=32)

text_model = getattr(engine.backbone, "language_model", engine.backbone)
found = 0
for name, m in text_model.named_modules():
    if type(m).__name__ == "FusedExpertsFp4":
        gate = m.gate
        print(f"{name}: gate type = {type(gate).__module__}.{type(gate).__name__}")
        print(f"  gate attrs: {[a for a in dir(gate) if not a.startswith('_')][:20]}")
        print(f"  has .weight: {hasattr(gate, 'weight')}")
        import inspect
        try:
            src = inspect.getsource(type(gate).forward)
            print(f"  forward source:\n{src}")
        except Exception as e:
            print(f"  (could not get source: {e})")
        found += 1
        if found >= 1:
            break

print(f"\nFusedExpertsFp4 modules found: {found}")
