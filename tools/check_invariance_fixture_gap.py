"""inv2 round4: does test_gpu.py's `engine_paged` fixture (construct `Prismyra(MODEL)` with paged's default
False, then flip `.paged = True` as a plain attribute afterward) actually leave the dispatcher-level batch
invariance override (`engine.py::_enable_batch_invariance`, which only runs inside `__init__` when the
*constructor's* `paged` argument is True) registered or not? `_enable_batch_invariance()`'s dispatcher
registration is a torch.library.Library global, so it is a process-wide side effect, not an instance attribute
-- this checks the module-level singleton `prismyra.engine._BATCH_INVARIANT_DISPATCH_LIB` directly, exactly
reproducing the fixture's own two steps in order.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prismyra import Prismyra  # noqa: E402
import prismyra.engine as engine_mod  # noqa: E402

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

print(f"before construction: _BATCH_INVARIANT_DISPATCH_LIB={engine_mod._BATCH_INVARIANT_DISPATCH_LIB}")
engine = Prismyra(MODEL)  # exactly test_gpu.py's `engine()` fixture -- no paged=True
print(f"after Prismyra(MODEL) [paged defaults False]: _BATCH_INVARIANT_DISPATCH_LIB={engine_mod._BATCH_INVARIANT_DISPATCH_LIB}")
print(f"engine.paged={engine.paged}")

engine.paged = True  # exactly test_gpu.py's `engine_paged` fixture
print(f"after engine.paged = True: _BATCH_INVARIANT_DISPATCH_LIB={engine_mod._BATCH_INVARIANT_DISPATCH_LIB}")
print(f"engine.paged={engine.paged}")

print()
if engine_mod._BATCH_INVARIANT_DISPATCH_LIB is None:
    print("CONFIRMED: engine_paged's flip-after-construct never re-runs _enable_batch_invariance(); the "
          "dispatcher override (aten::mm/addmm/matmul/linear fixed-tile Triton) is NOT registered for any test "
          "using only this fixture pattern in isolation.")
else:
    print("the dispatcher override IS registered -- the gap theory is wrong, something else must explain the "
          "residual difference.")
