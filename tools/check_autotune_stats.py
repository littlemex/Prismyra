"""inv2 round4 (coordinator's acc/A5 follow-up): what `engine.stats()["autotune"]` actually reports pinned vs
falling back to, before and after one real forward pass -- used to check whether the GDN triangular-solve/merge
kernel acc's A5 flagged (`chunk_gated_delta_rule_fwd_kkt_solve_kernel`, from the standalone `fla` PyPI package)
has an equivalent gap in the kernel prismyra actually serves with (`vllm.third_party.flash_linear_attention`,
confirmed via `_borrowed_delta`'s own import and this file's git history: always this path, never bare `fla`).
See RUN-inv.md's inv2 section 9 for the full finding: the real, BT=64 kernel (`merge_16x16_to_64x64_inverse_
kernel`) is already in both `pinned/sm_89.json` and `pinned/sm_120.json`, and the two sibling kernels that are
genuinely unpinned (`solve_tril_16x16_kernel`, `merge_16x16_to_32x32_inverse_kernel`) are dead code for this
model (BT is always 64, confirmed by reading `solve_tril()`'s own dispatch in vLLM's vendored copy).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prismyra import Boolean, Prismyra  # noqa: E402

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
engine = Prismyra(MODEL, paged=True)
print(json.dumps(engine.stats()["autotune"], indent=2))

# force a real forward pass (reads context + answers 1 question) and check again, in case some kernels
# only materialise an Autotuner object on first import triggered inside the forward itself.
r = engine.ask("Returns are accepted within thirty days of delivery.", [Boolean(id="q", prompt="Within how many days?")])
print("---after a real forward pass---")
print(json.dumps(engine.stats()["autotune"], indent=2))
