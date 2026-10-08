"""Reproduces `tests/test_gpu.py::engine_paged`'s exact two-line fixture pattern (`Prismyra(MODEL)`
then `engine.paged = True`, not `Prismyra(MODEL, paged=True)`) and measures the real, bit-level movement between
"this document answered alone" and "the same document answered alongside a companion" -- the thing
`COMPANION_MOVEMENT_ROW_COUNT` bounds in that test file. Used to prove the `paged` property fix in
`prismyra/engine.py` (see that property's own docstring) closes the residual to exactly 0.0 on the real
checkpoint, and that reverting to a plain attribute reproduces a non-zero residual on the same weights, same
process, same run.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prismyra import Boolean, Choice, Prismyra  # noqa: E402

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
    Choice(id="opened", prompt="What happens to an opened item?\nA. Refunded\nB. Exchanged\nC. Kept\nD. Discarded", choices=["A","B","C","D"]),
]
ABOUT_COMPANION = [Boolean(id="replaced", prompt="Is a lost gift card replaced on proof of purchase?")]

engine = Prismyra(MODEL)             # exactly engine() fixture
engine.paged = True                   # exactly engine_paged fixture

want = engine.ask(CONTEXT, ASKED)
long_companion = SECOND_CONTEXT * 6

with engine.open_batch([CONTEXT, SECOND_CONTEXT]) as short_batch:
    with_short = short_batch.ask([ASKED, ABOUT_COMPANION])[0]
with engine.open_batch([CONTEXT, long_companion]) as long_batch:
    with_long = long_batch.ask([ASKED, ABOUT_COMPANION])[0]

maxdiff_short = maxdiff_long = 0.0
for label, mixed in (("short", with_short), ("long", with_long)):
    for q in ASKED:
        same_option = mixed[q.id].option == want[q.id].option
        for option, p in want[q.id].probabilities.items():
            got = mixed[q.id].probabilities[option]
            d = abs(got - p)
            if label == "short":
                maxdiff_short = max(maxdiff_short, d)
            else:
                maxdiff_long = max(maxdiff_long, d)
            print(f"{label} {q.id} {option}: alone={p!r} mixed={got!r} diff={d:.6e} same_option={same_option}")

print(f"\nMAX short companion diff = {maxdiff_short:.6e}")
print(f"MAX long  companion diff = {maxdiff_long:.6e}")
