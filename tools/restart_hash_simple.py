"""inv2 round4: a dataset-free version of `restart_hash_check.py` (that one needs the `datasets` package and the
`race` task, not installed in every pod) -- same idea, hard-coded fixtures instead. Run three times, as three
independent fresh processes, and diff the printed SHA256: used to confirm cross-restart determinism after the
`paged` property fix (and, separately, after checking the GDN triangular-solve/merge autotune pin coverage --
see RUN-inv.md's inv2 section 9 for what each check was for).
"""

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prismyra import Boolean, Choice, Prismyra  # noqa: E402

MODEL = os.environ["PRISMYRA_MODEL"]
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

engine = Prismyra(MODEL, paged=True, graphs=False)
with engine.open_batch([CONTEXT, SECOND_CONTEXT]) as batch:
    results = batch.ask([ASKED, ABOUT_COMPANION])[0]

parts = []
for q in ASKED:
    probs = results[q.id].probabilities
    for opt in sorted(probs):
        parts.append(f"{probs[opt]!r}")
blob = "|".join(parts)
digest = hashlib.sha256(blob.encode()).hexdigest()
print(f"PROBS_BLOB={blob}")
print(f"SHA256={digest}")
