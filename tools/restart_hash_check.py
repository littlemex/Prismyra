"""inv (coordinator round 2, point 5): process-restart x3 determinism check for one representative config.

Runs ONE config (open_batch, 2 documents, 2 questions each) in a fresh process each invocation, hashes the full
probability tensor, and prints the hash so three separate process launches can be diffed byte-for-byte. This is
what "restart x3" means operationally: there is no cross-process state (CUDA context, allocator arena, autotuner
cache) that could coincidentally make two calls inside the *same* process agree while a fresh process would not --
each restart reloads the model, re-runs autotune (fresh FlashInfer tactic search), and re-registers the dispatcher.
"""
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import tasks  # noqa: E402
from prismyra import Prismyra  # noqa: E402

MODEL = os.environ["PRISMYRA_MODEL"]
os.environ.setdefault("PRISMYRA_EXPERTS", "nvfp4")

items = tasks.load("race", 4, split="validation", seed=3)
engine = Prismyra(MODEL, paged=True, graphs=False, group=32)


def questions_at(item, n):
    import dataclasses

    base = item.questions
    return [dataclasses.replace(base[i % len(base)], id=f"{base[i % len(base)].id}_{i}") for i in range(n)]


with engine.open_batch([items[0].context, items[1].context]) as batch:
    results = batch.ask([questions_at(items[0], 2), questions_at(items[1], 2)])

parts = []
for doc_idx, item in enumerate(items[:2]):
    for q in questions_at(item, 2):
        probs = results[doc_idx][q.id].probabilities
        for opt in sorted(probs):
            parts.append(f"{probs[opt]!r}")
blob = "|".join(parts)
digest = hashlib.sha256(blob.encode()).hexdigest()
print(f"PROBS_BLOB={blob}")
print(f"SHA256={digest}")
