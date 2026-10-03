"""Answers must not depend on the process. Each cold start below runs in its own process with an empty Triton cache,
so every autotuned kernel would choose its configuration afresh -- and a timing-picked configuration is what used to
change probabilities from one start to the next (see `prismyra.kernels.autotune`). Marked `gpu`; in its own file
because each start loads the weights and they must not be held by another test at the same time.

    pytest -m gpu tests/test_gpu_cold_start.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.gpu

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
STARTS = int(os.environ.get("PRISMYRA_COLD_STARTS", "3"))

SCRIPT = r"""
import json, sys
from prismyra import Boolean, Choice, Prismyra
from prismyra.kernels.autotune import autotuners, kernel_name
engine = Prismyra(sys.argv[1])
context = ("Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
           "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
           "seller when the item is faulty and by the buyer otherwise. ") * 12
questions = [Boolean(id="refund", prompt="Is an opened item refunded?"),
             Choice(id="pays", prompt="Who pays return shipping for a faulty item?", choices=["seller", "buyer"])]
answers = {q.id: engine.ask(context, [q])[q.id].probabilities for q in questions}
answers.update({"fork_" + k: v.probabilities for k, v in engine.ask(context, questions).items()})
raced = sorted(kernel_name(t) for t in autotuners() if len(t.configs) > 1 and getattr(t, "cache", None))
print(json.dumps({"answers": answers, "raced": raced, "autotune": engine.stats()["autotune"]}))
"""


def test_cold_starts_with_an_empty_triton_cache_answer_bit_identically(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    runs = []
    for start in range(STARTS):
        env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"triton-{start}"))
        done = subprocess.run(
            [sys.executable, "-c", SCRIPT, MODEL], env=env, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr[-2000:]
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    for run in runs:
        # Every autotuner that ran had been held to one configuration: none timed its candidates.
        assert run["raced"] == [], run["raced"]
        assert run["autotune"]["pinned"] or run["autotune"]["fallback"]
    first = runs[0]["answers"]
    for run in runs[1:]:
        assert run["answers"] == first  # exact equality of every float, not a tolerance
