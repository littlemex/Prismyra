"""Answers must not depend on the process. Each cold start below runs in its own process with an empty Triton cache,
so every autotuned kernel would choose its configuration afresh -- and a timing-picked configuration is what used to
change probabilities from one start to the next (see `prismyra.kernels.autotune`). Marked `gpu`; in its own file
because each start loads the weights and they must not be held by another test at the same time.

    pytest -m gpu tests/test_gpu_cold_start.py

The test above only ever loaded `MODEL` -- `PRISMYRA_TEST_MODEL`, defaulting to the FP8 checkpoint -- so a run on
the RTX PRO 4500 pod that never set that override exercised the FP8 code path there too, never the NVFP4 one, and
its `raced` check only enumerates `triton.runtime.autotuner.Autotuner` instances (`kernels.autotune.autotuners()`):
NVFP4's own GEMM tactic is chosen by a *different* autotuner, FlashInfer's `flashinfer.autotuner.AutoTuner`, which
that enumeration cannot see at all. Both gaps let a real defect through two verification rounds (`RUN-ship2.md`,
`RUN-recon.md`) before a third round (`RUN-tac.md`) measured it directly: the same release, as two separate
processes with no `PRISMYRA_NVFP4_TACTICS`, answered the same request bit-for-bit differently in 81 of 81 entries.
`test_cold_starts_pin_the_nvfp4_tactic_too` below closes both gaps -- it loads the real NVFP4 checkpoint, and it
asserts on `engine.stats()["nvfp4_tactics"]` (the mechanism: did this process load a pinned tactic, or time its
own) rather than only on whether two or three samples happened to disagree. That distinction matters: pinning the
tactic was measured to make the *selection* deterministic every time, but a handful of cold starts with it pinned
still disagreed once in three -- not from the tactic, but from a second, unrelated bug (an uninitialised scratch
buffer `FusedExpertsFp4._workspace()` reuses across calls) that happened to answer identically on this specific
request often enough that a small sample could have missed it too. Comparing answers alone is suggestive;
asserting the mechanism plus comparing answers is a test that cannot pass for the wrong reason.

    PRISMYRA_NVFP4_TEST_MODEL=/path/to/nvfp4-36l PRISMYRA_EXPERTS=nvfp4 PRISMYRA_NVFP4_EXPERTS=... \\
        PRISMYRA_NVFP4_CALIB=... pytest -m gpu tests/test_gpu_cold_start.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from gpu_room import no_room_reason

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
    # Each start is a separate process and needs the card's memory. In a session that ran test_gpu.py first, the
    # engines there are gone but this process's allocator may still hold their blocks; hand them back to the device.
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    reason = no_room_reason(MODEL, __file__)
    if reason:
        pytest.skip(reason)
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


NVFP4_SCRIPT = r"""
import json, sys
from prismyra import Boolean, Choice, Prismyra
engine = Prismyra(sys.argv[1], require_kernels=False)
context = ("Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
           "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
           "seller when the item is faulty and by the buyer otherwise. ") * 12
questions = [Boolean(id="refund", prompt="Is an opened item refunded?"),
             Choice(id="pays", prompt="Who pays return shipping for a faulty item?", choices=["seller", "buyer"])]
answers = {q.id: engine.ask(context, [q])[q.id].probabilities for q in questions}
answers.update({"fork_" + k: v.probabilities for k, v in engine.ask(context, questions).items()})
print(json.dumps({"answers": answers, "nvfp4_tactics": engine.stats()["nvfp4_tactics"]}))
"""


def test_cold_starts_pin_the_nvfp4_tactic_too(tmp_path):
    """The NVFP4 counterpart of the test above: a different model, a different autotuner the first test's `raced`
    check cannot see (module docstring), and a second bug the tactic pin alone did not cover."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = os.environ.get("PRISMYRA_NVFP4_TEST_MODEL")
    if not model or os.environ.get("PRISMYRA_EXPERTS") != "nvfp4":
        pytest.skip("set PRISMYRA_NVFP4_TEST_MODEL, PRISMYRA_EXPERTS=nvfp4, PRISMYRA_NVFP4_EXPERTS, PRISMYRA_NVFP4_CALIB")
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    reason = no_room_reason(model, __file__)
    if reason:
        pytest.skip(reason)
    runs = []
    for start in range(STARTS):
        env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"triton-nvfp4-{start}"))
        done = subprocess.run(
            [sys.executable, "-c", NVFP4_SCRIPT, model], env=env, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr[-2000:]
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    for run in runs:
        # The mechanism, not just the outcome: this process loaded a tactic someone already measured (the
        # bundled table or PRISMYRA_NVFP4_TACTICS), rather than timing the candidates itself. A process that
        # profiled its own tactic can still happen to agree with another one by chance (see the module
        # docstring) -- this assertion is what makes that impossible to confuse with a real pin.
        assert run["nvfp4_tactics"]["pinned"] is True, run["nvfp4_tactics"]
    first = runs[0]["answers"]
    for run in runs[1:]:
        assert run["answers"] == first  # exact equality of every float, not a tolerance
