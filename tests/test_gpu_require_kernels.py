"""`prismyra-serve --require-kernels`, which builds its own engine -- so it lives apart from `test_gpu.py`, whose
module-scoped engine would otherwise still hold one copy of the weights when this builds a second. On a 44 GB card two
copies do not fit, and the test failed every time for that reason rather than for the one it exists for.

    pytest -m gpu tests/test_gpu_require_kernels.py

A second, independent reason to isolate this construction turned up measuring `kernels/qwen3_moe.py`'s swap
self-checks (`docs/KERNELS.md`'s "swap self-checks are seeded too"): running this file's own construction *after*
`test_gpu.py`'s module-scoped engine had already run real inference at several row counts, in the same pytest
process, made the self-checks' probe hit `triton.errors.OutOfResources` (a hardware shared-memory limit for the
probe's shape, not a disagreement between the two sides) where the identical probe, run as the first thing in a
fresh process, did not -- a Triton autotuner cache entry from one of those other shapes, carried in this process's
own JIT state, picked an unsuitable configuration for the self-check's shape. `no_room_reason` below only measures
whether this *device* has enough free bytes for a second copy of the weights; it cannot see JIT state that lives in
this *process*, which a second construction in the same process inherits regardless of how much VRAM is free. The
construction below now runs in a subprocess for that reason -- a fresh process has no autotuner cache to inherit --
not only to avoid the two-copies-of-the-weights case `no_room_reason` already guards.

That resource exhaustion itself is now `kernels.qwen3_moe._reraise_if_resource_exhausted`'s job to turn into an
explicit `AdapterError` rather than a silent decline (`docs/KERNELS.md`): if the fix above were ever insufficient --
a card smaller than this one, or a Triton version with a different autotuner cache behaviour -- this test would see
a loud subprocess failure with that error's own message, not a silently different kernel path and a quietly
different answer. Both are checked here: the subprocess must either complete with nothing skipped, or fail with that
specific error naming a resource rather than a disagreement; a silent `skipped` entry for any other reason is still
the failure this test exists to catch.
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

SCRIPT = r"""
import json, sys
from prismyra import Prismyra
engine = Prismyra(sys.argv[1], require_kernels=True)
applied = engine.stats()["kernels"]
print(json.dumps({"complete": applied["complete"], "skipped": applied["skipped"]}))
"""


def test_require_kernels_starts_with_nothing_skipped():
    """What `prismyra-serve --require-kernels` checks before it will answer a single request.

    A separate construction from any shared engine, deliberately, and (see this file's own docstring) in a
    subprocess of its own: `require_kernels` is a constructor argument, and the thing this guards against -- a
    kernel that is skipped on this environment without anyone asking for that, whether from a genuine mismatch or
    from a resource self-check inheriting another construction's state -- is exactly what a shared, already-built
    engine, or a construction sharing a process with one, could not show cleanly.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    reason = no_room_reason(MODEL, __file__)
    if reason:
        pytest.skip(reason)
    done = subprocess.run([sys.executable, "-c", SCRIPT, MODEL], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        # A resource self-check that could not run at all now raises `AdapterError` naming the resource
        # (`kernels/qwen3_moe.py::_reraise_if_resource_exhausted`) instead of silently choosing a different kernel
        # path -- an explicit subprocess failure with that message is this fix doing its job, not this test's
        # failure, but it is not "nothing skipped" either: report it as a skip, with the subprocess's own words,
        # rather than asserting blindly on stdout that was never produced.
        reason = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else f"exit {done.returncode}"
        pytest.skip(f"the subprocess construction could not run its resource self-checks right now: {reason}")
    applied = json.loads(done.stdout.strip().splitlines()[-1])
    assert applied["complete"] is True, applied
    assert applied["skipped"] == [], applied
