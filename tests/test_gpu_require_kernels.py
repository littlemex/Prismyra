"""`prismyra-serve --require-kernels`, which builds its own engine -- so it lives apart from `test_gpu.py`, whose
module-scoped engine would otherwise still hold one copy of the weights when this builds a second. On a 44 GB card two
copies do not fit, and the test failed every time for that reason rather than for the one it exists for.

    pytest -m gpu tests/test_gpu_require_kernels.py
"""

from __future__ import annotations

import os

import pytest
import torch

from prismyra import Prismyra

pytestmark = pytest.mark.gpu

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")


def test_require_kernels_starts_with_nothing_skipped():
    """What `prismyra-serve --require-kernels` checks before it will answer a single request.

    A separate construction from any shared engine, deliberately: `require_kernels` is a constructor argument, and
    the thing this guards against -- a kernel that is skipped on this environment without anyone asking for that --
    is exactly what a shared, already-built engine could not show. It loads the checkpoint itself, which is why it is
    in a file of its own.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        engine = Prismyra(MODEL, require_kernels=True)
    except torch.OutOfMemoryError:
        # Only when another engine in this process still holds the weights (this file run in the same session as one
        # that keeps an engine alive). Run alone, it has the card to itself.
        pytest.skip("another engine in this process holds the weights; run this file on its own")
    applied = engine.stats()["kernels"]
    assert applied["complete"] is True, applied
    assert applied["skipped"] == [], applied
