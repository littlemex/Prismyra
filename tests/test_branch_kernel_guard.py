"""`ask()`'s multi-question path must refuse clearly, not crash inside `transformers`, when this engine has no
borrowed attention kernel -- found on real hardware (see RUN-branch-bug.md): `Qwen/Qwen3-0.6B` (no adapter;
`kernels.qwen3_moe.Qwen3MoeAdapter.supports` wants a `Qwen3_5Moe*` architecture) answers one question
(`_ask_in_one_pass`, which never forks) but raised an `AttributeError` three frames inside
`transformers.integrations.sdpa_attention` for two, sixteen or sixty-four -- `cache.ForkLayer.update()` returns
`(None, None)` for a branch write by design, and only the borrowed kernel knows to read the page table it leaves
instead. Reproduced at transformers 5.15.0, 5.16.0 and 5.17.0 alike, with and without `interleaved_fork`, so the
cause is this engine's own missing precondition check, not a `transformers` version.

No device and no model load: `Prismyra._require_branch_kernel` and the top of `interleave.read_and_branch`/
`read_and_branch_shelf`/`read_and_branch_shelf_many` are plain Python checks that run before anything touches a
tensor, the tokenizer or the backbone -- the same reason `test_admission.py` checks its own arithmetic against a
bare `Fake` rather than a real, 35 GiB engine. Each branch function below is called with every other argument left
`None`/garbage on purpose: if the guard really runs first, nothing after it is ever reached, so a `PrismyraError`
(not a `TypeError`/`AttributeError` from the garbage) is the only way any of these calls can finish.
"""

from __future__ import annotations

import pytest

from prismyra import interleave
from prismyra.engine import Prismyra, PrismyraError
from prismyra.kernels import Applied


class _NoKernelEngine:
    """Enough of a kernel-less `Prismyra` for `_require_branch_kernel` to explain itself -- the real `self.applied`
    an engine built against an unsupported architecture carries (`Prismyra.__init__`'s own
    `kernels.Applied(adapter="none", skipped=[...])` on CPU, or `Qwen3MoeAdapter.supports() == False`'s
    `"no adapter for this architecture"` on CUDA).

    `_require_branch_kernel` is the real, unbound `Prismyra` method, not a copy: this class exists only to stand
    in for the handful of attributes it reads, so a future edit to the real method cannot drift from what this
    test checks the way a hand-copied reimplementation could.
    """

    model_name = "Qwen/Qwen3-0.6B"
    _borrowed_kernel = False
    applied = Applied(adapter="none", skipped=["no adapter for this architecture"])
    _require_branch_kernel = Prismyra._require_branch_kernel


class _KernelEngine(_NoKernelEngine):
    _borrowed_kernel = True


def test_require_branch_kernel_refuses_without_the_borrowed_kernel():
    with pytest.raises(PrismyraError, match="no adapter for this architecture"):
        Prismyra._require_branch_kernel(_NoKernelEngine())


def test_require_branch_kernel_names_the_model_and_the_escape_hatches():
    with pytest.raises(PrismyraError) as exc:
        Prismyra._require_branch_kernel(_NoKernelEngine())
    message = str(exc.value)
    assert "Qwen/Qwen3-0.6B" in message
    assert "_ask_in_one_pass" in message
    assert "require_kernels" in message


def test_require_branch_kernel_is_silent_when_the_kernel_is_installed():
    Prismyra._require_branch_kernel(_KernelEngine())  # must not raise


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (
            interleave.read_and_branch,
            {"encoded": None, "texts": [], "width": 0, "group": 0, "padded_rows": 0},
        ),
        (
            interleave.read_and_branch_shelf,
            {"shelf": None, "context": "", "texts": [], "width": 0, "padded_rows": 0},
        ),
        (
            interleave.read_and_branch_shelf_many,
            {"shelf": None, "contexts": [], "texts_per_doc": [], "width": 0, "padded_rows_per_doc": []},
        ),
    ],
)
def test_interleave_branch_entry_points_refuse_before_touching_the_backbone(fn, kwargs):
    """Every garbage keyword here (`encoded=None`, an empty `shelf`, zero-length lists) would raise a
    `TypeError`/`AttributeError`/`ValueError` of its own the moment the function's *second* line ran -- so a
    `PrismyraError` from `_require_branch_kernel` is only possible if the guard is still the *first* thing each
    function does, exactly as `RUN-branch-bug.md` fixed it (one line, right after the docstring, before
    `engine.backbone`/`engine.tokenizer`/`encoded.tokens` is ever read)."""
    with pytest.raises(PrismyraError, match="no adapter for this architecture"):
        fn(_NoKernelEngine(), **kwargs)


def test_run_recorded_refuses_before_touching_the_cache_or_the_fork():
    """`cache`/`fork`/`run`/`ids`/`positions` are all `None`: `_run_branch` and `_branch_across` both land here
    (see `_run_recorded`'s own docstring), so one check here covers the two-pass branch path the same way the
    three `interleave` functions above cover the fused one."""
    engine = _NoKernelEngine()
    with pytest.raises(PrismyraError, match="no adapter for this architecture"):
        Prismyra._run_recorded(engine, cache=None, fork=None, run=None, ids=None, positions=None, rows=1, width=1)
