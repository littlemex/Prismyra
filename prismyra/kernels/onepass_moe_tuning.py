"""A faster, pinned MoE tile config for the one-pass solo-question path only.

`_enable_batch_invariance()` sets `VLLM_BATCH_INVARIANT=1` for the whole process, unconditionally, on every CUDA
engine (`engine.py`'s `__init__`) -- including `Prismyra._ask_in_one_pass`, a path with no companion and no fork
by construction (docs/FORK.md, `ask()`'s own docstring) and already measured bit-divergent from the forked path at
question count one (`tools/audit_sm120.py`'s `paged_vs_joined` section, `PAGED_VS_JOINED_MOVEMENT`). The companion
protection that env var buys has nothing to protect here, and vLLM's own fused-MoE kernel (`fused_moe.py`'s
`get_default_config`) pays for it anyway: with the env var set, every expert GEMM uses a fixed small tile
(`BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8`) regardless of how many rows the call carries. At the row counts
a one-pass read actually sees -- a whole document's context plus one question, measured in the thousands of
tokens -- that tile is the wrong shape for the GEMM: `RUN-spd5.md`'s nsys trace found the routed-expert kernel
taking 58% longer than an unwrapped vLLM serving the same checkpoint shape, from twice the grid blocks and an eighth
of the shared memory per block.

What this ships instead is not a second heuristic. Querying vLLM's own `get_default_config` with
`VLLM_BATCH_INVARIANT` unset, at every token count this project's one-pass buckets or real documents reach
(64 up to 16,384; see `tools/pin_onepass_moe.py`), returns exactly one dict for every row count at or above 128 --
the whole range a context-plus-question read falls in. That one dict, frozen into
`kernels/pinned/onepass_moe_configs/<device>.json`, is what gets installed: a single, fixed table, read once at
import time, never re-chosen by row count, never timed, the same shape `kernels/fp8_tuning.py` already ships for
the dense-FP8 projections. A card with no file here is untouched -- the ordinary, already-shipped batch-invariant
tile keeps running, same as before this module existed.

Scope, by construction rather than by a flag: this module is only ever entered from `Prismyra._read_one_pass` and
`onepass.record_bucket`'s `run()`, the two call sites `_ask_in_one_pass`/`onepass.record_all` use for the model's
real forward. The fork path (`_run_branch`/`_run_recorded`, `graphs.py`), `open_batch`, and `Batcher`/`Shelf` never
call either, so they keep paying -- and keep being protected by -- the global `VLLM_BATCH_INVARIANT` registration
exactly as before. The router's own GEMM (`qwen3_moe.py`'s `_ROUTER_LINEAR`) is unconditional, not gated on this env
var at all, so which experts a token routes to does not move; what this changes is only the reduction order inside
the already-chosen experts' own GEMM, the same kind of change `_enable_batch_invariance`'s own docstring describes
for every other fixed-tile choice this project pins.

NVFP4 (`kernels/nvfp4.py`, the Blackwell/RTX PRO 4500 checkpoint) was checked and found to need nothing here: its
expert GEMM is a CUTLASS grouped kernel tuned by `autotune_tactics()`, which already pins *one* tactic -- sized for
the largest bucket it tunes, `max_tokens=16384` -- for every row count a process ever asks it to run, specifically so
a one-pass read and a branch pass never pick different tactics for the same reason this module exists for the FP8
path. It does not read `VLLM_BATCH_INVARIANT` at all (checked by searching `kernels/nvfp4.py` for the name: zero
hits), so there is no analogous small-tile forcing on that card's one-pass path to undo.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

PINNED_DIR = Path(__file__).parent / "pinned" / "onepass_moe_configs"


def device_name() -> str | None:
    """This process's device name, spelled the way `kernels.fp8_tuning.device_name` already spells it (spaces ->
    underscores) -- the same convention, so a future table for a new card can be dropped in without inventing a
    second naming rule."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        from vllm.platforms import current_platform

        return current_platform.get_device_name().replace(" ", "_")
    except Exception:
        return None


_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _load(name: str) -> dict | None:
    if not _NAME_RE.match(name):
        return None
    path = PINNED_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    return config


#: Read from disk once, read-only for the rest of the process -- the file never changes while a process runs, so
#: there is nothing to gain by re-reading it. `None` means "no pinned table for this card" (the ordinary
#: batch-invariant tile keeps running); `False` is the sentinel for "not loaded yet", so a card without a table is
#: not re-read from disk on every one-pass call.
_PINNED: dict | None | bool = False


def pinned_config() -> dict | None:
    """This card's pinned table, from disk, cached after the first call. Not gated on the escape hatch below --
    that is checked fresh on every call in `scope()`, not baked into this cache, so flipping it mid-process (an
    interleaved A/B measurement, `BRIEF-COMMON.md` rule 3) takes effect on the very next call."""
    global _PINNED
    if _PINNED is False:
        name = device_name()
        _PINNED = _load(name) if name else None
    return _PINNED or None


def _enabled() -> bool:
    """An escape hatch, not a second selection rule: read fresh every call, the same way
    `PRISMYRA_INVARIANCE_SCOPE`/`PRISMYRA_WITHOUT` already read their own env vars elsewhere in this package --
    for measuring this change against the unpinned baseline in one process without restarting it, and for turning
    it off without a new release if it is ever found to be unsafe. Does not change *what* is pinned, only
    *whether* a given call uses it."""
    import os

    return os.environ.get("PRISMYRA_ONEPASS_MOE_PIN", "1") != "0"


@contextlib.contextmanager
def scope():
    """While this block runs, vLLM's fused-MoE call picks this card's pinned one-pass tile instead of whatever
    `VLLM_BATCH_INVARIANT`/the row-count heuristic would otherwise choose -- and always the same one, regardless of
    how many rows this particular call carries. Restores whatever override (usually none) was active before, even
    if the block raises; a no-op, including the restore, on a card with nothing pinned.

    Writes `vllm.model_executor.layers.fused_moe`'s own module-level `_config` directly rather than using that
    module's `override_config` context manager: that context manager's own generator has no `try`/`finally` around
    its `yield`, so an exception raised inside its `with` block leaves `_config` set to this call's override for
    every later call in the process -- a global leak this module cannot risk for a path meant to touch nothing
    outside its own one-pass calls.
    """
    config = pinned_config() if _enabled() else None
    if config is None:
        yield
        return
    import vllm.model_executor.layers.fused_moe as _moe_pkg

    old = _moe_pkg._config
    _moe_pkg._config = config
    try:
        yield
    finally:
        _moe_pkg._config = old
