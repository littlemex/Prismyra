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

Until `speed-tiles` (see `RUN-speed-tiles.md`), what this shipped was not a second heuristic, but was still only
one tile: querying vLLM's own `get_default_config` with `VLLM_BATCH_INVARIANT` unset, at every token count this
project's one-pass buckets or real documents reach (64 up to 16,384), returns one identical dict for every row
count at or above 128 -- not because this kernel is row-count invariant, but because that function's own
block-quantised-FP8 branch only varies BLOCK_SIZE_M/GROUP_SIZE_M below M=64 ("Use a small M tile for decode-like
batches") and freezes everything at BLOCK_SIZE_M=64 above it, a decode-era tile applied unconditionally to a
one-pass read's M, which is a whole document's context plus one question -- thousands of rows, not a decode
step. `tools/pin_onepass_moe.py` measured and froze exactly that single answer.

What ships now is a tile *per M bucket*, the same idea `kernels/fp8_tuning.py` already ships for the dense-FP8
projections, extended to the one place that still had a single frozen tile for every M above 64. Safe to vary
without moving the answer: `fused_moe_kernel`'s K-loop (`for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))`) is strictly
sequential and its trip count and stride depend only on `BLOCK_SIZE_K`; `SPLIT_K` is declared as a kernel
parameter but never read inside the kernel body in the vLLM version this project pins (checked: grep finds it
only ever assigned by callers, always to 1, never referenced inside the `@triton.jit` function), so it cannot
move the reduction order either. `BLOCK_SIZE_M`/`BLOCK_SIZE_N` only choose which tile of the *output* a program
computes; `GROUP_SIZE_M` only reorders *which* program computes which tile (L2 locality); `num_warps` splits a
tile's own compute across warps, not the K-reduction inside one `tl.dot`; `num_stages` only changes
software-pipelining depth. None of these can change the sequence of floating-point operations that produces one
output element once `BLOCK_SIZE_K` (and `SPLIT_K`) are held fixed -- confirmed, not assumed: `tools/speedtiles_build.py`
checked every candidate `torch.equal` against today's frozen tile, on this checkpoint's real routed-expert GEMM
shapes, before timing it; zero mismatches across 216 candidates at every M bucket this table ships
(`RUN-speed-tiles.md`). `kernels/pinned/onepass_moe_configs/<device>.json` is now `{"<m bucket>": {tile...}, ...}`,
read once at import time; `tile_for(m)` picks the bucket whose key is numerically closest to the real M (the same
nearest-bucket rule vLLM's own `try_get_optimal_moe_config`/`get_w8a8_block_fp8_configs` already use for their own
tables), so a given M always maps to the same tile every time -- determinism is a property of the table being a
pure function of M, not of the table being flat. A card with no file here is untouched -- the ordinary,
already-shipped batch-invariant tile keeps running, same as before this module existed.

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
    except Exception:  # noqa: BLE001 - best-effort device probe; any failure here means "no pinned table"
        return None


_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

#: The 7 keys a single tile needs -- unchanged from the flat, pre-`speed-tiles` table format. A bucketed table
#: is a dict whose keys are *these* (the legacy flat format, kept recognisable rather than silently reinterpreted
#: as a one-bucket table with a bucket key that happens to collide with a tile key) or a dict of `{"<m>": tile}`
#: entries; `is_flat_tile` tells the two apart without guessing from key *count*, which a future tile schema
#: could change.
_TILE_KEYS = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "SPLIT_K", "num_warps", "num_stages"}


def is_flat_tile(data: dict) -> bool:
    return set(data) == _TILE_KEYS


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
_PINNED: dict | bool | None = False


def pinned_config() -> dict | None:
    """This card's pinned table, from disk, cached after the first call: either `{"<m bucket>": tile, ...}`
    (the current, `speed-tiles` format) or a single flat tile dict (the format every table shipped before it,
    kept readable by `tile_for()` below rather than forced through a migration). `None` means no table for this
    card at all. Not gated on the escape hatch below -- that is checked fresh on every call in `scope()`, not
    baked into this cache, so flipping it mid-process (an interleaved A/B measurement, `BRIEF-COMMON.md` rule 3)
    takes effect on the very next call. Callers that only need "is there a table for this card at all"
    (`engine.py`'s `require_kernels` check) can keep using this function's truthiness exactly as before; callers
    that need the tile for a specific M should use `tile_for()`."""
    global _PINNED  # noqa: PLW0603 - a process-wide, read-only-after-first-call cache, by design (see above)
    if _PINNED is False:
        name = device_name()
        _PINNED = _load(name) if name else None
    return _PINNED or None


def tile_for(m: int) -> dict | None:
    """The tile this card's table says to use for row count `m`, or `None` if this card has no table at all.

    A bucketed table picks the bucket whose key is numerically closest to `m` -- the same rule vLLM's own
    `try_get_optimal_moe_config`/`get_w8a8_block_fp8_configs` already use for *their* per-M tables
    (`configs[min(configs.keys(), key=lambda x: abs(x - M))]`), so this project's own bucket choice is not a new
    convention. A flat (pre-`speed-tiles`) table has no buckets to choose between and returns its one tile for
    every `m`, exactly as `scope()` already behaved before this function existed."""
    table = pinned_config()
    if table is None:
        return None
    if is_flat_tile(table):
        return table
    bucket = _nearest_bucket(table, m)
    return None if bucket is None else table[bucket]


def _nearest_bucket(table: dict, m: int) -> str | None:
    """The key of `table` (a `{"<m bucket>": tile}` dict) numerically closest to `m`, or `None` if `table` has
    no buckets at all. A standalone function, not inlined into `tile_for()`, so a test can check the selection
    rule itself -- independent of file loading, caching, or the flat-table escape hatch -- against a table whose
    keys do not have to be real measured buckets."""
    if not table:
        return None
    return min(table, key=lambda k: abs(int(k) - m))


def _enabled() -> bool:
    """An escape hatch, not a second selection rule: read fresh every call, the same way
    `PRISMYRA_INVARIANCE_SCOPE`/`PRISMYRA_WITHOUT` already read their own env vars elsewhere in this package --
    for measuring this change against the unpinned baseline in one process without restarting it, and for turning
    it off without a new release if it is ever found to be unsafe. Does not change *what* is pinned, only
    *whether* a given call uses it."""
    import os

    return os.environ.get("PRISMYRA_ONEPASS_MOE_PIN", "1") != "0"


@contextlib.contextmanager
def scope(m: int):
    """While this block runs, vLLM's fused-MoE call picks this card's pinned tile for row count `m` instead of
    whatever `VLLM_BATCH_INVARIANT`/the row-count heuristic would otherwise choose -- and always the same one for
    the same `m`, regardless of anything else about the process. `m` must be the real row count this block's own
    forward will carry (`Prismyra._read_one_pass`'s `ids.shape[1]`, or `onepass.record_bucket`'s own `length`):
    since a one-pass call has no companion by construction (this module's own docstring), `m` here is never
    shared with, or moved by, anyone else's request -- the one condition a per-M table needs to keep the
    companion/question-count invariance every other pinned table in this project already has. Restores whatever
    override (usually none) was active before, even if the block raises; a no-op, including the restore, on a
    card with nothing pinned.

    Writes `vllm.model_executor.layers.fused_moe`'s own module-level `_config` directly rather than using that
    module's `override_config` context manager: that context manager's own generator has no `try`/`finally` around
    its `yield`, so an exception raised inside its `with` block leaves `_config` set to this call's override for
    every later call in the process -- a global leak this module cannot risk for a path meant to touch nothing
    outside its own one-pass calls.
    """
    config = tile_for(m) if _enabled() else None
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
