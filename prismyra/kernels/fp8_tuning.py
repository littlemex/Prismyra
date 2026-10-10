"""Carry this card's dense-FP8 matmul tuning with the package, instead of leaving it to evaporate with the machine
it was tuned on.

Why this exists. vLLM's block-FP8 Triton matmul (`w8a8_triton_block_scaled_mm`, used by `Fp8Linear` and by every FP8
dense projection in both the FP8-only and the NVFP4-experts checkpoints) is not itself a `triton.autotune` kernel --
it is a plain `@triton.jit` kernel whose tile size vLLM looks up from a static table, keyed by shape and this card's
name, in `get_w8a8_block_fp8_configs()`. That table is a JSON file vLLM reads from *inside its own installed package*
(`vllm/model_executor/layers/quantization/utils/configs/`). Nothing ships it there: the project's own tuner
writes the file straight into that pip-installed path, on whichever machine happened to run it. A freshly built
machine -- the normal state once a short-lived GPU machine is torn down and rebuilt -- starts with none of
those files, vLLM logs "Using default W8A8 Block FP8 kernel config... Performance might be sub-optimal!" for every
shape, and every dense FP8 matmul runs vLLM's one hard-coded fallback tile (`BLOCK_SIZE_M=64, N=128, K=128,
GROUP_SIZE_M=32, num_warps=4, num_stages=2`) at every row count, including the branch passes (M~16-128) where the
fallback is measured at roughly 2x slower than the tuned tile (`BLOCK_SIZE_M=16, num_stages=4` wins there) and the
long prefill passes (M~thousands) where it is roughly 10-20% slower. The project's own measured "+10-20% per matmul"
figure for this tuning was never at risk of being wrong -- it was at risk of being present only on the one machine that
happened to run the tuner, and silently absent on every machine built since.

What this does. Ships the already-tuned tables for the generations this project measures (one JSON file per
(N, K, device name) cell, same format the project's own tuner writes, same `BLOCK_SIZE_K=128` as the untuned
fallback so the inner reduction loop sums the same K-blocks in the same order -- `torch.equal` against the
fallback was checked for every cell this project's checkpoints use before any file here was kept) under
`pinned/fp8_block_configs/`, and copies the ones that match this process's device into vLLM's own configs
directory before the first dense FP8 matmul runs, if vLLM does not already have a file there. It never
overwrites a file vLLM already has: a machine someone tuned by hand, or a newer measurement placed there in this
same session, is left alone.
"""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

TUNED_DIR = Path(__file__).parent / "pinned" / "fp8_block_configs"
_NAME_RE = re.compile(r"device_name=([^,]+),dtype=")


def device_name() -> str | None:
    """This process's device name, as vLLM's own config loader spells it (spaces -> underscores)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        from vllm.platforms import current_platform

        return current_platform.get_device_name().replace(" ", "_")
    except Exception:  # noqa: BLE001 - best-effort device probe; any failure here means "untuned", not a crash
        return None


def _vllm_configs_dir() -> Path | None:
    try:
        import vllm.model_executor.layers.quantization.utils.fp8_utils as F
    except Exception:  # noqa: BLE001 - best-effort probe; an unexpected vLLM layout means "nothing to install"
        return None
    return Path(F.__file__).parent / "configs"


def install() -> list[str]:
    """Copy this process's device's tuned tables into vLLM's configs directory. Returns the filenames copied; a file
    vLLM already has is left untouched and does not appear here."""
    name = device_name()
    dest_dir = _vllm_configs_dir()
    if name is None or dest_dir is None or not TUNED_DIR.exists():
        return []
    copied = []
    for src in sorted(TUNED_DIR.glob("*.json")):
        m = _NAME_RE.search(src.name)
        if not m or m.group(1) != name:
            continue
        dest = dest_dir / src.name
        if dest.exists():
            continue  # never clobber a table already on this machine, hand-placed or freshly re-tuned
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(src.name)
        except OSError as e:
            warnings.warn(f"could not install the tuned FP8 matmul table {src.name}: {e}", stacklevel=2)
    return copied
