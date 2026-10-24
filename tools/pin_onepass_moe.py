#!/usr/bin/env python3
"""Measure this card's one-pass MoE tile and freeze it into `kernels/pinned/onepass_moe_configs/<device>.json`
-- the table `kernels/onepass_moe_tuning.py` reads at import time (see that module's own docstring for why one
dict for every row count, not a second heuristic, is what gets shipped).

Queries vLLM's own `get_default_config` (`vllm.model_executor.layers.fused_moe.fused_moe`) directly, with
`VLLM_BATCH_INVARIANT` unset, at the live engine's own MoE shapes (`E`, the two GEMMs' own `N`/`K`, `topk`, and
the project's fixed FP8 block shape) -- read off a constructed engine rather than hardcoded, so a future
checkpoint with different MoE dimensions is measured on its own shapes, not assumed to match this one -- across
every one-pass bucket width this project ships (`prismyra.fork.WIDTHS`) and every row count a real document's
one-pass read reaches, up to 16,384. `get_default_config`'s own CUDA, block-quantised branch does not read `N`
or `K` at all once `M` decides which arm it takes (checked by reading the function: the tile below `M=65` keys
only on `M`; at and above it, only `block_shape` and `current_platform.is_rocm()` matter) -- but this script
still measures both GEMMs' real shapes rather than one representative pair, because that is itself a fact about
this generation's `get_default_config`, not a given, and the whole point of measuring a card instead of assuming
one already measured generation transfers to the next is to not re-introduce exactly that kind of assumption.

Writes the file only if every queried M, for both GEMMs, returns one identical dict -- the same invariant
`onepass_moe_tuning.py`'s own docstring already claims ("exactly one dict for every row count at or above 128");
a generation whose MoE shapes break that invariant is left unpinned and this script says why on stderr and exits
non-zero, rather than shipping a table that quietly stops applying at some M this project never happened to try.

Usage:
    PRISMYRA_MODEL=littlemex/prismyra-decision-35b-a3b-nvfp4 PRISMYRA_SRC=. python3 tools/pin_onepass_moe.py

Needs a CUDA device (the table is per `torch.cuda.get_device_name`-derived name, the same spelling
`kernels.onepass_moe_tuning.device_name()`/`kernels.fp8_tuning.device_name()` already use) and whatever env vars
the checkpoint on `PRISMYRA_MODEL` needs to load far enough to report its MoE config (see the model card's `Use`
section); `require_kernels=False` on purpose, since this script's whole job is to produce the file that lets a
later `--require-kernels` run succeed, and it must be able to run before that file exists.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))


def moe_shapes(engine) -> list[tuple[str, int, int, int, int]]:
    """(label, E, N, K, topk) for every distinct routed-expert GEMM this engine's backbone declares, read off the
    loaded model rather than assumed -- `qwen3_moe.FusedExperts` (FP8) and `nvfp4.FusedExpertsFp4` (NVFP4) name
    their own weights differently (`w1`/`w2`, packed two values per byte on the NVFP4 side), so this reads
    `w1`/`w2`'s own `.shape[0]` for `E` (unaffected by column packing either way) and both tensors' remaining
    dims for `N`/`K` as found -- *not* corrected for NVFP4's 2-per-byte packing, which would halve them. That
    correction is not needed and is deliberately skipped: `get_default_config`'s own block-quantised branch
    (`kernels/onepass_moe_tuning.py`'s docstring, and this script's own module docstring) does not read `N` or
    `K` once `M` decides which arm it takes, so the exact magnitude only matters for the `M<=64` arm this
    script also probes (and logs, see `measure()`) -- it does not change whether a single dict covers M>64,
    which is the invariant this file is pinning."""
    shapes: dict[str, tuple[int, int, int, int]] = {}
    for _name, m in engine.backbone.named_modules():
        cls = type(m).__name__
        if cls not in ("FusedExperts", "FusedExpertsFp4"):
            continue
        w1, w2 = getattr(m, "w1", None), getattr(m, "w2", None)
        topk = getattr(m, "top_k", None)
        if w1 is None or w2 is None or topk is None:
            continue
        e = int(w1.shape[0])
        shapes[f"{cls}.gemm1"] = (e, int(w1.shape[1]), int(w1.shape[2]), int(topk))
        shapes[f"{cls}.gemm2"] = (int(w2.shape[0]), int(w2.shape[1]), int(w2.shape[2]), int(topk))
    if not shapes:
        raise RuntimeError(
            "no FusedExperts/FusedExpertsFp4 module found on engine.backbone -- is PRISMYRA_MODEL a Prismyra "
            "decision checkpoint with the qwen3_moe adapter applied?"
        )
    return [(label, *dims) for label, dims in sorted(shapes.items())]


#: Every one-pass bucket width this project ships, plus the row counts a real document's one-pass read reaches
#: in practice (a whole document's context plus one question, per `kernels/onepass_moe_tuning.py`'s own
#: docstring) -- enough to find a break in the "one dict above M=64" invariant if this generation has one, not
#: just enough to reproduce what the function happens to do at the handful of M this project tries elsewhere.
PROBE_MS = (
    1, 2, 4, 8, 16, 32, 64, 65, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
    3072, 4096, 6144, 8192, 10000, 12288, 14336, 16384,
)

BLOCK_SHAPE = [128, 128]  # this project's fixed FP8 quantisation block, same as kernels/fp8_tuning.py ships


def measure(shapes: list[tuple[str, int, int, int, int]]) -> dict:
    import os

    os.environ.pop("VLLM_BATCH_INVARIANT", None)
    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    by_m: dict[int, dict] = {}
    for label, e, n, k, topk in shapes:
        for m in PROBE_MS:
            config = fm.get_default_config(m, e, n, k, topk, "fp8_w8a8", BLOCK_SHAPE)
            prior = by_m.get(m)
            if prior is not None and prior != config:
                raise RuntimeError(
                    f"get_default_config disagrees at M={m} between GEMM shapes: {prior} (an earlier GEMM) vs "
                    f"{config} ({label}, E={e} N={n} K={k} topk={topk}) -- the single-dict-per-card approach "
                    "this project ships does not hold for this checkpoint's MoE shapes; do not pin one dict"
                )
            by_m[m] = config

    above_64 = {m: c for m, c in by_m.items() if m > 64}
    # Compared sorted (dict equality, order-independent) so two calls that agree on every key/value but built
    # their dict in a different order do not look like a disagreement; returned in `get_default_config`'s own
    # insertion order (not sorted) so the file this writes reads the same way `NVIDIA_L40S.json` already does.
    distinct_sorted = {tuple(sorted(c.items())) for c in above_64.values()}
    if len(distinct_sorted) != 1:
        raise RuntimeError(
            f"get_default_config is not one constant dict for M>64 on this card: {len(distinct_sorted)} distinct "
            f"configs seen across M={sorted(above_64)} -- the per-card single-tile file this module ships "
            "assumes exactly one; measured values: " + json.dumps([dict(d) for d in distinct_sorted])
        )
    return next(iter(above_64.values()))


def main() -> int:
    model = os.environ.get("PRISMYRA_MODEL")
    if not model:
        print("PRISMYRA_MODEL is required", file=sys.stderr)
        return 2

    from prismyra import Prismyra
    from prismyra.kernels import onepass_moe_tuning

    engine = Prismyra(model, require_kernels=False)
    name = onepass_moe_tuning.device_name()
    if name is None:
        print("no CUDA device visible to this process -- nothing to pin", file=sys.stderr)
        return 2

    shapes = moe_shapes(engine)
    print(f"device: {name}")
    for label, e, n, k, topk in shapes:
        print(f"  {label}: E={e} N={n} K={k} topk={topk}")

    config = measure(shapes)
    print(f"measured (constant for M>64 on all {len(shapes)} GEMM shape(s)): {config}")

    out_path = onepass_moe_tuning.PINNED_DIR / f"{name}.json"
    existing = onepass_moe_tuning.PINNED_DIR / f"{name}.json"
    if existing.exists() and "--force" not in sys.argv:
        print(f"{existing} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
