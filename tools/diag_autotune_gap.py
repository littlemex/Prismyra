#!/usr/bin/env python3
"""recon: check whether any Triton autotuner relevant to the one-pass path is still open to timing-based
racing (len(tuner.configs) > 1) *after* construction finishes -- i.e. after both `pin_autotunes()`
(engine.py:664) and `onepass.record_all()` (engine.py:679) have run. If `pin_autotunes()` ran too early
(before record_all() created/exercised some autotuner), that autotuner would still show >1 live
candidate here, which is the mechanism hypothesised for the re-record noise seen in
recon_diag_onepass_capture.py (off vs off-re-recorded maxdiff 0.0081, same order as off vs on 0.0063)."""
import os
import sys

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))

from prismyra import Prismyra
from prismyra.kernels.autotune import autotuners, kernel_name

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")

engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=False, wide_group=False)
print("engine.autotune:", engine.autotune)
print()
tuners = autotuners()
print(f"total Autotuner objects visible via gc after construction: {len(tuners)}")
open_to_racing = []
for t in tuners:
    n = kernel_name(t)
    ncfg = len(getattr(t, "configs", []) or [])
    cache_len = len(getattr(t, "cache", {}) or {})
    if ncfg > 1:
        open_to_racing.append((n, ncfg, cache_len))
    print(f"  {n}: configs={ncfg} cache_entries={cache_len}")

print()
print(f"STILL OPEN TO RACING (configs>1) after full construction: {len(open_to_racing)}")
for n, ncfg, cache_len in open_to_racing:
    print(f"  OPEN: {n} configs={ncfg} cache_entries={cache_len}")
