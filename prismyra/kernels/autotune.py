"""Pin every timing-based Triton autotuner to one configuration, so that the same input gives the same probabilities in
every process, on every machine of a GPU generation.

Why this exists. Several kernels on the read path are Triton kernels decorated with `triton.autotune`: on their first
call in a process they run each candidate configuration and keep the fastest. When candidates are close, which one
wins is decided by timing noise, and some candidates do not compute the same floating-point sums. The one that was
measured to matter is the flash-linear-attention kernel that inverts the gated-delta layers' triangular blocks
(`merge_16x16_to_64x64_inverse_kernel`): with `num_warps=4` and `num_warps=2` it gives probabilities that differ by up
to 0.59 on single questions, and on an L40S about one cold start in five picked 2. Identical code, weights, driver and
card then answered differently from run to run -- which reads as "the machines disagree".

What it does. At engine start every `triton.runtime.autotuner.Autotuner` in the process is given a single configuration
and its cache of earlier choices is cleared, so no candidate is ever timed again. Which configuration is decided by
data, not code: a file per GPU generation (`pinned/sm_<major><minor>.json`) maps each kernel's name to the
configuration to keep. A kernel the file does not name, or a generation with no file, keeps its **first** declared
candidate -- still the same in every process, possibly slower -- and the result says so (`Pinned.fallback`), as does
`engine.stats()`. Which configuration is fastest, and whether a candidate changes the sums at all, depend on the
generation: on an RTX PRO 4500 (sm_120) the inverse kernel's timing picks num_warps=2 every time, and there 2 and 4
give the same outputs.

File format:

    {"measured_on": "...", "kernels": {"<kernel function name>": {"num_warps": 4, "num_stages": 5, "kwargs": {}}}}

A named configuration must be one of the kernel's own candidates; if a framework upgrade removed it, the kernel falls
back as above and the mismatch is reported, rather than a configuration the kernel never declared being forced on it.
"""

from __future__ import annotations

import gc
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

PINNED_DIR = Path(__file__).parent / "pinned"


@dataclass
class Pinned:
    """What was pinned at start, for `engine.stats()` and for tests."""

    arch: str | None = None
    file: str | None = None
    pinned: dict[str, str] = field(default_factory=dict)
    fallback: dict[str, str] = field(default_factory=dict)
    skipped: str | None = None

    def as_dict(self) -> dict:
        return {
            "arch": self.arch,
            "file": self.file,
            "pinned": dict(self.pinned),
            "fallback": dict(self.fallback),
            "skipped": self.skipped,
        }


def arch_of(device) -> str | None:
    import torch

    if device is None or torch.device(device).type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    return f"sm_{major}{minor}"


def load_table(arch: str | None, directory: Path = PINNED_DIR) -> tuple[dict[str, dict], str | None]:
    """The kernel -> configuration table for a generation, and the file it came from (None when there is none)."""
    if arch is None:
        return {}, None
    path = directory / f"{arch}.json"
    if not path.exists():
        return {}, None
    return dict(json.loads(path.read_text())["kernels"]), str(path)


def kernel_name(tuner) -> str:
    fn = getattr(tuner, "base_fn", None) or tuner.fn
    while not hasattr(fn, "__name__") and hasattr(fn, "fn"):
        fn = fn.fn
    return getattr(fn, "__name__", repr(fn))


def describe(config) -> str:
    kw = ", ".join(f"{k}={v}" for k, v in sorted(config.kwargs.items()))
    return f"num_warps={config.num_warps}, num_stages={config.num_stages}" + (f", {kw}" if kw else "")


def matches(config, wanted: dict) -> bool:
    return (
        config.num_warps == wanted["num_warps"]
        and config.num_stages == wanted["num_stages"]
        and dict(config.kwargs) == dict(wanted.get("kwargs", {}))
    )


def autotuners() -> list:
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return []
    return [o for o in gc.get_objects() if isinstance(o, Autotuner)]


def pin_all(tuners, table: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Give each autotuner one configuration: the table's, or its first. Returns (pinned, fallback) by kernel name."""
    pinned, fallback = {}, {}
    for tuner in tuners:
        name = kernel_name(tuner)
        candidates = list(getattr(tuner, "configs", []) or [])
        if not candidates:
            continue
        wanted = table.get(name)
        chosen = next((c for c in candidates if matches(c, wanted)), None) if wanted is not None else None
        if chosen is None:
            chosen = candidates[0]
            fallback[name] = describe(chosen) + (
                "" if wanted is None else " (the table's configuration is not a candidate)"
            )
        else:
            pinned[name] = describe(chosen)
        tuner.configs = [chosen]
        # Choices made before this point -- the kernels verified at install time run once -- must not survive, or the
        # first, timing-picked configuration would keep being used for those keys.
        if isinstance(getattr(tuner, "cache", None), dict):
            tuner.cache.clear()
    return pinned, fallback


def pin(device, enabled: bool = True) -> Pinned:
    """Pin every autotuner in the process for this device's generation. Process-wide, like the kernel replacements."""
    arch = arch_of(device)
    if not enabled:
        return Pinned(arch=arch, skipped="disabled")
    if arch is None:
        return Pinned(arch=None, skipped="not a CUDA device")
    table, path = load_table(arch)
    pinned, fallback = pin_all(autotuners(), table)
    if fallback and path is None:
        warnings.warn(
            f"no pinned autotune configurations for {arch}: {len(fallback)} autotuned kernel(s) keep their first "
            "candidate. Answers are reproducible from process to process; speed may not be the best available. "
            "engine.stats()['autotune'] lists them.",
            stacklevel=2,
        )
    return Pinned(arch=arch, file=path, pinned=pinned, fallback=fallback)
