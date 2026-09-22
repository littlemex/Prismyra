"""Faster kernels, applied through a per-model adapter.

Prismyra does not own the model's forward pass. The framework's implementation stays in place and specific modules are
replaced inside it, which keeps that implementation available as the reference every replacement was verified against
-- the same reason a change should not remove its own oracle.

An adapter declares which architectures it handles and how many modules of each kind it expects to find. Finding a
different number means the model is not the one the adapter was measured on, and it fails rather than silently leaving
the slow path in place: a swap that matches nothing looks exactly like a swap that worked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from torch import nn


@dataclass(frozen=True)
class Swap:
    """One replacement that was applied, how many modules it touched, and how that count was checked.

    `expected` is `None` when the count is not the check. Some replacements are identified by structure rather than by
    position, and how many of those a model contains is a fact about one revision of somebody else's module tree -- a
    hardcoded number there breaks on a framework upgrade while proving nothing. Those are verified instead: the
    replacement is run against the implementation it replaces and the outputs must agree. `verified` says so.
    """

    name: str
    replaced: int
    expected: int | None = None
    verified: str | None = None


@dataclass
class Applied:
    """What an adapter did. Deliberately carries no timings: measured figures belong in docs, not in code."""

    adapter: str
    swaps: list[Swap] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: Limits of a replacement that *was* applied, which is not the same as one that was skipped. A caller comparing
    #: against a measured figure needs to know a kernel covers one pass and not the other.
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.replaced == s.expected for s in self.swaps if s.expected is not None)

    def summary(self) -> str:
        parts = [f"{s.name}={s.replaced}" for s in self.swaps]
        if self.skipped:
            parts.append("skipped=" + ",".join(self.skipped))
        return f"{self.adapter}: " + " ".join(parts)

    def as_dict(self) -> dict:
        """The same thing as data. "Am I running degraded?" should not need a string parsed to answer it."""
        return {
            "adapter": self.adapter,
            "complete": self.ok and not self.skipped,
            "applied": {s.name: s.replaced for s in self.swaps},
            "verified": {s.name: s.verified for s in self.swaps if s.verified},
            "skipped": list(self.skipped),
            "notes": list(self.notes),
        }


class Adapter(Protocol):
    """A per-model set of replacements."""

    name: str

    def supports(self, config) -> bool:
        """Whether this adapter was measured on this architecture."""

    def replace(self, model: nn.Module, config) -> Applied:
        """Apply every replacement, or raise. Must be idempotent."""


class AdapterError(RuntimeError):
    """An adapter found a model it does not recognise, or found the wrong number of modules."""


_ADAPTERS: list[Adapter] = []


def register(adapter: Adapter) -> Adapter:
    _ADAPTERS.append(adapter)
    return adapter


def find(config) -> Adapter | None:
    for adapter in _ADAPTERS:
        if adapter.supports(config):
            return adapter
    return None


def apply(model: nn.Module, config, required: bool = False) -> Applied:
    """Replace what can be replaced. Returns what happened.

    With `required`, an unrecognised model raises instead of running slowly. Off by default so a new checkpoint still
    answers, on for a server that would rather not start than serve at a quarter of the speed.
    """
    adapter = find(config)
    if adapter is None:
        arch = getattr(config, "architectures", None)
        if required:
            raise AdapterError(f"no adapter for {arch}; pass required=False to run without the faster kernels")
        return Applied(adapter="none", skipped=["no adapter for this architecture"])
    applied = adapter.replace(model, config)
    if not applied.ok:
        bad = [
            f"{s.name} replaced {s.replaced} of an expected {s.expected}"
            for s in applied.swaps
            if s.expected is not None and s.replaced != s.expected
        ]
        raise AdapterError(f"{adapter.name} found a model it does not recognise: " + "; ".join(bad))
    if required and applied.skipped:
        # A skip is not a failure to find modules, so `ok` stays true; it is a kernel that will not be used. Required
        # means required, and a server that asked for the kernels would rather not start than serve without them.
        raise AdapterError(f"{adapter.name} could not apply every kernel: " + "; ".join(applied.skipped))
    return applied


from . import qwen3_moe as _qwen3_moe  # noqa: F401  (registers itself on import)

__all__ = ["Adapter", "AdapterError", "Applied", "Swap", "apply", "find", "register"]
