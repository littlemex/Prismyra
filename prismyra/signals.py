"""Signals read from the context pass: linear probes on a hidden state the engine already computes.

A probe is one layer, one position -- the last context token -- a weight vector and a bias. It is applied to the
residual stream that layer produces while the context is read, so it adds one dot product per probe and no pass. The
value is `sigmoid(w . h + b)`, returned beside the answers as `Result.signals[name]`.

What a probe predicts is a property of the file, not of this module: the file names its target and what it was
trained on, and a probe fit on one checkpoint's hidden states means nothing on another's. So a probe must record the
hidden size and layer count it was fit against, and is refused without them or on a model with different ones. It cannot
carry the weights themselves, so rebuilding a probe after changing the checkpoint -- merging an adapter, truncating
layers -- remains the caller's job, and `recipes/signals` says how.

Measured on the supported model: a self_solve probe (does this model, used as a chat LLM, answer the context
correctly) ranks problems within a family at AUROC 0.752 against 0.688 for asking the question through the read-out.
Probes for PII and prompt injection did not transfer across sources; asking through the read-out was better there.

The forked read and the one-pass read (`Prismyra.ask` with one question) reach the last context token by different
arithmetic -- the one-pass read computes it inside a longer sequence -- so a probe's value differs between them by
the rounding of that position's hidden state, amplified by the probe: over 100 problems, 1e-7 on the forked path
against the value the probe was fit on, and a mean of 0.016 (largest 0.18) on the one-pass path.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


class ProbeError(ValueError):
    """A probe file that cannot be applied to this model."""


@dataclass
class Probe:
    name: str
    layer: int  # 1-based: the output of decoder layer `layer`, as `output_hidden_states[layer]` would number it
    weight: torch.Tensor
    bias: float
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Probe:
        raw = json.loads(Path(path).read_text())
        missing = {"name", "layer", "weight", "bias"} - set(raw)
        if missing:
            raise ProbeError(f"{path}: probe file is missing {sorted(missing)}")
        if raw.get("position", "last_context_token") != "last_context_token":
            raise ProbeError(f"{path}: only last_context_token probes are supported, not {raw['position']!r}")
        meta = {k: v for k, v in raw.items() if k not in ("name", "layer", "weight", "bias")}
        weight = torch.tensor(raw["weight"], dtype=torch.float32)
        if not (torch.isfinite(weight).all() and math.isfinite(float(raw["bias"]))):
            raise ProbeError(f"{path}: the probe's weights or bias are not finite")
        return cls(raw["name"], int(raw["layer"]), weight, float(raw["bias"]), meta)

    def check(self, hidden_size: int, layers: int) -> None:
        if self.weight.shape != (hidden_size,):
            raise ProbeError(
                f"probe {self.name!r} has {self.weight.numel()} weights and this model's hidden size is {hidden_size}"
            )
        if not 1 <= self.layer <= layers:
            raise ProbeError(f"probe {self.name!r} reads layer {self.layer} and this model has {layers}")
        fit = self.meta.get("model")
        if not isinstance(fit, dict) or "layers" not in fit or "hidden_size" not in fit:
            # A layer index means nothing without the depth it was counted in: a layer-31 probe fit on a 40-layer model
            # would load on a truncated 32-layer one of the same width and read a different stage of the computation.
            raise ProbeError(
                f"probe {self.name!r} does not record the model shape it was fit on (model.layers, model.hidden_size)"
            )
        if (fit["hidden_size"], fit["layers"]) != (hidden_size, layers):
            raise ProbeError(
                f"probe {self.name!r} was fit on a {fit['hidden_size']} x {fit['layers']} model, "
                f"not {hidden_size} x {layers}"
            )


class SignalReader:
    """Hooks the layers the probes read and evaluates them at one position of a forward pass it is armed for.

    Armed rather than always on: the branch passes run the same layers and must not overwrite what the context read
    left, and a model with no probes loaded installs no hook at all. `capture` disarms however the pass ends, so a read
    that raises cannot leave a stale position for the next, unrelated pass.
    """

    def __init__(self, probes: list[Probe], decoder_layers: nn.ModuleList, device: torch.device):
        names = [p.name for p in probes]
        if len(set(names)) != len(names):
            raise ProbeError(f"two probes share a name, and one value would silently replace the other: {names}")
        self.probes = probes
        self._at: int | None = None
        self._seen: dict[int, torch.Tensor] = {}
        for p in probes:
            p.weight = p.weight.to(device)
        layers = sorted({p.layer for p in probes})
        self._handles = [decoder_layers[layer - 1].register_forward_hook(self._hook(layer)) for layer in layers]

    def _hook(self, layer: int):
        def capture(module, args, output):
            if self._at is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if not 0 <= self._at < hidden.shape[1]:
                raise ProbeError(f"asked to read position {self._at} of a pass {hidden.shape[1]} tokens long")
            # Copied: a view would keep the whole layer output alive for the rest of the read.
            self._seen[layer] = hidden[0, self._at].to(torch.float32, copy=True)

        return capture

    @contextmanager
    def capture(self, position: int):
        """Arm for one forward pass at `position`; the dict it yields holds the probe values once the block exits."""
        out: dict[str, float] = {}
        self._at, self._seen = position, {}
        try:
            yield out
            for p in self.probes:
                if p.layer in self._seen:
                    out[p.name] = float(torch.sigmoid(self._seen[p.layer] @ p.weight + p.bias))
        finally:
            self._at, self._seen = None, {}

    def close(self) -> None:
        """Remove the hooks. The engine never needs this; a test that attaches a reader to a shared engine does."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
