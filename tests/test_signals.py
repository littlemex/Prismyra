"""Probe files and the reader that applies them, on the CPU."""

import json

import pytest
import torch
from torch import nn

from prismyra.signals import Probe, ProbeError, SignalReader


def write(tmp_path, **kw):
    body = {
        "name": "self_solve",
        "layer": 2,
        "weight": [0.5, -1.0, 2.0],
        "bias": 0.25,
        "model": {"hidden_size": 3, "layers": 4},
        **kw,
    }
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(body))
    return path


def test_a_probe_loads_and_checks_its_shape(tmp_path):
    probe = Probe.load(write(tmp_path))
    probe.check(hidden_size=3, layers=4)
    with pytest.raises(ProbeError):
        probe.check(hidden_size=4, layers=4)
    with pytest.raises(ProbeError):
        probe.check(hidden_size=3, layers=1)


def test_a_probe_without_its_model_shape_is_refused(tmp_path):
    for model in (None, "a checkpoint name", {"hidden_size": 3}):
        body = {"model": model} if model is not None else {}
        path = write(tmp_path, **body)
        if model is None:
            raw = json.loads(path.read_text())
            raw.pop("model")
            path.write_text(json.dumps(raw))
        with pytest.raises(ProbeError):
            Probe.load(path).check(hidden_size=3, layers=4)


def test_non_finite_weights_are_refused(tmp_path):
    with pytest.raises(ProbeError):
        Probe.load(write(tmp_path, weight=[0.0, float("nan"), 1.0]))


def test_a_probe_fit_on_another_model_is_refused(tmp_path):
    probe = Probe.load(write(tmp_path, model={"hidden_size": 3, "layers": 40}))
    with pytest.raises(ProbeError):
        probe.check(hidden_size=3, layers=32)


def test_only_the_last_context_token_is_supported(tmp_path):
    with pytest.raises(ProbeError):
        Probe.load(write(tmp_path, position="mean"))


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Identity() for _ in range(3))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + 1
        return x


def test_the_reader_applies_the_probe_at_the_armed_position_only(tmp_path):
    stack = Stack()
    reader = SignalReader([Probe.load(write(tmp_path))], stack.layers, torch.device("cpu"))
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    stack(x)  # not armed: nothing is captured, and nothing raises
    with reader.capture(0) as got:
        stack(x)
    # layer 2 is the second module; its input and output are x + 1
    z = float((x[0, 0] + 1) @ torch.tensor([0.5, -1.0, 2.0])) + 0.25
    assert got["self_solve"] == pytest.approx(float(torch.sigmoid(torch.tensor(z))))


def test_an_extreme_logit_does_not_overflow(tmp_path):
    stack = Stack()
    reader = SignalReader([Probe.load(write(tmp_path, bias=-1e6))], stack.layers, torch.device("cpu"))
    with reader.capture(0) as got:
        stack(torch.ones(1, 2, 3))
    assert got["self_solve"] == 0.0


def test_a_failed_pass_leaves_the_reader_disarmed(tmp_path):
    stack = Stack()
    reader = SignalReader([Probe.load(write(tmp_path))], stack.layers, torch.device("cpu"))
    with pytest.raises(RuntimeError), reader.capture(0):
        raise RuntimeError("the read ran out of memory")
    stack(torch.ones(1, 1, 3))  # would raise on a stale position if the reader were still armed
    assert reader._at is None


def test_two_probes_with_one_name_are_refused(tmp_path):
    a = Probe.load(write(tmp_path))
    b = Probe.load(write(tmp_path, layer=3))
    with pytest.raises(ProbeError):
        SignalReader([a, b], Stack().layers, torch.device("cpu"))
