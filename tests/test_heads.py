"""Option-set heads without a device: the registry, the matching rule, the arithmetic, and the refusals."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismyra import PrismyraError
from prismyra.heads import Heads

HIDDEN = 8


def write_spec(tmp_path, entries, tensors):
    for name, t in tensors.items():
        save_file(t, str(tmp_path / f"{name}.safetensors"))
    path = tmp_path / "heads.json"
    path.write_text(json.dumps(entries))
    return path


def linear(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"W": torch.randn(k, HIDDEN, generator=g), "b": torch.randn(k, generator=g)}


def entry(name, options, form="linear"):
    return {"name": name, "options": options, "form": form, "weights": f"{name}.safetensors"}


def test_no_spec_is_empty_and_changes_nothing():
    heads = Heads(None, HIDDEN, "cpu")
    rows = [torch.tensor([0.3, 0.7]), torch.tensor([0.1, 0.2, 0.7])]
    assert not heads
    out = heads.apply(torch.randn(2, HIDDEN), [["no", "yes"], ["a", "b", "c"]], rows)
    assert out is rows


def test_only_the_exact_option_list_is_answered_by_the_head(tmp_path):
    spec = write_spec(tmp_path, [entry("tiers", ["low", "mid", "high"])], {"tiers": linear(3)})
    heads = Heads(spec, HIDDEN, "cpu")
    assert heads.name_for(["low", "mid", "high"]) == "tiers"
    assert heads.name_for(["high", "mid", "low"]) == "tiers"  # one set, as the rendered question is one prompt
    assert heads.name_for(["low", "mid"]) is None
    assert heads.name_for(["low", "mid", "high", "none"]) is None
    assert heads.name_for(["no", "yes"]) is None


def test_a_reordered_declaration_gets_the_heads_probabilities_in_its_own_order(tmp_path):
    w = linear(3)
    heads = Heads(write_spec(tmp_path, [entry("tiers", ["low", "mid", "high"])], {"tiers": w}), HIDDEN, "cpu")
    h = torch.randn(1, HIDDEN)
    (own,) = heads.apply(h, [["low", "mid", "high"]], [torch.zeros(3)])
    (other,) = heads.apply(h, [["high", "low", "mid"]], [torch.zeros(3)])
    torch.testing.assert_close(other, own[[2, 0, 1]])


def test_rows_without_a_head_come_back_as_the_same_tensors(tmp_path):
    w = linear(3)
    heads = Heads(write_spec(tmp_path, [entry("tiers", ["low", "mid", "high"])], {"tiers": w}), HIDDEN, "cpu")
    hidden = torch.randn(3, HIDDEN)
    rows = [torch.tensor([0.4, 0.6]), torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.5, 0.25, 0.25])]
    out = heads.apply(hidden, [["no", "yes"], ["low", "mid", "high"], ["a", "b", "c"]], rows)
    assert out[0] is rows[0] and out[2] is rows[2]
    expected = torch.softmax(hidden[1] @ w["W"].t() + w["b"], dim=-1)
    torch.testing.assert_close(out[1], expected)


def test_an_mlp_head_computes_what_its_spec_says(tmp_path):
    g = torch.Generator().manual_seed(1)
    w = {
        "W1": torch.randn(5, HIDDEN, generator=g),
        "b1": torch.randn(5, generator=g),
        "W2": torch.randn(2, 5, generator=g),
        "b2": torch.randn(2, generator=g),
    }
    heads = Heads(write_spec(tmp_path, [entry("pair", ["left", "right"], "mlp")], {"pair": w}), HIDDEN, "cpu")
    h = torch.randn(1, HIDDEN)
    (out,) = heads.apply(h, [["left", "right"]], [torch.tensor([0.5, 0.5])])
    z = torch.nn.functional.gelu(h[0] @ w["W1"].t() + w["b1"])
    torch.testing.assert_close(out, torch.softmax(z @ w["W2"].t() + w["b2"], dim=-1))


@pytest.mark.parametrize(
    "entries, tensors, message",
    [
        ([entry("x", ["a", "b"])], {"x": {"W": torch.zeros(2, HIDDEN + 1), "b": torch.zeros(2)}}, "hidden state"),
        ([entry("x", ["a", "b"])], {"x": linear(3)}, "scores 3 options"),
        ([entry("x", ["a", "b"], "tree")], {"x": linear(2)}, "form must be"),
        ([entry("x", ["a", "b"])], {"x": {"W": torch.zeros(2, HIDDEN)}}, "missing b"),
        ([entry("x", ["a", "a"])], {"x": linear(2)}, "twice"),
        ([entry("x", ["a", "b"]), entry("y", ["a", "b"])], {"x": linear(2), "y": linear(2)}, "same options"),
    ],
)
def test_a_spec_that_cannot_be_right_is_refused_at_load(tmp_path, entries, tensors, message):
    with pytest.raises(PrismyraError, match=message):
        Heads(write_spec(tmp_path, entries, tensors), HIDDEN, "cpu")


def test_recording_keeps_only_rows_of_the_named_set_and_changes_no_answer(tmp_path):
    from prismyra.heads import Recorder

    rec = Recorder(["low", "mid", "high"], Heads(None, HIDDEN, "cpu"))
    hidden = torch.randn(3, HIDDEN)
    rows = [torch.tensor([0.5, 0.5]), torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.6, 0.4])]
    out = rec.apply(hidden, [["no", "yes"], ["high", "low", "mid"], ["a", "b"]], rows)
    assert out is rows
    assert len(rec.rows) == 1 and torch.equal(rec.rows[0], hidden[1])
