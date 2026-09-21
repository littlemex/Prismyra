"""The read-out's refusals and its arithmetic. A stub tokenizer stands in for a real one, so no device is needed."""

from __future__ import annotations

import pytest
import torch

from prismyra import Boolean, Choice, QuestionError
from prismyra.readout import option_token_ids, score


class StubTokenizer:
    """Maps text to ids by a table, so the leading-space rule is visible rather than incidental."""

    def __init__(self, table: dict[str, list[int]]):
        self.table = table

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": self.table[text]}


def test_the_leading_space_is_what_gets_scored():
    """A name that is one token bare and two with a space in front must be refused, not scored on the wrong token."""
    tok = StubTokenizer({"maybe": [7], " maybe": [7, 8], " yes": [1], " no": [2]})
    with pytest.raises(QuestionError, match="tokens with a leading space"):
        option_token_ids(Choice(id="c", prompt="p", choices=["maybe", "yes"]), tok)


def test_options_that_share_a_first_token_are_refused():
    tok = StubTokenizer({" seller": [5], " selling": [5]})
    with pytest.raises(QuestionError, match="same token"):
        option_token_ids(Choice(id="c", prompt="p", choices=["seller", "selling"]), tok)


def test_an_option_with_no_tokens_is_refused():
    tok = StubTokenizer({" a": [], " b": [3]})
    with pytest.raises(QuestionError, match="no tokens"):
        option_token_ids(Choice(id="c", prompt="p", choices=["a", "b"]), tok)


def test_ids_come_back_in_the_declared_order():
    tok = StubTokenizer({" no": [11], " yes": [22]})
    assert option_token_ids(Boolean(id="b", prompt="p"), tok) == [11, 22]


def test_probabilities_are_over_the_declared_options_only():
    """Two questions with different option counts, scored against a tiny vocabulary."""
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # three tokens of width two
    out = score(hidden, unembedding, [[0, 1], [0, 1, 2]])
    assert len(out) == 2
    assert out[0].shape == (2,)
    assert out[1].shape == (3,)
    for p in out:
        assert pytest.approx(float(p.sum()), abs=1e-6) == 1.0
    # The first row points at token 0, so that option must win its own question.
    assert int(out[0].argmax()) == 0


def test_a_row_count_mismatch_is_refused():
    with pytest.raises(ValueError, match="branch rows"):
        score(torch.zeros(2, 4), torch.zeros(3, 4), [[0, 1]])


# --------------------------------------------------------------------------- the quantised output embedding
def test_a_quantised_head_is_refused_when_its_scales_are_missing(monkeypatch, tmp_path):
    """Casting fp8 straight to bf16 drops the scales, and each option reads a different row: the options reorder."""
    from prismyra import readout

    tensors = {"lm_head.weight": torch.zeros(8, 4, dtype=torch.float8_e4m3fn)}
    _stub_checkpoint(monkeypatch, tmp_path, tensors)
    with pytest.raises(RuntimeError, match="no scale was found"):
        readout.load_unembedding("stub/model", hidden_size=4, device="cpu", dtype=torch.float32)


def test_a_quantised_head_is_restored_by_its_scales(monkeypatch, tmp_path):
    from prismyra import readout

    stored = torch.full((4, 4), 2.0, dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[3.0, 5.0]])  # one block of rows, two blocks of columns
    _stub_checkpoint(monkeypatch, tmp_path, {"lm_head.weight": stored, "lm_head.weight_scale_inv": scales})

    got = readout.load_unembedding("stub/model", hidden_size=4, device="cpu", dtype=torch.float32)
    assert torch.equal(got[:, :2], torch.full((4, 2), 6.0))
    assert torch.equal(got[:, 2:], torch.full((4, 2), 10.0))


def _stub_checkpoint(monkeypatch, tmp_path, tensors: dict) -> None:
    """A real single-file checkpoint on disk, with no index.

    A real file rather than a stubbed loader, because what is being tested includes how the file is read: only the two
    tensors that are wanted, not the shard around them.
    """
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    def download(_repo, filename):
        if filename.endswith("index.json"):
            raise EntryNotFoundError("no index")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)


def test_a_partial_last_block_takes_its_own_blocks_scale(monkeypatch, tmp_path):
    """The block size is recovered, not divided out.

    Two hundred rows in two blocks of 128 leaves the second block holding 72 rows. Dividing 200 by 2 gives 100 and hands
    rows 100 to 127 the second multiplier instead of the first -- wrong by a scale factor, on a seventh of the matrix.
    """
    from prismyra import readout

    stored = torch.ones(200, 128, dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[2.0], [7.0]])  # two row blocks of 128, one column block
    _stub_checkpoint(monkeypatch, tmp_path, {"lm_head.weight": stored, "lm_head.weight_scale_inv": scales})

    got = readout.load_unembedding("stub/model", hidden_size=128, device="cpu", dtype=torch.float32)
    assert torch.equal(got[:128], torch.full((128, 128), 2.0))
    assert torch.equal(got[128:], torch.full((72, 128), 7.0))


def test_an_unrecognisable_block_layout_is_refused_rather_than_guessed(monkeypatch, tmp_path):
    from prismyra import readout

    stored = torch.ones(7, 4, dtype=torch.float8_e4m3fn)
    _stub_checkpoint(monkeypatch, tmp_path, {"lm_head.weight": stored, "lm_head.weight_scale_inv": torch.ones(3, 1)})
    with pytest.raises(RuntimeError, match="cannot tell what block size"):
        readout.load_unembedding("stub/model", hidden_size=4, device="cpu", dtype=torch.float32)
