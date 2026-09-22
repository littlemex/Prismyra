"""Turning a branch's final hidden state into a probability over the declared options.

The model's own output embedding does the work: each option's single token is scored, and those scores are softmaxed
over the declared options only. Nothing is trained. See `schema.SCORING` for what that means and does not mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .schema import Question, QuestionError, render_question


@dataclass(frozen=True)
class Plan:
    """How one question will be asked and read: the text of its branch, and the token scored for each option."""

    text: str
    token_ids: list[int]
    trailing_space: bool


def plan(question: Question, tokenizer) -> Plan:
    """Choose the rendering this tokenizer can actually answer, and the token to score for each option.

    One position is read, so one token per option has to identify it. Two renderings are tried, and which one works is a
    property of the tokenizer rather than of the question:

    * the prompt ends `Answer:` and the option is scored **with a leading space**, which is how the token appears there.
      This model's tokenizer merges a space into a word, so " yes" and " buyer" are each one token;
    * the prompt ends `Answer: ` and the bare option is scored. The same tokenizer splits a space from a digit, so " 1"
      is two tokens and the first is the space -- identical for every option, and so no answer at all. Putting the space
      in the prompt moves the digit to the position that gets read.

    The first that gives every option a distinct single token wins. If neither does, the question is refused rather than
    scored on the wrong token, because an option truncated to its first piece is answerable and wrong.
    """
    attempts = []
    for trailing_space in (False, True):
        try:
            ids = option_token_ids(question, tokenizer, trailing_space=trailing_space)
        except QuestionError as e:
            attempts.append(str(e))
            continue
        return Plan(text=render_question(question, trailing_space), token_ids=ids, trailing_space=trailing_space)
    raise QuestionError(f"question {question.id!r} cannot be scored: " + "; ".join(attempts))


def option_token_ids(question: Question, tokenizer, trailing_space: bool = False) -> list[int]:
    """One vocabulary id per option: the first token of the option as it appears after the prompt.

    With `trailing_space`, the prompt already ends in a space and the option is tokenised bare; without it, the option
    carries the leading space. Options whose first token collides are refused rather than silently scored as ties.
    """
    ids: list[int] = []
    seen: dict[int, str] = {}
    for option in question.options:
        text = option if trailing_space else " " + option
        pieces = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not pieces:
            raise QuestionError(f"question {question.id!r}: option {option!r} produces no tokens")
        if len(pieces) > 1:
            raise QuestionError(
                f"question {question.id!r}: option {option!r} is {len(pieces)} tokens as {text!r}. "
                f"This read-out scores a single token, so use a shorter name."
            )
        token = pieces[0]
        if token in seen:
            raise QuestionError(
                f"question {question.id!r}: options {option!r} and {seen[token]!r} start with the same token as "
                f"{text!r}, so they cannot be told apart."
            )
        seen[token] = option
        ids.append(token)
    return ids


def load_unembedding(model_name: str, hidden_size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Fetch the output embedding matrix by name from the shard that holds it.

    Instantiating a language-model wrapper to reach its head would materialise a second copy of every parameter to keep
    one matrix, which on a large mixture-of-experts exhausts the device. Matched by suffix because checkpoints in the
    same family prefix the key differently; the output projection is preferred and the input embedding is the fallback
    for a checkpoint that ties them.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors import safe_open

    def pick(keys) -> str | None:
        for suffix in ("lm_head.weight", "embed_tokens.weight"):
            hits = sorted(k for k in keys if k.endswith(suffix))
            if hits:
                return min(hits, key=len)  # `lm_head.weight` over `something.lm_head.weight`
        return None

    try:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    except EntryNotFoundError:
        # A checkpoint small enough to fit in one file has no index, so every key is in that one file.
        weight_map = None

    if weight_map is None:
        path = hf_hub_download(model_name, "model.safetensors")
    else:
        key = pick(weight_map)
        if key is None:
            raise RuntimeError(f"{model_name} has no lm_head or embed_tokens in its index")
        path = hf_hub_download(model_name, weight_map[key])

    # Opened rather than loaded: reading the file whole would materialise every tensor in the shard on the host to keep
    # one of them, which on a large mixture-of-experts is tens of gigabytes for a single matrix.
    with safe_open(path, framework="pt") as shard:
        present = set(shard.keys())
        key = pick(present)
        if key is None:
            raise RuntimeError(f"{model_name} has no lm_head or embed_tokens")
        matrix = shard.get_tensor(key)
        scale_key = next(
            (k for k in (f"{key}_scale_inv", key.replace("weight", "weight_scale_inv")) if k in present), None
        )
        scale = shard.get_tensor(scale_key) if scale_key else None

    if matrix.shape[1] != hidden_size:
        raise RuntimeError(f"{key} is {tuple(matrix.shape)}, which does not match a hidden size of {hidden_size}")
    if matrix.dtype in FP8:
        if scale is None:
            raise RuntimeError(
                f"{key} is {matrix.dtype} and no scale was found beside it. Casting it directly would drop the scales, "
                f"and because each option reads a different row of this matrix, a dropped per-row scale reorders the "
                f"options rather than blurring them -- a wrong answer with nothing to show it."
            )
        matrix = _dequantise(matrix, scale)
    return matrix.to(device=device, dtype=dtype)


FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


#: Block sizes a checkpoint plausibly used, most common first. The size has to be recovered rather than read, because
#: what is stored beside the weight is the number of blocks, and a count does not determine a size.
BLOCK_SIZES = (128, 256, 64, 32, 512)


def _block_size(dimension: int, blocks: int) -> int:
    """Recover the block size from how many blocks cover a dimension.

    Not `ceil(dimension / blocks)`. That is the *smallest* size that would need this many blocks, and it is wrong
    whenever the last block is partial: 200 rows in 2 blocks of 128 gives 100, which then hands rows 100 to 127 the
    second block's multiplier instead of the first's. Wrong by a scale factor, on a fifth of the rows, silently.
    """
    for size in BLOCK_SIZES:
        if -(-dimension // size) == blocks:
            return size
    if dimension % blocks == 0:
        return dimension // blocks
    raise RuntimeError(
        f"cannot tell what block size puts {dimension} values into {blocks} blocks; the scales beside this weight do "
        f"not match any block size this reader knows"
    )


def _dequantise(matrix: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Undo block quantisation. `scale` holds one multiplier per block of the matrix, so it is expanded back to shape.

    `_scale_inv` is the multiplier that restores a stored value, not its reciprocal -- the same convention the borrowed
    matrix-multiply kernels use, and the one thing here that is easy to get backwards.
    """
    wide = matrix.float()
    if scale.dim() == 0 or scale.numel() == 1:
        return wide * scale.float()
    if scale.dim() == 1:  # one multiplier per row
        return wide * scale.float().reshape(-1, 1)
    down = _block_size(matrix.shape[0], scale.shape[0])
    across = _block_size(matrix.shape[1], scale.shape[1])
    expanded = scale.float().repeat_interleave(down, dim=0).repeat_interleave(across, dim=1)
    return wide * expanded[: matrix.shape[0], : matrix.shape[1]]


def logits_for(hidden: torch.Tensor, unembedding: torch.Tensor, token_ids: list[list[int]]) -> list[torch.Tensor]:
    """Per question, the raw score of each declared option. No softmax, so the numbers can still be shifted.

    Separated from `score` because a prior has to be subtracted before the softmax, not after: after it, the correction
    is a reweighting of something already normalised and no longer removes a bias.
    """
    if hidden.shape[0] != len(token_ids):
        raise ValueError(f"{hidden.shape[0]} branch rows against {len(token_ids)} questions")
    return [(hidden[row : row + 1].float() @ unembedding[ids].float().t())[0] for row, ids in enumerate(token_ids)]


def score(
    hidden: torch.Tensor,
    unembedding: torch.Tensor,
    token_ids: list[list[int]],
    priors: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Per question, a probability over its declared options.

    `hidden` is (branches, hidden_size), one row per question, taken at that branch's final position. The softmax is
    over each question's own options, which is why the result is a list rather than a tensor: questions declare
    different numbers of options.

    With `priors`, each question's scores have its prior subtracted first. See `calibration` for what that is and what
    it is worth. The short version: some option tokens are likelier than others before the context is read at all, and
    subtracting that leaves what the context contributed.
    """
    scores = logits_for(hidden, unembedding, token_ids)
    if priors is not None:
        if len(priors) != len(scores):
            raise ValueError(f"{len(priors)} priors against {len(scores)} questions")
        scores = [row - prior.to(row.device, row.dtype) for row, prior in zip(scores, priors, strict=True)]
    return [torch.softmax(row, dim=-1) for row in scores]
