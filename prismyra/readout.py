"""Turning a branch's final hidden state into a probability over the declared options.

The model's own output embedding does the work: each option's single token is scored, and those scores are softmaxed
over the declared options only. Nothing is trained. See `schema.SCORING` for what that means and does not mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .schema import Question, QuestionError


def option_token_ids(question: Question, tokenizer) -> list[int]:
    """One vocabulary id per option: the first token of the option's text **with a leading space**.

    The leading space matters because that is how the token appears after "Answer:". Checking the bare text instead
    would accept a name that is one token alone and two with a space in front, and then score the wrong token.

    Options whose first token collides are refused rather than silently scored as ties, and an option that needs more
    than one token is refused rather than truncated -- a truncated option is answerable and wrong.
    """
    ids: list[int] = []
    seen: dict[int, str] = {}
    for option in question.options:
        pieces = tokenizer(" " + option, add_special_tokens=False)["input_ids"]
        if not pieces:
            raise QuestionError(f"question {question.id!r}: option {option!r} produces no tokens")
        if len(pieces) > 1:
            raise QuestionError(
                f"question {question.id!r}: option {option!r} is {len(pieces)} tokens with a leading space. "
                f"This read-out scores a single token, so use a shorter name."
            )
        token = pieces[0]
        if token in seen:
            raise QuestionError(
                f"question {question.id!r}: options {option!r} and {seen[token]!r} start with the same token, "
                f"so they cannot be told apart."
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


def score(hidden: torch.Tensor, unembedding: torch.Tensor, token_ids: list[list[int]]) -> list[torch.Tensor]:
    """Per question, a probability over its declared options.

    `hidden` is (branches, hidden_size), one row per question, taken at that branch's final position. The softmax is
    over each question's own options, which is why the result is a list rather than a tensor: questions declare
    different numbers of options.
    """
    if hidden.shape[0] != len(token_ids):
        raise ValueError(f"{hidden.shape[0]} branch rows against {len(token_ids)} questions")
    out = []
    for row, ids in enumerate(token_ids):
        columns = unembedding[ids]  # (options, hidden)
        logits = (hidden[row : row + 1].float() @ columns.float().t())[0]
        out.append(torch.softmax(logits, dim=-1))
    return out
