"""The fork: read a context once, then answer each question as a row of one batch.

The whole saving is here. Asking Q questions the obvious way reads the context Q times; this reads it once and gives
each question a row carrying only its own tokens. What makes that safe is one asymmetry: the context's keys and values
are only read by the branches, so one write can serve every row, while the recurrence state and each row's own tokens
are written, so those must be copied per row.

Being read-only is what makes one write *correct* for every row; it is not what makes it cheap. `ForkLayer` is
preallocated at the full batch, so the context ends up physically replicated across the group -- the memory cost
`cache.cache_bytes` reports. See docs/FORK.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers.cache_utils import Cache

#: Branch widths a suffix is padded up to. Pinning the width keeps the batch shape out of a small set, which makes a
#: row's answer independent of its companions and keeps kernel compilations off the request path.
WIDTHS = (32, 64, 96, 128, 192, 256, 384, 512)


class TooWide(ValueError):
    """A question does not fit the widest branch.

    Its own type because the alternative is silent corruption. The cache is allocated for `WIDTHS[-1]` extra tokens, so
    a wider suffix writes past the end of a device buffer, and on CUDA that is asynchronous: the process carries on and
    fails later somewhere unrelated, or answers.
    """


def round_width(width: int) -> int:
    for w in WIDTHS:
        if width <= w:
            return w
    raise TooWide(
        f"a question needs {width} tokens and the widest branch is {WIDTHS[-1]}; shorten the prompt or declare "
        f"fewer options"
    )


@dataclass
class Prefill:
    """A context that has been read. Branches fork from this."""

    cache: Cache
    tokens: int
    last_position: torch.Tensor
    #: Where a branch's positions start. Equal to `tokens` for text; larger when the context held images or video,
    #: whose three-axis positions advance by a grid rather than by a token count. See `media.position_offset`.
    position_from: int = 0
    snapshot: dict | None = field(default=None, repr=False)


def snapshot(cache) -> dict:
    """Clone the batch-of-one state so it can be forked more than once.

    Forking and running the suffixes both mutate the cache in place, and the recurrent layers write their advanced state
    back regardless of what the caller asked for. Without this, a second group of branches would start from a state the
    first group had already advanced -- and would answer plausibly.
    """
    snap: dict = {}
    for i, layer in enumerate(cache.layers):
        entry: dict = {}
        for attr in ("recurrent_states", "conv_states"):
            d = getattr(layer, attr, None)
            if isinstance(d, dict):
                entry[attr] = {k: (None if v is None else v.clone()) for k, v in d.items()}
        if getattr(layer, "max_cache_len", None) is None:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if torch.is_tensor(t):
                    entry[attr] = t.clone()
        entry[LENGTHS] = _lengths(layer)
        snap[i] = entry
    return snap


#: Where a layer's own record of how many tokens it holds is kept in a snapshot. A private key rather than an
#: attribute name, so it cannot collide with one.
LENGTHS = "__lengths__"

#: Attributes a cache layer may use to count the tokens it holds. Restoring the state tensors without these leaves a
#: layer holding the context but claiming it holds the context plus the last group's suffix, which does not raise: the
#: next group advances from the wrong offset and answers plausibly.
LENGTH_ATTRS = ("cumulative_length", "_host_length", "_seen_tokens", "seq_length")


def _lengths(layer) -> dict:
    out: dict = {}
    for attr in LENGTH_ATTRS:
        value = getattr(layer, attr, None)
        if torch.is_tensor(value):
            out[attr] = value.clone()
        elif isinstance(value, int):
            out[attr] = value
    return out


def _restore_lengths(layer, lengths: dict) -> None:
    for attr, value in lengths.items():
        current = getattr(layer, attr, None)
        if torch.is_tensor(current) and torch.is_tensor(value):
            # In place, so a reference taken to this tensor stays valid.
            current.copy_(value)
        else:
            setattr(layer, attr, value)


def restore_and_fork(cache, snap: dict, rows: int) -> None:
    """Put the one-row state back, then widen it to `rows`.

    The two kinds of state widen differently, and that difference is the design:

    * recurrent and convolution state is **copied**, because each row advances its own over its own tokens;
    * keys and values are broadcast from one row to all of them, because every row reads the same context. They are then
      materialised per row rather than left as a view of the snapshot: a layer that writes its own keys in place would
      otherwise write through every row at once, and through the snapshot every later group reads.

    Each layer's own count of the tokens it holds is restored too, and that is not a detail. Restoring the tensors while
    leaving the count where the last group left it gives a layer that holds the context and believes it holds more; the
    next group then advances from the wrong offset, and the failure is a wrong answer rather than an error.
    """
    for i, layer in enumerate(cache.layers):
        entry = snap.get(i, {})
        _restore_lengths(layer, entry.get(LENGTHS, {}))
        for attr in ("recurrent_states", "conv_states"):
            if attr not in entry:
                continue
            d = getattr(layer, attr)
            for k, v in entry[attr].items():
                if v is None:
                    d[k] = None
                    continue
                want = (rows, *v.shape[1:])
                cur = d.get(k)
                if torch.is_tensor(cur) and tuple(cur.shape) == want:
                    # Written in place rather than replaced: reallocating on every group would put allocator work on
                    # the request path, and would invalidate any reference already taken to the old buffer.
                    cur.copy_(v.expand(want) if v.shape[0] == 1 else v)
                else:
                    d[k] = (v.expand(want) if v.shape[0] == 1 else v).contiguous()
        if getattr(layer, "max_cache_len", None) is not None:
            # Preallocated: the context is already in every row, so there is nothing to fan out.
            continue
        for attr in ("keys", "values"):
            if attr not in entry:
                continue
            t = entry[attr]
            # Materialised, not a view of the snapshot. A layer that writes its own keys in place would otherwise write
            # through every row at once and through the snapshot, making every later group wrong rather than failing.
            setattr(layer, attr, t.expand(rows, *t.shape[1:]).contiguous())


def build_suffixes(
    texts: list[str], tokenizer, device: str, rows: int | None, width: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise each branch's text and right-pad to a common width. Returns ids, positions and read positions.

    Takes the rendered text rather than the question, because which of two renderings a question needs is decided by the
    read-out -- see `readout.plan` -- and the branch has to read the one that was planned for it. Rendering again here
    could disagree with what the token ids were chosen for, and a disagreement there scores the wrong position.

    Padding a branch is safe in a way padding the context is not: a pad token after a branch's own tokens is never read
    by the read-out and cannot be attended to by a real position, whereas one added to the context passes through the
    recurrence where there is nothing to mask it out of.
    """
    pieces = [tokenizer("\n" + text, add_special_tokens=False)["input_ids"] for text in texts]
    real = len(pieces)
    longest = max(len(p) for p in pieces)
    target = width or round_width(longest)
    if longest > target:
        raise ValueError(f"a question needs {longest} tokens and the width is pinned to {target}")
    n_rows = rows or real
    if real > n_rows:
        raise ValueError(f"{real} branches will not fit in a batch pinned to {n_rows} rows")

    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    ids = torch.full((n_rows, target), pad, dtype=torch.long, device=device)
    read_at = torch.zeros(n_rows, dtype=torch.long, device=device)
    for r, piece in enumerate(pieces):
        ids[r, : len(piece)] = torch.tensor(piece, device=device)
        read_at[r] = len(piece) - 1
    for r in range(real, n_rows):
        # Padded rows repeat the first question so they compute something well-formed; their answers are discarded.
        ids[r] = ids[0]
        read_at[r] = read_at[0]
    return ids, read_at, torch.tensor(real, device=device)
