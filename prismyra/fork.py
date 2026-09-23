"""The fork: read a context once, then answer each question as a row of one batch.

The whole saving is here. Asking Q questions the obvious way reads the context Q times; this reads it once and gives
each question a row carrying only its own tokens. What makes that safe is one asymmetry: the context's keys and values
are only read by the branches, so one write can serve every row, while the recurrence state and each row's own tokens
are written, so those must be copied per row.

Being read-only is what makes one write *correct* for every row, and `ForkLayer` keeps the context in one row rather
than replicating it across the group -- `cache.cache_bytes` reports what that holds. A group uses only as many rows as
it has questions, because the read joins the context with each row's own tokens and that join is per row. See
docs/FORK.md.
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

    Its own type because the alternative is silent corruption. The cache is allocated for `WIDTHS[-1]` extra tokens,
    so a wider suffix writes past the end of a device buffer, and on CUDA that is asynchronous: the process carries on
    and fails later somewhere unrelated, or answers.
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
    #: The bucket its cache was allocated for, which is how `Context.close` knows where to return it.
    room: int | None = None
    position_from: int = 0
    snapshot: dict | None = field(default=None, repr=False)


def snapshot(cache) -> dict:
    """Clone the batch-of-one state so it can be forked more than once.

    Forking and running the suffixes both mutate the cache in place, and the recurrent layers write their advanced
    state back regardless of what the caller asked for. Without this, a second group of branches would start from a
    state the first group had already advanced -- and would answer plausibly.
    """
    snap: dict = {}
    for i, layer in enumerate(cache.layers):
        entry: dict = {}
        for attr in ("recurrent_states", "conv_states"):
            d = getattr(layer, attr, None)
            if isinstance(d, dict):
                entry[attr] = {k: (None if v is None else v.clone()) for k, v in d.items()}
        if not _holds_attention(layer):
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if torch.is_tensor(t):
                    entry[attr] = t.clone()
        entry[LENGTHS] = _lengths(layer)
        snap[i] = entry
    return snap


def _holds_attention(layer) -> bool:
    """Whether this layer stores keys and values rather than a recurrent state.

    Asked by a class attribute rather than by guessing from the presence of `max_cache_len`. The guess worked until
    the attention layer's storage changed shape, at which point these two functions would have started cloning buffers
    that are not per-row -- silently, and with the wrong answers arriving a group later.
    """
    return bool(getattr(layer, "holds_attention", False))


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


#: Where a layer's full-width state buffers are kept, and the views of them handed to the framework.
#:
#: The framework allocates this state at whatever batch it first saw -- one row, during the context pass -- and a fork
#: then needs it at the group's width. Reallocating per width is what it used to do, and that is why a cache could not
#: be reused: `reset()` clears the contents and keeps the batch dimension, so a cache returned by a three-row pass holds
#: three-row state and the next context's fork tries to widen three rows to thirty-two.
#:
#: Owned here instead. One buffer per layer and key, allocated once at the full width, with a cached view per row count.
#: The addresses never move, which is what lets a recorded pass outlive the context it was taken on, and the cached
#: views mean the tensor identity the framework holds is stable for a given width -- a recording checks that identity.
OWNED = "_prismyra_state"


def _owned(layer, attr: str, key, like: torch.Tensor, width: int, rows: int) -> torch.Tensor:
    """The full-width buffer for this piece of state, and the view of its first `rows` rows.

    `like` is the snapshot's one-row tensor, which gives the shape of everything but the batch. Allocated on first use
    and never again: a caller may hold the view, and a recorded pass holds the address.
    """
    store = getattr(layer, OWNED, None)
    if store is None:
        store = {}
        setattr(layer, OWNED, store)
    slot = store.get((attr, key))
    full = slot["full"] if slot else None
    want = (width, *like.shape[1:])
    if not torch.is_tensor(full) or tuple(full.shape) != want or full.dtype != like.dtype:
        full = torch.empty(want, dtype=like.dtype, device=like.device)
        slot = {"full": full, "views": {}}
        store[(attr, key)] = slot
    assert slot is not None
    view = slot["views"].get(rows)
    if view is None:
        view = full[:rows]
        slot["views"][rows] = view
    return view


def pick(snap: dict, row: int) -> dict:
    """One document's slice of a snapshot taken over a batched read.

    A batched read is one pass over several documents, so the recurrence returns **one state per document** and the
    snapshot of it has a row each. A view rather than a clone: the snapshot is already a clone and nothing writes
    through these, so copying again would double a gigabyte for no reason.
    """
    out: dict = {}
    for i, entry in snap.items():
        taken: dict = {}
        for attr in ("recurrent_states", "conv_states"):
            if attr in entry:
                taken[attr] = {k: (None if v is None else v[row : row + 1]) for k, v in entry[attr].items()}
        for attr in ("keys", "values"):
            if attr in entry:
                taken[attr] = entry[attr][row : row + 1]
        if LENGTHS in entry:
            taken[LENGTHS] = entry[LENGTHS]
        out[i] = taken
    return out


def restore_and_fork_many(
    cache, parts: list[tuple[dict, int]], width: int | None = None, rows_for: list[int] | None = None
) -> None:
    """Fork several documents into one batch: each document's state into the rows answering about it.

    This is the engine side of a mixed batch, and it is short for a reason worth stating. The recurrent state was
    already per row and already **copied** from a snapshot rather than broadcast, because each row advances its own over
    its own tokens -- so a row carrying a different document's state is not a new mechanism, it is the same copy with a
    different source. What could not be done was the attention side, and that is why the pages exist.

    `parts` is each document's snapshot with the number of rows it gets, in row order. `rows_for` names the document
    each row belongs to and is handed to the attention layers, which need it to build the page table.
    """
    if not parts:
        raise ValueError("a fork needs at least one document")
    rows = sum(count for _, count in parts)
    first = parts[0][0]
    for i, layer in enumerate(cache.layers):
        shape = first.get(i, {})
        # The lengths of the first document. For the attention layers this is overwritten immediately by
        # `begin_branches`, which sets a length per row from the pool; for the recurrent layers there is one counter,
        # and every document in a batch has been read to its own end, so any of them says the same thing about where
        # the next tokens go.
        _restore_lengths(layer, shape.get(LENGTHS, {}))
        begin = getattr(layer, "begin_branches", None)
        if begin is not None:
            if rows_for is not None and _takes_rows(begin):
                begin(rows_for)
            else:
                begin()
        for attr in ("recurrent_states", "conv_states"):
            if attr not in shape:
                continue
            bound = getattr(layer, attr)
            for key, example in shape[attr].items():
                if example is None:
                    bound[key] = None
                    continue
                held = _owned(layer, attr, key, example, width or rows, rows)
                _fill_per_document(held, [(snap[i][attr][key], count) for snap, count in parts])
                bound[key] = held
        if _holds_attention(layer):
            continue
        for attr in ("keys", "values"):
            if attr not in shape:
                continue
            held = _owned(layer, attr, attr, shape[attr], width or rows, rows)
            _fill_per_document(held, [(snap[i][attr], count) for snap, count in parts])
            setattr(layer, attr, held)


def _fill_per_document(held: torch.Tensor, pieces: list[tuple[torch.Tensor, int]]) -> None:
    """Write each document's state into its own run of rows.

    One buffer, filled in row order, because the rows of a batch are contiguous per document by construction -- the
    engine assigns them that way so that a row's page range and its state come from the same arithmetic.
    """
    at = 0
    for state, count in pieces:
        held[at : at + count].copy_(state.expand((count, *state.shape[1:])) if state.shape[0] == 1 else state[:count])
        at += count
    if at != held.shape[0]:
        raise ValueError(f"{at} rows were filled and this buffer holds {held.shape[0]}")


def _takes_rows(begin) -> bool:
    """Whether a layer's `begin_branches` accepts the row assignment. The recurrent layers do not have one at all and
    the joined attention layer's takes nothing, so this is asked rather than assumed."""
    import inspect

    try:
        return bool(inspect.signature(begin).parameters)
    except (TypeError, ValueError):  # pragma: no cover - a builtin with no introspectable signature
        return False


def restore_and_fork(cache, snap: dict, rows: int, width: int | None = None) -> None:
    """Put the one-row state back, then widen it to `rows`.

    The two kinds of state widen differently, and that difference is the design:

    * recurrent and convolution state is **copied**, because each row advances its own over its own tokens;
    * keys and values are broadcast from one row to all of them, because every row reads the same context. They are then
      materialised per row rather than left as a view of the snapshot: a layer that writes its own keys in place would
      otherwise write through every row at once, and through the snapshot every later group reads.

    Each layer's own count of the tokens it holds is restored too, and that is not a detail. Restoring the tensors
    while leaving the count where the last group left it gives a layer that holds the context and believes it holds
    more; the next group then advances from the wrong offset, and the failure is a wrong answer rather than an error.
    """
    for i, layer in enumerate(cache.layers):
        entry = snap.get(i, {})
        _restore_lengths(layer, entry.get(LENGTHS, {}))
        # A group is starting. The attention layers are told rather than left to guess, because a row count cannot
        # distinguish a context write from a branch write when the group is one.
        begin = getattr(layer, "begin_branches", None)
        if begin is not None:
            begin()
        for attr in ("recurrent_states", "conv_states"):
            if attr not in entry:
                continue
            d = getattr(layer, attr)
            for k, v in entry[attr].items():
                if v is None:
                    d[k] = None
                    continue
                # Into a buffer this package owns at the full width, with a cached view of the rows this group needs.
                # See `OWNED`: the alternative reallocated whenever the row count changed, which moved the addresses a
                # recorded pass had written down and left a reused cache holding the previous group's batch dimension.
                held = _owned(layer, attr, k, v, width or rows, rows)
                held.copy_(v.expand((rows, *v.shape[1:])) if v.shape[0] == 1 else v[:rows])
                d[k] = held
        if _holds_attention(layer):
            # This layer keeps the context in one row and each branch's own tokens in another, and joins them on read.
            # There is nothing to fan out and nothing to restore beyond the length, which was done above.
            continue
        for attr in ("keys", "values"):
            if attr not in entry:
                continue
            t = entry[attr]
            # Materialised, not a view of the snapshot. A layer that writes its own keys in place would otherwise
            # write through every row at once and through the snapshot, making every later group wrong rather than
            # failing.
            held = _owned(layer, attr, attr, t, width or rows, rows)
            held.copy_(t.expand((rows, *t.shape[1:])))
            setattr(layer, attr, held)


def build_suffixes(
    texts: list[str], tokenizer, device: str, rows: int | None, width: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise each branch's text and right-pad to a common width. Returns ids, positions and read positions.

    Takes the rendered text rather than the question, because which of two renderings a question needs is decided by
    the read-out -- see `readout.plan` -- and the branch has to read the one that was planned for it. Rendering again
    here could disagree with what the token ids were chosen for, and a disagreement there scores the wrong position.

    Padding a branch is safe in a way padding the context is not: a pad token after a branch's own tokens is never
    read by the read-out and cannot be attended to by a real position, whereas one added to the context passes through
    the recurrence where there is nothing to mask it out of.
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
