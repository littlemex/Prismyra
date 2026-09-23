"""Several documents read in one forward pass, by telling the kernels where one document ends and the next begins.

Reading is almost entirely fixed cost, and that is what makes this worth building. Measured on the supported model, one
document read on its own:

| tokens | read |
|---|---|
| 289 | 113.2 ms |
| 545 | 113.2 ms |
| 1,057 | 116.0 ms |
| 2,081 | 132.5 ms |

which fits **110.1 ms of fixed cost plus 10.8 ms per thousand tokens**. At 289 tokens, 97% of reading a document is
paying for kernel launches rather than for the document. So eight documents of that size read one at a time cost 906 ms
and read together should cost about 135 ms -- the launches once, the tokens eight times.

The way to get there is the way a serving engine already does it: **concatenate the documents into one flat sequence and
carry the boundaries as data.** Nothing may attend across a boundary, and three kernels in this model have to be told
where they are. All three take an argument for it, which is the reason this is possible at all rather than a rewrite:

* the attention kernel takes `cu_seqlens_q` and `cu_seqlens_k`;
* the borrowed gated-delta-rule kernel takes `cu_seqlens`, and with it returns **one final state per sequence** --
  exactly the per-document state a fork needs;
* the convolution takes the starts already. It is this package's own Triton kernel, written with `seq_starts` for the
  framework's packed-sequence path, and a batched read is the same thing arriving from somewhere else. The framework's
  own convolution has no such argument, so when the borrowed kernel is not installed a batched read is refused rather
  than silently convolving across a boundary.

**The boundaries are ambient, and that is a deliberate and uncomfortable choice.** The kernels are reached through the
framework's own module-level names, so there is no argument to thread from the engine down to them. The alternative is
patching the model's forward, which is a larger surface and a worse one. What makes it safe enough is that the window is
a context manager, so it cannot be left open, and that the wrappers treat "no boundaries" as the only default: a pass
that forgets to open the window reads one document, which is the behaviour that was already there.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Boundaries:
    """Where each document starts and ends inside one flat run of tokens.

    `offsets` is the cumulative form every one of these kernels wants: `n + 1` entries, the first zero and the last the
    total. `lengths` is kept beside it because the cumulative form cannot be read back without a device copy, and
    something has to know how long each document was without stalling the pass to find out.
    """

    offsets: torch.Tensor
    lengths: tuple[int, ...]
    #: One entry per convolution that ran during the pass, in the order they ran: that layer's input tail per document.
    #:
    #: Recorded because the framework builds its convolution state by slicing the end of the pass's input, and the end
    #: of a flat run of documents is the end of the **last** one. Measured: two documents and a state of shape
    #: (1, 8192, 4) where two rows were needed, so every row of the first document would have started its branch
    #: convolution from the second document's tail. Nothing raises; the answers are about a window that was never there.
    conv_tails: list = field(default_factory=list)

    @property
    def documents(self) -> int:
        return len(self.lengths)

    @property
    def tokens(self) -> int:
        return sum(self.lengths)

    @property
    def longest(self) -> int:
        return max(self.lengths)

    def tails(self, tokens_major: torch.Tensor, width: int, extra: int = 0) -> torch.Tensor:
        """Each document's last `width` tokens of a flat run, padded at the front when a document is shorter.

        `tokens_major` is `(total_tokens, channels)` and the result is `(documents, channels, width)`, the layout
        the framework's own convolution state uses. Padded at the front because that is where a convolution's window
        would have nothing: a document of two tokens has no third-from-last.
        """
        out = tokens_major.new_zeros((self.documents, tokens_major.shape[1], width))
        # Past whatever the framework prepended. The prefix is the first document's left context and is not part of it.
        at = extra
        for d, n in enumerate(self.lengths):
            take = min(width, n)
            piece = tokens_major[at + n - take : at + n]
            out[d, :, width - take :] = piece.t()
            at += n
        return out

    def with_prefix(self, extra: int, device: str | torch.device) -> torch.Tensor:
        """Cumulative offsets for a run that carries `extra` leading tokens belonging to the first document.

        The framework's linear-attention layer prepends the convolution state it was holding, convolves, and then drops
        the prefix again. On a cache that has never held anything there is nothing to prepend and the run is exactly the
        documents; on a cache that already holds documents -- a shelf -- the run is `kernel - 1` zeros longer.

        Those leading tokens are the first document's left context, so its start stays at zero and every later boundary
        moves by `extra`. Getting this wrong does not raise: the second document's convolution would begin three tokens
        inside the first one.
        """
        moved = [0, *[int(x) + extra for x in self.offsets[1:].tolist()]]
        return torch.tensor(moved, dtype=torch.int32, device=device)

    def positions(self, device: str | torch.device) -> torch.Tensor:
        """One row of positions for the flat run, restarting at zero for each document.

        A single `arange` over the whole run would give the second document positions continuing from the first, so it
        would be read as a continuation of it. Nothing raises; the answers are simply about a document that was never
        written.
        """
        return torch.cat([torch.arange(n, device=device) for n in self.lengths]).unsqueeze(0)


#: The boundaries the pass in progress covers, or None when a pass covers one document. Module-level because the kernels
#: are reached through the framework's own module-level names and there is no argument to thread down to them.
_current: Boundaries | None = None


def current() -> Boundaries | None:
    """The boundaries of the pass in progress, if it carries more than one document."""
    return _current


@contextmanager
def reading(lengths: list[int], device: str | torch.device):
    """Declare that the pass inside this block carries these documents, in this order.

    A context manager rather than a pair of calls, because the failure from leaving it set is a later pass reading its
    single document as though it were several -- which does not raise.
    """
    global _current  # noqa: PLW0603 - see the module docstring: there is no argument to thread down to the kernels
    if not lengths:
        raise ValueError("a batched read needs at least one document")
    if any(n <= 0 for n in lengths):
        raise ValueError(f"every document needs at least one token: {lengths}")
    if _current is not None:
        raise RuntimeError("a batched read is already in progress; these do not nest")
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    _current = Boundaries(offsets=offsets, lengths=tuple(lengths))
    try:
        yield _current
    finally:
        _current = None
