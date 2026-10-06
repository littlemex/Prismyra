"""Option-set heads: a small learned read-out, used in place of the output embedding for the questions whose declared
options are exactly a registered list.

Prismyra's read-out scores each option with the model's own output embedding (docs/READOUT.md). For a fixed question
that is asked again and again -- the same options, often the same prompt -- a head fitted to the hidden state that
read-out sees can be much more accurate than the embedding rows of the option words. A head is **data**: a spec file
names the option list it answers, its form and its weights. Nothing here knows about any task.

Which questions a head answers is decided by one rule: the question declares the same set of options as the head.
Order does not matter, for the same reason it does not matter to the read-out: the rendered question lists its options
sorted (docs/READOUT.md), so two declarations of one set reach the same hidden state, and the head's probabilities are
returned in the question's own order. A question with any other set of options -- every yes/no question, and every
lettered one unless a head names exactly those letters -- is read exactly as without heads: same tensors, same
arithmetic, bit-identical probabilities. The backbone is never touched, so a head cannot move the context read either.

Spec (a JSON list, paths relative to the spec file):

    [{"name": "ticket-priority-v1", "options": ["low", "normal", "urgent"], "form": "mlp",
      "weights": "ticket-priority-v1.safetensors"}]

Weights, in the space of the final hidden state (the engine's hidden size, before any normalisation of your own):

    linear: W (k, hidden), b (k,)                         logits = h W^T + b
    mlp:    W1 (m, hidden), b1 (m,), W2 (k, m), b2 (k,)   logits = gelu(h W1^T + b1) W2^T + b2

k is the number of options, and row i of the output scores the head's i-th declared option. The head's softmax over
its k logits is the answer's probabilities.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .schema import PrismyraError

FORMS = {"linear": ("W", "b"), "mlp": ("W1", "b1", "W2", "b2")}


class Head:
    """One registered head. Computed in float32 whatever the engine's dtype, as the output-embedding read-out is."""

    def __init__(
        self, name: str, options: Sequence[str], form: str, weights: dict[str, torch.Tensor], hidden_size: int, device
    ):
        if form not in FORMS:
            raise PrismyraError(f"head {name!r}: form must be one of {sorted(FORMS)}, not {form!r}")
        missing = [k for k in FORMS[form] if k not in weights]
        if missing:
            need, gone = ", ".join(FORMS[form]), ", ".join(missing)
            raise PrismyraError(f"head {name!r}: {form} weights need {need}; missing {gone}")
        self.name = name
        self.options = tuple(options)
        self.form = form
        self.w = {k: weights[k].to(device=device, dtype=torch.float32) for k in FORMS[form]}
        first, last = (self.w["W"], self.w["W"]) if form == "linear" else (self.w["W1"], self.w["W2"])
        if first.shape[1] != hidden_size:
            raise PrismyraError(f"head {name!r} reads a hidden state of {first.shape[1]}, the model's is {hidden_size}")
        if last.shape[0] != len(self.options):
            raise PrismyraError(f"head {name!r} scores {last.shape[0]} options but declares {len(self.options)}")

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.float()
        if self.form == "linear":
            return x @ self.w["W"].t() + self.w["b"]
        z = torch.nn.functional.gelu(x @ self.w["W1"].t() + self.w["b1"])
        return z @ self.w["W2"].t() + self.w["b2"]


class Heads:
    """The registered heads, by option list. Empty -- and a no-op everywhere -- when no spec is given."""

    def __init__(self, spec: str | Path | list | None, hidden_size: int, device):
        self.by_options: dict[frozenset[str], Head] = {}
        if spec is None:
            return
        if isinstance(spec, (str, Path)):
            base = Path(spec).parent
            entries = json.loads(Path(spec).read_text())
        else:
            base, entries = Path("."), spec
        from safetensors.torch import load_file

        for entry in entries:
            path = Path(entry["weights"])
            head = Head(
                entry["name"],
                entry["options"],
                entry["form"],
                load_file(str(path if path.is_absolute() else base / path)),
                hidden_size,
                device,
            )
            if len(set(head.options)) != len(head.options):
                raise PrismyraError(f"head {head.name!r} declares an option twice")
            key = frozenset(head.options)
            if key in self.by_options:
                raise PrismyraError(f"heads {self.by_options[key].name!r} and {head.name!r} answer the same options")
            self.by_options[key] = head

    def __bool__(self) -> bool:
        return bool(self.by_options)

    def _find(self, options: Sequence[str]) -> Head | None:
        # A question cannot declare an option twice (schema), so the set loses nothing; the length check keeps a
        # head from answering a question whose options merely include its own.
        head = self.by_options.get(frozenset(options))
        return head if head is not None and len(head.options) == len(options) else None

    def name_for(self, options: Sequence[str]) -> str | None:
        """The head that answers a question with these options, or None when the output embedding does."""
        head = self._find(options)
        return head.name if head is not None else None

    def apply(
        self, hidden: torch.Tensor, options: list[Sequence[str]], probabilities: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Replace the rows a head answers. Every other row is returned as the very tensor it came in as."""
        if not self.by_options:
            return probabilities
        out = list(probabilities)
        for row, opts in enumerate(options):
            head = self._find(opts)
            if head is not None:
                p = torch.softmax(head.logits(hidden[row : row + 1])[0], dim=-1)
                out[row] = p[[head.options.index(o) for o in opts]]  # into the question's declared order
        return out


class Recorder:
    """Stands in for the registered heads while recording the hidden states a head for `options` would read.

    The hidden states are taken at exactly the point a head is applied, so they are the features a head fitted on them
    will see when served. Answers are left as they are: nothing is replaced, every probability is the engine's own.
    """

    def __init__(self, options: Sequence[str], inner: Heads):
        self.key = frozenset(options)
        self.size = len(options)
        self.inner = inner
        self.rows: list[torch.Tensor] = []

    def __bool__(self) -> bool:
        return True

    def name_for(self, options: Sequence[str]) -> str | None:
        return self.inner.name_for(options)

    def apply(
        self, hidden: torch.Tensor, options: list[Sequence[str]], probabilities: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        for row, opts in enumerate(options):
            if frozenset(opts) == self.key and len(opts) == self.size:
                self.rows.append(hidden[row].detach().float().cpu())
        return self.inner.apply(hidden, options, probabilities)


class record_hidden:
    """`with record_hidden(engine, options) as rec: engine.ask(...)` -- then `torch.stack(rec.rows)` is one row per
    question asked with that option set, in the order they were answered. For fitting a head; see docs/HEADS.md."""

    def __init__(self, engine, options: Sequence[str]):
        self.engine, self.options = engine, options

    def __enter__(self) -> Recorder:
        self.saved = self.engine.heads
        self.engine.heads = Recorder(self.options, self.saved)
        return self.engine.heads

    def __exit__(self, *exc) -> None:
        self.engine.heads = self.saved
