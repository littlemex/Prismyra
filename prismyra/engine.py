"""The engine: load a model, read a context once, answer typed questions about it.

`open_context` is the primitive and `ask` is sugar over it. The distinction matters: a follow-up against an open
context costs a branch, while calling `ask` again re-reads the context. A library that only offered `ask` would hide the
thing it exists to provide.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from . import kernels
from .cache import build_cache, cache_bytes
from .fork import (
    WIDTHS,
    Prefill,
    TooWide,
    build_suffixes,
    restore_and_fork,
    round_width,
    snapshot,
)
from .readout import load_unembedding, plan, score
from .schema import (
    Answer,
    PrismyraError,
    Question,
    Request,
    Result,
    Timing,
)

#: Questions per traversal. A traversal costs the same carrying one row or this many, so a group is the unit of
#: cost and the default fills it. Measured figures are in docs/PERFORMANCE.md.
GROUP = 32


@dataclass
class Context:
    """A context that has been read. Ask as many questions as you like; the reading is not repeated.

    `context_ms` is the reading, reported here rather than inside each `Result`. Charging it to every ask would make
    the timings of several asks sum to more than the wall clock, and the sum is exactly what a caller adds up.
    """

    _engine: Prismyra
    _prefill: Prefill | None
    tokens: int
    context_ms: float

    def ask(self, questions: list[Question]) -> Result:
        if self._prefill is None:
            raise PrismyraError("this context has been closed")
        self._engine.validate(questions)
        return self._engine._answer(self._prefill, questions, self.tokens, context_ms=0.0)

    def close(self) -> None:
        """Release the cache. Worth doing explicitly, because it is measured in tens of gigabytes.

        Every branch holds its own copy of the context's keys and values, so an open context costs the context's
        key-value cache times the group -- about 3.4 GiB at 5,000 tokens with the default group of 32, and 12.5 GiB at
        20,000. `Prismyra.cache_bytes` reports the figure for a given length.
        """
        self._prefill = None

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class Prismyra:
    """Read one context, answer many typed questions about it.

    Prefill only: nothing here generates text. That is what makes the fork possible and what makes a single question
    slower than a general serving engine would be -- see the scope note in the README.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        fast_kernels: bool = True,
        require_kernels: bool = False,
        group: int = GROUP,
    ):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if group < 1:
            raise PrismyraError(f"group must be at least one question, not {group}")
        self.model_name = model
        # Normalised through `torch.device`, so that "cuda:1" is recognised as a CUDA device. Comparing the string to
        # "cuda" would send a second card down the CPU path: float32, no kernels, and a load that ignores the index.
        default = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(device) if device else torch.device(default)
        self.device = str(self.torch_device)
        on_cuda = self.torch_device.type == "cuda"
        self.dtype = dtype or (torch.bfloat16 if on_cuda else torch.float32)
        self.group = group
        # The caches are held by the engine and mutated in place, so two threads asking at once would interleave one
        # another's branches. The lock makes that safe; `prismyra.queue.Worker` is still what makes it fast.
        self._lock = threading.RLock()

        if require_kernels and not (fast_kernels and on_cuda):
            raise PrismyraError(
                f"require_kernels was asked for with fast_kernels={fast_kernels} on {self.device}; the borrowed "
                f"kernels exist only on CUDA, so this configuration can never satisfy it"
            )

        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # No language-model head: it projects to the whole vocabulary and nothing here generates a token.
        self.backbone = AutoModel.from_pretrained(model, dtype=self.dtype, device_map=self.device if on_cuda else None)
        self.backbone.eval()
        if not on_cuda:
            self.backbone.to(self.torch_device)

        decoder = getattr(self.config, "text_config", self.config)
        self.hidden_size = decoder.hidden_size
        self.applied = (
            kernels.apply(self.backbone, self.config, required=require_kernels)
            if fast_kernels and on_cuda
            else kernels.Applied(adapter="none", skipped=[f"not applied on {self.device}"])
        )
        self.unembedding = load_unembedding(model, self.hidden_size, self.device, self.dtype)

    # ------------------------------------------------------------------ public
    def validate(self, questions: list[Question]) -> None:
        """Refuse a question that cannot be scored, before any context is read.

        Two of the read-out's refusals need the tokenizer -- an option that is more than one token with a leading space,
        and two options sharing a first token -- so they cannot happen when the question is constructed. Running them
        here keeps them off the device: a request refused after the context pass has already spent the expensive half.
        """
        if not questions:
            raise PrismyraError("ask needs at least one question")
        ids = [q.id for q in questions]
        if len(set(ids)) != len(ids):
            # Answers are keyed by id, so a repeat would overwrite one and the caller would see fewer answers than
            # questions with nothing saying which was lost.
            raise PrismyraError(f"duplicate question ids: {ids}")
        for q in questions:
            chosen = plan(q, self.tokenizer)
            rendered = self.tokenizer("\n" + chosen.text, add_special_tokens=False)["input_ids"]
            if len(rendered) > WIDTHS[-1]:
                raise PrismyraError(
                    f"question {q.id!r} renders to {len(rendered)} tokens and the widest branch is {WIDTHS[-1]}"
                )

    def cache_bytes(self, context_tokens: int) -> int:
        """The device memory one open context of this length will hold.

        Exposed because the number is large: the fan-out that lets branches be written independently means every row
        carries its own copy of the context, so this grows with the group as well as with the context.
        """
        return cache_bytes(self.config, context_tokens + WIDTHS[-1], self.group, self.dtype)

    def open_context(self, context: str) -> Context:
        """Read a context and keep it open. The expensive half happens here, once.

        The returned context holds device memory until it is closed -- `Context.close`, or a `with` block. See
        `cache_bytes`.
        """
        if not context.strip():
            # The same refusal `Request` makes. Without it the direct path answers a question about nothing, and the
            # answer looks like an answer.
            raise PrismyraError("a context cannot be empty")
        ids = self.tokenizer(context, add_special_tokens=True)["input_ids"]
        self._check_fits(len(ids))
        with self._lock:
            start = _now(self.torch_device)
            with torch.inference_mode():
                prefill = self._read(ids)
            return Context(_engine=self, _prefill=prefill, tokens=len(ids), context_ms=_since(start, self.torch_device))

    def ask(self, context: str, questions: list[Question]) -> Result:
        """Read a context and answer questions about it. Sugar for `open_context(...).ask(...)`."""
        self.validate(questions)
        with self._lock, self.open_context(context) as opened:
            assert opened._prefill is not None
            answered = self._answer(opened._prefill, questions, opened.tokens, context_ms=opened.context_ms)
        return answered

    def ask_many(self, requests: list[Request]) -> list[Result | PrismyraError]:
        """Answer several independent requests. One request failing does not fail the others.

        Errors are returned in place rather than raised, so a caller reading results by index always finds something
        there. Nothing is shared between requests: different contexts share no state.

        The original exception is kept as the returned error's cause, so a slot that failed for a reason worth
        escalating -- the device out of memory, a framework mismatch -- can still be told from a malformed question.
        """
        out: list[Result | PrismyraError] = []
        for request in requests:
            try:
                out.append(self.ask(request.context, list(request.questions)))
            except PrismyraError as e:
                out.append(e)
            except Exception as e:  # noqa: BLE001 - surfaced per slot rather than taking the batch down
                wrapped = PrismyraError(f"{type(e).__name__}: {e}")
                wrapped.__cause__ = e
                out.append(wrapped)
        return out

    def stats(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "group": self.group,
            "kernels": self.applied.as_dict(),
        }

    # ------------------------------------------------------------------ internals
    def _check_fits(self, context_tokens: int) -> None:
        """Refuse a context that cannot fit, by name, before the allocator refuses it by address.

        An out-of-memory error from inside a framework allocation says how many bytes it wanted and nothing about which
        knob to turn. This says the context length and the group, which are the two knobs.
        """
        if self.torch_device.type != "cuda":
            return
        wanted = self.cache_bytes(context_tokens)
        free, total = torch.cuda.mem_get_info(self.torch_device)
        if wanted >= free:
            raise PrismyraError(
                f"a context of {context_tokens} tokens needs {wanted / 1024**3:.1f} GiB of key-value cache at "
                f"group={self.group}, and {free / 1024**3:.1f} GiB of {total / 1024**3:.1f} GiB is free. Every branch "
                f"holds its own copy of the context, so halving the group halves this; a shorter context does too."
            )

    def _read(self, ids: list[int]) -> Prefill:
        x = torch.tensor([ids], device=self.device)
        # Room for the context plus the widest branch, since the same cache carries both.
        cache = build_cache(self.config, len(ids) + WIDTHS[-1], self.group, self.dtype, self.device)
        self.backbone(input_ids=x, use_cache=True, past_key_values=cache)
        return Prefill(cache=cache, tokens=len(ids), last_position=torch.tensor([len(ids) - 1], device=self.device))

    def _answer(self, prefill: Prefill, questions: list[Question], tokens: int, context_ms: float) -> Result:
        # Already validated: both public entry points call `validate` before the context is read, and repeating it
        # here would tokenise every question a second time on the request path.
        # The batch width is the one the cache was allocated for. It is not a per-call option: the cache is preallocated
        # at construction time and a write of any other row count is refused, which is the point of preallocating.
        rows = self.group
        plans = [plan(q, self.tokenizer) for q in questions]
        token_ids = [p.token_ids for p in plans]
        widest = max(len(self.tokenizer("\n" + p.text, add_special_tokens=False)["input_ids"]) for p in plans)
        try:
            width = round_width(widest)
        except TooWide as e:
            raise PrismyraError(str(e)) from e

        start = _now(self.torch_device)
        probabilities: list[torch.Tensor] = []
        with self._lock, torch.inference_mode():
            for lo in range(0, len(questions), rows):
                chunk = [p.text for p in plans[lo : lo + rows]]
                hidden = self._branch(prefill, chunk, rows, width)
                probabilities.extend(score(hidden, self.unembedding, token_ids[lo : lo + rows]))
        readout_ms = _since(start, self.torch_device)

        answers = {}
        for q, p in zip(questions, probabilities, strict=True):
            values = p.tolist()
            best = max(range(len(values)), key=lambda i: values[i])
            option = q.options[best]
            answers[q.id] = Answer(
                id=q.id,
                kind=q.kind,
                value=q.value_of(option),
                option=option,
                probabilities=dict(zip(q.options, values, strict=True)),
            )
        return Result(
            answers=answers,
            model=self.model_name,
            context_tokens=tokens,
            timing=Timing(context_ms=context_ms, readout_ms=readout_ms),
        )

    def _branch(self, prefill: Prefill, texts: list[str], rows: int, width: int) -> torch.Tensor:
        # The snapshot is taken on the first branch, when the cache holds exactly the context, so restoring it also puts
        # every layer's token count back to the end of the context. One mechanism, not a state restore plus a separate
        # rewind: two of them can disagree, and the one that is wrong answers plausibly.
        if prefill.snapshot is None:
            prefill.snapshot = snapshot(prefill.cache)
        restore_and_fork(prefill.cache, prefill.snapshot, rows)

        ids, read_at, _ = build_suffixes(texts, self.tokenizer, self.device, rows, width)
        positions = torch.arange(prefill.tokens, prefill.tokens + ids.shape[1], device=self.device).expand(rows, -1)
        out = self.backbone(input_ids=ids, position_ids=positions, use_cache=True, past_key_values=prefill.cache)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        # Each row is read at its own last real token, which is why the padding cannot reach an answer.
        return hidden[torch.arange(rows, device=self.device), read_at][: len(texts)]


def _now(device: torch.device) -> float:
    """The clock, after the device has caught up. Synchronising the *named* device matters: a second card's work would
    otherwise be timed against whatever the default card happened to be doing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _since(start: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e3
