"""The engine: load a model, read a context once, answer typed questions about it.

`open_context` is the primitive and `ask` is sugar over it. The distinction matters: a follow-up against an open
context costs a branch, while calling `ask` again re-reads the context. A library that only offered `ask` would hide
the thing it exists to provide.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from . import kernels, varlen
from .cache import build_cache, cache_bytes, join_bytes_per_token
from .calibration import Calibration
from .fork import (
    WIDTHS,
    Prefill,
    TooWide,
    build_suffixes,
    pick,
    restore_and_fork,
    restore_and_fork_many,
    round_width,
    snapshot,
)
from .graphs import keeping_pays, pays_from, record
from .media import encode, position_offset
from .readout import load_unembedding, plan, score
from .schema import (
    Answer,
    PrismyraError,
    Question,
    Request,
    Result,
    Timing,
)

#: The widest group of questions one traversal may carry. A group of questions is answered in one pass and a request
#: with more than this is answered in several; a request with fewer uses only as many rows as it has questions, which
#: is not the same as it once was -- see docs/PERFORMANCE.md for the measurement that changed it.
GROUP = 32

#: How much more than the model predicts a branch pass is budgeted at. An allocator's peak is blocks rounded up and
#: reused rather than a sum of tensor sizes, and the prediction is two terms answering to a handful of measurements.
#: One number in one place, so there is one thing to argue with.
ANSWERING_MARGIN = 1.15

#: How far a recording's first replay may sit from the pass it was taken from before the recording is thrown away. Tight
#: on purpose: a replay runs the same kernels on the same addresses, so the honest expectation is zero and anything else
#: is a symptom. Not zero exactly, because the comparison is of bf16 hidden states and an exact test would reject a
#: recording for a rounding step that cannot reach an answer.
REPLAY_TOLERANCE = 1e-3

#: Replays run before a recording is trusted. Two, because the first one is the easy case: it happens with the device in
#: exactly the state the recording was taken in, and a recording that diverges does so from the second replay onwards.
REPLAY_CHECKS = 2

#: Context lengths a cache is allocated for. A context of 900 tokens and one of 1,000 both get the 1,024 allocation, and
#: that is the point: an allocation shared between contexts keeps its addresses, and addresses are what a recording
#: holds. Without this every context built its own cache and freed it on close, so a recording could never outlive the
#: document it was taken on -- which gave a session asking many groups the 3.5x and a request asking one group nothing.
#:
#: Powers of two, because the waste is bounded at half the allocation and the alternative is a tuned ladder that would
#: have to be retuned per model. The largest is what `open_context` will refuse above, and refusing by name is what this
#: engine already does when a context will not fit.
CONTEXT_SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


@dataclass
class Batch:
    """Several documents read into one cache, answered in one forward pass.

    `context_ms` is the reading of all of them, reported once. The documents were read one at a time -- reading is one
    pass over one document and there is nothing to share there -- and what this object exists for is that **answering**
    them is one pass.
    """

    _engine: Prismyra
    _prefills: list[Prefill] | None
    _room: int | None
    context_ms: float

    @property
    def tokens(self) -> list[int]:
        if self._prefills is None:
            raise PrismyraError("this batch has been closed")
        return [p.tokens for p in self._prefills]

    def ask(self, questions: list[list[Question]]) -> list[Result]:
        """One list of questions per document, in the order the documents were given. One pass answers all of them."""
        if self._prefills is None:
            raise PrismyraError("this batch has been closed")
        if len(questions) != len(self._prefills):
            raise PrismyraError(
                f"{len(questions)} lists of questions for {len(self._prefills)} documents; a batch answers every "
                f"document it holds, so pass one list each"
            )
        return self._engine._answer_batch(self._prefills, questions, context_ms=self.context_ms)

    def close(self) -> None:
        if self._prefills is not None:
            self._engine._release_cache(self._prefills[0])
            self._prefills = None

    def __enter__(self) -> Batch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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

        The context's keys and values are held once and each branch holds only its own short run of tokens, so an open
        context costs about 0.41 GiB at 5,000 tokens with the default group of 32 and 0.69 GiB at 20,000 --
        `Prismyra.cache_bytes` reports the figure for a given length. It used to be 3.4 and 12.5 GiB, when every branch
        held its own copy.
        """
        if self._prefill is not None:
            self._engine._release_cache(self._prefill)
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
        calibrate: bool = False,
        graphs: bool = False,
        paged: bool = False,
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
        #: Whether to record a branch pass and replay it. Off by default, and the reason is memory rather than doubt: a
        #: recording holds a private allocator pool, and this engine refuses a context by name from a budget it
        #: measures, so a feature that quietly takes device memory behind that budget would make the refusal wrong.
        #: Measured worth: a recorded pass replays in 28.9 ms against 107.7 eagerly, and recording costs 151.3 ms once.
        self.graphs = graphs
        #: Whether the attention read goes through a page table. Off by default. Its point is not memory -- that was
        #: measured at zero -- but that the read's shape stops depending on the context's length, so one recorded graph
        #: serves every context instead of one per open context. See `prismyra/paged.py`.
        self.paged = paged
        if paged:
            from .paged import PagedUnavailable, kernel_supports_pages

            usable, why = kernel_supports_pages()
            if not (usable and fast_kernels and on_cuda):
                raise PagedUnavailable(
                    f"paged storage was asked for and cannot be provided: {why}"
                    if not usable
                    else f"paged storage needs the borrowed kernels on CUDA, and this is fast_kernels={fast_kernels} "
                    f"on {self.device}"
                )
        #: Shapes a recording was attempted on and refused, with the reason. Attempted once per shape, not once per
        #: group, and reported through `stats()` rather than retried in silence.
        self.declined_recordings: dict = {}
        #: What each accepted recording's replays disagreed with their own pass by. Reported so that "it was accepted"
        #: and "it agreed" are separate statements: a check that silently measures the wrong thing reads as the second.
        self.verified_recordings: dict = {}
        #: How many times each recording has answered. Reported because "a recording was taken" and "a recording was
        #: used" are different claims, and a test that asserts the first while meaning the second is the mistake this
        #: package keeps finding: with the old rule every recording was taken on a shape's second use and there was no
        #: third use, so the count here would have been zero everywhere.
        self.replays: dict = {}
        #: What a pass at each shape cost eagerly and replayed, as measured. The recording decision is made from these
        #: rather than from a constant, because the ratio between them runs from 0.684 at a suffix of sixteen tokens to
        #: 0.997 at 128 -- see `_worth_keeping`.
        self.replay_cost: dict = {}
        self._made_caches = 0
        #: Recordings by the identity of the cache they were taken on. Keyed by `id` because a cache is not hashable and
        #: because identity is exactly the right test: a recording is valid for one allocation and no other.
        self._cache_recordings: dict[int, dict] = {}
        #: The largest per-row transient seen with the context-proportional part taken out -- activations and kernel
        #: workspace for one row -- and the widest pass it was seen at. None until the first pass.
        #:
        #: The width is kept because per-row cost is **not** flat in width at the bottom of the range: at 24,327
        #: context tokens a pass cost 0.111 GiB per row at width 1 and 0.155 at widths 8 and 32. So an observation
        #: taken at width 1 under-states a wide pass by 29%, and admission must not claim to have budgeted one.
        self._observed_row_constant: int | None = None
        self._observed_at_rows = 0
        #: The largest transient seen while **reading** a context, per context token, above the cache it leaves behind.
        #: A separate number because it is the larger of the two on this model -- 4.78 GiB against 4.97 for a full-width
        #: branch pass at 24,327 tokens, and 0.60 against 2.53 at 3,040 -- and because budgeting only the answering
        #: transient admits a context whose *read* runs out of memory before any question is asked.
        #:
        #: Proportional to the length with no constant worth keeping: measured at 1.96e-4 GiB per token at both 3,040
        #: and 24,327 tokens, and independent of the group, which is what makes one observation usable at any length.
        self._observed_reading_per_token: int | None = None

        if require_kernels and not (fast_kernels and on_cuda):
            raise PrismyraError(
                f"require_kernels was asked for with fast_kernels={fast_kernels} on {self.device}; the borrowed "
                f"kernels exist only on CUDA, so this configuration can never satisfy it"
            )

        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # A processor only if the checkpoint has one. It is what turns an image into pixels and expands the
        # placeholder into as many pad tokens as the resolution needs; a text-only checkpoint has none and does not
        # need one.
        self.processor = _load_processor(model)
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
        #: Whether the borrowed attention kernel is in use, which decides whether the join is read in the layout it is
        #: stored in or handed to the framework's own attention as a strided view. That changes what a branch pass
        #: transiently allocates, so it changes what admission budgets.
        self._borrowed_kernel = self.applied.ok and not self.applied.skipped
        self.unembedding = load_unembedding(model, self.hidden_size, self.device, self.dtype)
        # Off unless asked for. It is a change to what a probability means, and whether it is an improvement is a
        # measured question rather than an obvious one -- `evals/run.py` compares the two.
        self.calibration = Calibration() if calibrate else None

    # ------------------------------------------------------------------ public
    def validate(self, questions: list[Question]) -> None:
        """Refuse a question that cannot be scored, before any context is read.

        Two of the read-out's refusals need the tokenizer -- an option that is more than one token with a leading
        space, and two options sharing a first token -- so they cannot happen when the question is constructed.
        Running them here keeps them off the device: a request refused after the context pass has already spent the
        expensive half.
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

    def room_for(self, context_tokens: int) -> int:
        """The bucket a context of this length is allocated at, which is not the same as its length.

        Exposed because every figure about held memory has to use it. Measuring the held cache at the context's own
        length while allocating at the bucket made the difference look like answering cost: a 51-token context in a
        1,024-token allocation put nearly a gigabyte per row into the observed constant, which admission then multiplied
        by the group and refused 24 GiB of work that needed one.
        """
        room = next((size for size in CONTEXT_SIZES if context_tokens <= size), None)
        if room is None:
            raise PrismyraError(
                f"a context of {context_tokens} tokens is longer than this engine allocates for ({CONTEXT_SIZES[-1]})"
            )
        return room

    def cache_bytes(self, context_tokens: int) -> int:
        """The device memory one open context of this length **holds** between questions.

        Not what answering one costs. The context is held once, so this grows with the context and only slightly with
        the group -- but a branch pass allocates several times this much transiently, and that peak is what decides how
        many contexts can be answered at once. `answering_bytes` is that number, and `stats()` reports both.
        """
        return cache_bytes(self.config, self.room_for(context_tokens) + WIDTHS[-1], self.group, self.dtype, WIDTHS[-1])

    def answering_bytes(self, context_tokens: int, questions: int | None = None) -> int:
        """What a branch pass will transiently allocate on top of the held cache.

        Two parts, because the measured shape has two parts. Per row, the transient is a constant plus a term in the
        context length -- 0.080 GiB per row at 3,040 context tokens and 0.155 GiB at 24,327 on the supported model:

        * the term in the length is the join of context and branch for each layer's read, and it is **arithmetic**:
          `cache.join_bytes_per_token` computes it from the config, and the figure it gives is within 7% of the
          measured slope;
        * the constant is one row's activations and whatever the kernels want as workspace, and that is **observed**,
          because deriving it would mean encoding one model's shapes into this package.

        Splitting it that way is not tidiness. Keeping the largest observed *prediction* instead would ratchet towards
        refusing work: a prediction scaled from a short context is larger per token than one from a long context, so
        the shortest context ever seen would win and be kept forever, and a full-width pass at 24,327 tokens would be
        budgeted at 17.7 GiB when it needs 4.97 -- refused with four gigabytes idle. The constant is context-free by
        construction, so ratcheting it cannot do that.

        Zero until a pass has happened, which is honest rather than convenient: admission cannot budget the first one,
        and the refusal message says so.
        """
        if self._observed_row_constant is None:
            return 0
        rows = min(self.group, questions) if questions else self.group
        per_token = join_bytes_per_token(self.config, self.dtype, doubled=not self._borrowed_kernel)
        per_row = self._observed_row_constant + per_token * (context_tokens + WIDTHS[-1])
        # A margin, because an allocator's peak is blocks rounded up and reused, not a sum of tensor sizes. One place,
        # so there is one number to argue with.
        return int(rows * per_row * ANSWERING_MARGIN)

    def reading_bytes(self, context_tokens: int) -> int:
        """What reading a context of this length transiently allocates, above the cache it leaves behind.

        Observed and scaled, with no constant term, because that is what was measured: 0.602 GiB of transient at 3,040
        context tokens and 4.775 GiB at 24,327, which is 1.96e-4 GiB per token both times and the same at every group
        width. A single observation therefore predicts any length, unlike the answering transient.

        This is the larger of the two phases on the supported model, and it is the one a caller cannot make smaller by
        asking fewer questions. Extrapolated, a context somewhere near 48,000 tokens needs more of it than a 44 GiB
        card has left after the weights -- which is a limit worth refusing by name rather than discovering.
        """
        if self._observed_reading_per_token is None:
            return 0
        return int(self._observed_reading_per_token * context_tokens * ANSWERING_MARGIN)

    def budget_is_evidenced(self, questions: int | None = None) -> bool:
        """Whether `answering_bytes` is an estimate this engine has seen a pass wide enough to support.

        False for a pass wider than any yet observed. It is a separate question from the estimate's value because the
        two have different consequences: a low estimate for a pass no wider than one already measured is a reason to
        refuse, and the same number for a wider pass is not evidence of anything and must not be used to refuse.
        Admission still checks it -- an estimate that already exceeds free memory is a refusal either way -- but the
        pass that goes ahead unbudgeted is caught by the allocator, which is why that error names the knobs.
        """
        rows = min(self.group, questions) if questions else self.group
        return self._observed_row_constant is not None and rows <= self._observed_at_rows

    def open_context(self, context: str, *, images: list | None = None, videos: list | None = None) -> Context:
        """Read a context and keep it open. The expensive half happens here, once.

        `images` and `videos` take anything the model's processor accepts -- a `PIL.Image`, a path, an array of frames
        -- and are read into the context alongside the text. This is where the design pays best: a frame costs the
        vision tower once and then behaves like any other context token, so the questions after it are nearly free.

        The returned context holds device memory until it is closed -- `Context.close`, or a `with` block. See
        `cache_bytes`.
        """
        if not context.strip() and not images and not videos:
            # The same refusal `Request` makes. Without it the direct path answers a question about nothing, and the
            # answer looks like an answer. Media on its own is a context, so only the empty-handed case is refused.
            raise PrismyraError("a context cannot be empty")
        encoded = encode(context, images, videos, self.processor, self.tokenizer, self.device)
        self._check_fits(encoded.tokens)
        with self._lock:
            start = _now(self.torch_device)
            before = self._peak_baseline()
            try:
                with torch.inference_mode():
                    prefill = self._read(encoded)
            except torch.OutOfMemoryError as e:
                raise PrismyraError(
                    f"ran out of memory reading a context of {encoded.tokens} tokens. This is the read rather than a "
                    f"question, so asking fewer questions will not help and a shorter context is the only knob; the "
                    f"transient this needs grows with the length and is larger than answering costs."
                ) from e
            self._observe_reading(before, encoded.tokens)
            return Context(
                _engine=self,
                _prefill=prefill,
                tokens=encoded.tokens,
                context_ms=_since(start, self.torch_device),
            )

    def ask(
        self,
        context: str,
        questions: list[Question],
        *,
        images: list | None = None,
        videos: list | None = None,
    ) -> Result:
        """Read a context and answer questions about it. Sugar for `open_context(...).ask(...)`."""
        self.validate(questions)
        with self._lock, self.open_context(context, images=images, videos=videos) as opened:
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
            "scoring": self.calibration.mode if self.calibration else "raw",
            "storage": "paged" if self.paged else "joined",
            # Counted by the layers themselves rather than taken from the flag. A previous version of the paged path
            # reported itself as installed and never ran, and this is the number that would have said so.
            "paged_reads_served": self._paged_reads(),
            "graphs": self.graphs,
            "graphs_declined": dict(self.declined_recordings),
            "graphs_verified": dict(self.verified_recordings),
            "graphs_replays": dict(self.replays),
            "graphs_cost": dict(self.replay_cost),
            "caches_allocated": self._made_caches,
            # Measured on this engine rather than derived, and zero until a question has been answered. Reported
            # because it is the number that decides how many contexts can be answered at once, and it is several times
            # the held cache.
            "answering_row_constant_bytes": self._observed_row_constant or 0,
            "answering_bytes_per_context_token_per_row": join_bytes_per_token(
                self.config, self.dtype, doubled=not self._borrowed_kernel
            ),
            "answering_observed_at_rows": self._observed_at_rows,
            "reading_bytes_per_context_token": self._observed_reading_per_token or 0,
        }

    # ------------------------------------------------------------------ internals
    def _check_fits(self, context_tokens: int) -> None:
        """Refuse a context that cannot fit, by name, before the allocator refuses it by address.

        An out-of-memory error from inside a framework allocation says how many bytes it wanted and nothing about
        which knob to turn. This says the context length and the group, which are the two knobs.
        """
        if self.torch_device.type != "cuda":
            return
        held = self.cache_bytes(context_tokens)
        # What the work costs as well as what holding costs. Budgeting only the cache admitted contexts that fitted
        # idle and ran out of memory on their first question: at 24,327 tokens the cache is 0.87 GiB and a full-width
        # pass peaks at 4.97 GiB, so the check was short by a factor of six.
        #
        # The larger of the two phases rather than their sum, because they do not overlap: the read's transient is
        # freed before any question is asked. Reading is the larger of them on the supported model, and it is the one a
        # caller cannot shrink by asking fewer questions.
        answering = self.answering_bytes(context_tokens)
        reading = self.reading_bytes(context_tokens)
        wanted = held + max(answering, reading)
        free, total = torch.cuda.mem_get_info(self.torch_device)
        # The device's free memory is not what is available. The allocator keeps a pool it has already taken from the
        # device and can hand out without asking again, and loading these weights leaves that pool large -- so asking
        # the device alone refused an 18,000-token context that had 9 GiB waiting for it inside the process.
        spare = torch.cuda.memory_reserved(self.torch_device) - torch.cuda.memory_allocated(self.torch_device)
        free += max(0, spare)
        if wanted >= free:
            phase = "reading it" if reading >= answering else f"answering at group={self.group}"
            work = max(answering, reading)
            if not work:
                budget = (
                    f"{held / 1024**3:.1f} GiB to hold at group={self.group}, and what the work adds is not yet known "
                    f"because nothing has been read or answered on this engine"
                )
            elif reading >= answering or self.budget_is_evidenced():
                budget = f"{held / 1024**3:.1f} GiB to hold and {work / 1024**3:.1f} GiB for {phase}"
            else:
                budget = (
                    f"{held / 1024**3:.1f} GiB to hold and at least {work / 1024**3:.1f} GiB for {phase} -- at least, "
                    f"because the widest pass measured on this engine was {self._observed_at_rows} rows and a wider "
                    f"one costs more per row"
                )
            # The advice has to follow whichever phase bound, or it is wrong half the time: a smaller group shrinks a
            # branch pass and does nothing at all to a read, and a read is the larger of the two on this model.
            knobs = (
                "only a shorter context reduces this: reading is what does not fit, and it costs the same whatever is "
                "asked afterwards"
                if reading >= answering
                else "a shorter context reduces both parts; a smaller group, or asking fewer questions at a time, "
                "reduces the second"
            )
            raise PrismyraError(
                f"a context of {context_tokens} tokens needs {budget}, and {free / 1024**3:.1f} GiB of "
                f"{total / 1024**3:.1f} GiB is available. The context is held once, so {knobs}."
            )

    def _read(self, encoded) -> Prefill:
        cache, room = self._claim_cache(encoded.tokens)
        self.backbone(input_ids=encoded.input_ids, use_cache=True, past_key_values=cache, **encoded.media)
        # Read after the forward, not before: the offset is something the model works out while reading the context.
        position_from = position_offset(self.backbone, encoded.tokens) if encoded.has_media else encoded.tokens
        # The fork's state buffers are allocated here rather than on the first branch pass, and the reason is the budget
        # rather than tidiness. `fork.OWNED` allocates that state once at the full group width -- about a gigabyte on
        # this model -- and doing it inside the first pass put the whole allocation into the peak that pass was measured
        # by. Divided by that pass's row count and multiplied by the group, a one-off gigabyte became a 24 GiB answering
        # estimate and admission refused contexts of fifty tokens. It is held memory, so it is allocated while the
        # context is being read and counted as held.
        taken = snapshot(cache)
        restore_and_fork(cache, taken, self.group, width=self.group)
        return Prefill(
            snapshot=taken,
            cache=cache,
            room=room,
            tokens=encoded.tokens,
            last_position=torch.tensor([encoded.tokens - 1], device=self.device),
            position_from=position_from,
        )

    def open_batch(self, contexts: list[str]) -> Batch:
        """Read several documents into one cache, so that one forward pass can answer about all of them.

        This is the answer to the measurement that says one card serves 1.08 requests a second however many arrive: the
        batch's width was being spent entirely on questions about one document, so a second caller waited. Here the
        width is divided between documents, which is what a serving engine does with its own batch and what this
        package could not do while the storage joined one context with every row.

        Requires the paged storage -- `Prismyra(paged=True)` -- and says so rather than quietly serving one document,
        because the joined storage puts the context in a row that every row reads and there is no arrangement of it that
        holds two.

        The documents are read one at a time, which is unchanged: reading is one pass over one document and there is
        nothing to share. What is shared is the **answering**.
        """
        if not contexts:
            raise PrismyraError("a batch needs at least one document")
        if not self.paged:
            raise PrismyraError(
                "a batch of documents needs the paged storage, because the joined storage keeps the context in one row "
                "that every row reads. Build the engine with Prismyra(..., paged=True)."
            )
        if len(contexts) > self.group:
            raise PrismyraError(
                f"{len(contexts)} documents and a group of {self.group}: every document needs at least one row. Build "
                f"the engine with a larger group, or read them in batches of {self.group}."
            )

        # The two replacements a batched read depends on, named rather than "all of them". The first version of this
        # check asked whether anything had been skipped at all, and a skipped head duplication -- nothing to do with
        # document boundaries -- refused every batch.
        needed = {"convolution", "gated_delta_rule"}
        installed = {swap.name for swap in self.applied.swaps}
        if missing := needed - installed:
            raise PrismyraError(
                f"a batch of documents needs the borrowed {' and '.join(sorted(missing))}: the framework's own has no "
                "argument for where one document ends, so a batched read would scan across the boundary and answer "
                f"about a document that was never written. What the adapter reported: "
                f"{self.applied.skipped or self.applied.summary()}."
            )
        encoded = [encode(text, None, None, self.processor, self.tokenizer, self.device) for text in contexts]
        if any(one.has_media for one in encoded):
            raise PrismyraError(
                "a batch of documents is text only for now: media widen a context's positions by a grid rather than by "
                "a token count, and a flat run of several would need each document's own offset threaded through."
            )
        lengths = [one.tokens for one in encoded]
        total = sum(lengths)
        cache, room = self._claim_cache(total)
        started = _now(self.torch_device)
        with torch.inference_mode():
            # One pass over all of them. Reading is 110 ms of fixed cost plus 11 ms per thousand tokens on this
            # model, so what this removes is that fixed cost paid per document rather than per batch.
            ids = torch.cat([one.input_ids for one in encoded], dim=1)
            with varlen.reading(lengths, self.device) as boundaries:
                self.backbone(
                    input_ids=ids,
                    position_ids=boundaries.positions(self.device),
                    use_cache=True,
                    past_key_values=cache,
                )
                _put_back_conv_states(cache, boundaries)
                self._check_batched_read(cache, boundaries)
            # One snapshot with a row per document, because the recurrence returns a state per document when it is told
            # the boundaries. `fork.pick` is how a document takes its own row of it.
            taken = snapshot(cache)
        prefills = [
            Prefill(
                snapshot=pick(taken, handle),
                cache=cache,
                room=room if handle == 0 else None,
                tokens=one.tokens,
                last_position=torch.tensor([one.tokens - 1], device=self.device),
                position_from=one.tokens,
            )
            for handle, one in enumerate(encoded)
        ]
        return Batch(_engine=self, _prefills=prefills, _room=room, context_ms=_since(started, self.torch_device))

    def _check_batched_read(self, cache, boundaries) -> None:
        """That the pass left one recurrent state and one convolution window per document, not one per batch.

        Asked rather than assumed, and it is the check that found the defect: the recurrence returns a state per
        document once it is told the boundaries, and the convolution does not -- the framework slices its state from the
        end of the pass, and the end of a flat run is the end of the last document. A row of the first document would
        then have started its branch convolution from the second document's tail, which answers plausibly.
        """
        want = boundaries.documents
        for n, layer in enumerate(cache.layers):
            for attr in ("recurrent_states", "conv_states"):
                held = getattr(layer, attr, None)
                if not isinstance(held, dict):
                    continue
                for key, state in held.items():
                    if state is None or state.shape[0] == want:
                        continue
                    raise PrismyraError(
                        f"a batched read of {want} documents left layer {n}'s {attr}[{key}] with "
                        f"{state.shape[0]} rows. Every document needs its own, or rows answering about one would "
                        f"continue from another."
                    )

    def _answer_batch(self, prefills: list[Prefill], asked: list[list[Question]], context_ms: float) -> list[Result]:
        """One forward pass carrying questions about several documents, one row per question.

        Rows are laid out document by document, contiguously, because a row's page range and a row's state come from two
        different pieces of arithmetic and laying them out the same way is what keeps those two in agreement.
        """
        for questions in asked:
            self.validate(questions)
        counts = [len(q) for q in asked]
        if sum(counts) > self.group:
            raise PrismyraError(
                f"{sum(counts)} questions across {len(asked)} documents and a group of {self.group}. A batch answers "
                f"in one pass, so its questions have to fit one group."
            )
        if any(count == 0 for count in counts):
            raise PrismyraError("every document in a batch needs at least one question; drop it from the batch instead")

        flat = [q for questions in asked for q in questions]
        plans = [plan(q, self.tokenizer) for q in flat]
        width = self._width_for(plans)
        rows_for = [handle for handle, count in enumerate(counts) for _ in range(count)]

        start = _now(self.torch_device)
        with self._lock, torch.inference_mode():
            hidden = self._branch_across(prefills, counts, rows_for, [p.text for p in plans], width)
            probabilities = score(hidden, self.unembedding, [p.token_ids for p in plans], None)
        readout_ms = _since(start, self.torch_device)

        results, at = [], 0
        for handle, questions in enumerate(asked):
            answers = {}
            for q, values in zip(questions, probabilities[at : at + len(questions)], strict=True):
                numbers = values.tolist()
                best = max(range(len(numbers)), key=lambda i: numbers[i])
                option = q.options[best]
                answers[q.id] = Answer(
                    id=q.id,
                    kind=q.kind,
                    value=q.value_of(option),
                    option=option,
                    probabilities=dict(zip(q.options, numbers, strict=True)),
                )
            at += len(questions)
            results.append(
                Result(
                    answers=answers,
                    model=self.model_name,
                    context_tokens=prefills[handle].tokens,
                    scoring="raw",
                    timing=Timing(context_ms=context_ms, readout_ms=readout_ms),
                )
            )
        return results

    def _branch_across(self, prefills, counts: list[int], rows_for: list[int], texts: list[str], width: int):
        """The pass itself. Every row's positions start at its own document's end, which is per row not per batch."""
        rows = sum(counts)
        ids, read_at, _ = build_suffixes(texts, self.tokenizer, self.device, rows, width)
        starts = [prefills[handle].position_from or prefills[handle].tokens for handle in rows_for]
        offsets = torch.tensor(starts, device=self.device).unsqueeze(1)
        positions = offsets + torch.arange(ids.shape[1], device=self.device).unsqueeze(0)

        restore_and_fork_many(
            prefills[0].cache,
            [(prefills[handle].snapshot, count) for handle, count in enumerate(counts)],
            width=self.group,
            rows_for=rows_for,
        )
        out = self.backbone(input_ids=ids, position_ids=positions, use_cache=True, past_key_values=prefills[0].cache)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden[torch.arange(rows, device=self.device), read_at][: len(texts)]

    def _answer(self, prefill: Prefill, questions: list[Question], tokens: int, context_ms: float) -> Result:
        # Already validated: both public entry points call `validate` before the context is read, and repeating it
        # here would tokenise every question a second time on the request path.
        #
        # `group` is the widest batch the cache was allocated for, and each group of questions uses only as many of
        # its rows as it has questions. That used to be the full width always, on the strength of a measurement
        # showing a branch pass cost the same at width 1 and width 32. That measurement was taken at 3,040 context
        # tokens and does not hold at longer ones: at 24,327 tokens the same pass costs 238 ms at width 1 and 360 ms
        # at width 32, and its transient peak goes from 0.111 GiB to 4.970 GiB. Both of those are per-row work
        # proportional to the context, so a request with three questions should not pay for thirty-two rows of it.
        largest_group = self.group
        plans = [plan(q, self.tokenizer) for q in questions]
        token_ids = [p.token_ids for p in plans]
        width = self._width_for(plans)

        # Before the clock starts, and outside the lock's timed section: a prior is cached per question, so charging
        # the first request for every later one's correction would report a cost that is not there.
        priors = self.calibration.priors(self, questions, plans) if self.calibration else None

        start = _now(self.torch_device)
        probabilities: list[torch.Tensor] = []
        widest_chunk = 0
        with self._lock, torch.inference_mode():
            before = self._peak_baseline()
            try:
                for lo in range(0, len(questions), largest_group):
                    chunk = [p.text for p in plans[lo : lo + largest_group]]
                    widest_chunk = max(widest_chunk, len(chunk))
                    # How many more groups of this exact shape this call will run, which is what decides whether
                    # recording the pass can pay for itself. Only full groups share the shape: the last chunk is
                    # narrower unless the questions divide evenly.
                    full_left = (len(questions) - lo - largest_group) // largest_group
                    hidden = self._branch(
                        prefill, chunk, len(chunk), width, remaining=full_left if len(chunk) == largest_group else 0
                    )
                    probabilities.extend(
                        score(
                            hidden,
                            self.unembedding,
                            token_ids[lo : lo + largest_group],
                            priors[lo : lo + largest_group] if priors else None,
                        )
                    )
            except torch.OutOfMemoryError as e:
                # The allocator is the authoritative answer to "does this fit", and admission is only a pre-filter:
                # it budgets from what has been observed, and the first pass on an engine has nothing to observe.
                # Naming the knobs here is the difference between a diagnosis and a byte count.
                raise PrismyraError(
                    f"ran out of memory answering {len(questions)} questions about {tokens} context tokens at "
                    f"group={self.group}. Ask fewer questions at a time, or build the engine with a smaller group; "
                    f"the context itself is held once and is not what grew."
                ) from e
            self._observe_peak(before, tokens, widest_chunk)
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
            scoring=self.calibration.mode if self.calibration else "raw",
            timing=Timing(context_ms=context_ms, readout_ms=readout_ms),
        )

    def _paged_reads(self) -> int:
        """Branch reads the paged layers have actually served in this process. Zero with the flag on means the flag is a
        lie, which is exactly what happened the first time this path was written -- and what the first version of this
        method reported, because it summed over an attribute that never existed. A counter that can only ever return
        zero is worse than no counter, since it reads as evidence."""
        from .paged import PagedForkLayer

        return PagedForkLayer.reads_served

    def _peak_baseline(self) -> int | None:
        """Where the allocator stood before a pass, or None off CUDA. Resets the peak so the next reading is this
        pass's own and not a larger one from some earlier request."""
        if self.torch_device.type != "cuda":
            return None
        torch.cuda.synchronize(self.torch_device)
        torch.cuda.reset_peak_memory_stats(self.torch_device)
        return int(torch.cuda.memory_allocated(self.torch_device))

    def _observe_reading(self, before: int | None, context_tokens: int) -> None:
        """Remember what reading a context transiently cost, per token, above the cache it left behind.

        The cache is subtracted because it is held rather than transient and is already budgeted separately; leaving it
        in would count it twice and grow the double-count with the group.
        """
        if before is None or not context_tokens:
            return
        peak = int(torch.cuda.max_memory_allocated(self.torch_device)) - before
        transient = max(0, peak - self.cache_bytes(context_tokens))
        per_token = transient // context_tokens
        seen = self._observed_reading_per_token
        self._observed_reading_per_token = per_token if seen is None else max(seen, per_token)

    def _observe_peak(self, before: int | None, context_tokens: int, rows: int) -> None:
        """Remember the largest transient per row, so the next admission can budget it.

        Kept as a maximum rather than an average: admission is deciding whether a pass will fit, and the pass that
        matters is the largest one. No synchronise here beyond the one the caller's timing already does, so this costs
        a host-side read.
        """
        if before is None or not rows:
            return
        peak = int(torch.cuda.max_memory_allocated(self.torch_device)) - before
        self._observed_row_constant = row_constant(
            self._observed_row_constant,
            max(0, peak) // rows,
            context_tokens + WIDTHS[-1],
            join_bytes_per_token(self.config, self.dtype, doubled=not self._borrowed_kernel),
        )
        self._observed_at_rows = max(self._observed_at_rows, rows)

    def _width_for(self, plans: list) -> int:
        """The branch width these questions need, rounded to a pinned bucket.

        One place, because the calibration measures its priors through the same passes an answer comes through, and a
        prior taken at a different width would correct for a different arrangement of the batch.
        """
        widest = max(len(self.tokenizer("\n" + p.text, add_special_tokens=False)["input_ids"]) for p in plans)
        try:
            return round_width(widest)
        except TooWide as e:
            raise PrismyraError(str(e)) from e

    def _branch(self, prefill: Prefill, texts: list[str], rows: int, width: int, remaining: int = 0) -> torch.Tensor:
        # The snapshot is taken on the first branch, when the cache holds exactly the context, so restoring it also
        # puts every layer's token count back to the end of the context. One mechanism, not a state restore plus a
        # separate rewind: two of them can disagree, and the one that is wrong answers plausibly.
        if prefill.snapshot is None:
            prefill.snapshot = snapshot(prefill.cache)

        ids, read_at, _ = build_suffixes(texts, self.tokenizer, self.device, rows, width)
        # From where the model thinks the context reached, which is past its token count when media widened it.
        start = prefill.position_from or prefill.tokens
        positions = torch.arange(start, start + ids.shape[1], device=self.device).expand(rows, -1)

        def run(suffix: torch.Tensor) -> torch.Tensor:
            """One branch pass, with the fork done by the caller.

            The fork is deliberately **not** in here, and that was found by measurement rather than reasoned out. With
            it inside, one replay disagreed with the eager pass and two replays disagreed with each other -- the
            signature of a buffer a recording both reads and writes without resetting. The layer rebinds its recurrent
            state to a new tensor on every pass, and a rebinding is Python: a recording keeps the tensor it saw and a
            replay cannot repeat the assignment. So the fork runs eagerly every time, at the cost of a few copies per
            layer, and the recording covers only what is pure device work.
            """
            out = self.backbone(input_ids=suffix, position_ids=positions, use_cache=True, past_key_values=prefill.cache)
            return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        hidden = self._run_branch(prefill, run, ids, rows, width, positions, remaining)
        # Each row is read at its own last real token, which is why the padding cannot reach an answer.
        return hidden[torch.arange(rows, device=self.device), read_at][: len(texts)]

    def _replay_disagreement(self, taken, prefill, ids, rows: int, reference) -> tuple[float, float]:
        """The worst a recording's replays are from the pass it was taken from, and what a replay costs -- both measured
        here because both need a replay and these are the only replays that answer nothing.

        Two replays rather than one.

        Two, because one is not enough and that was measured rather than supposed. A recording can reproduce its pass
        perfectly the first time it is replayed and then diverge, by the same amount, on every replay after -- 0.123 on
        this model, repeatable, and the cause is not identified. The first replay happens with the device in exactly the
        state the recording was taken in; the second happens after the fork and the bindings have been put back by
        Python, which is the state every real use is in. Checking only the first proves the easy case.
        """
        worst = 0.0
        # The fork is outside the recording and a real replay pays for it too, so it is inside the timing. What is being
        # measured is the cost of one replayed pass as a caller would experience it.
        started = _now(self.torch_device)
        for _ in range(REPLAY_CHECKS):
            taken.before_fork(prefill.cache)
            self.fork(prefill, rows)
            replayed = taken.replay(prefill.cache, ids)
            worst = max(worst, float((replayed.float() - reference.float()).abs().amax()))
        return worst, _since(started, self.torch_device) / REPLAY_CHECKS

    def _recordings_for(self, cache) -> dict:
        """The recordings taken on this cache, and how many times each shape has been seen on it.

        Attached to the cache rather than to the context because the cache outlives the context now, and a recording is
        only valid for the allocation it was taken on. Identity, not equality: two caches of the same size are different
        allocations and a recording from one must never answer on the other.
        """
        held = self._cache_recordings.get(id(cache))
        if held is None:
            held = {"taken": {}, "seen": {}}
            self._cache_recordings[id(cache)] = held
        return held

    def _claim_cache(self, tokens: int):
        """A cache sized for a bucket rather than for this context, reused if one is free.

        **Bucketed** means the allocation stops depending on the exact context length, so contexts of similar length
        share one size. That is shipped, and it is one of the two things a recorded pass needs to outlive one document.

        **Pooled** -- keeping a closed context's cache and resetting it for the next one, so its addresses survive -- is
        not. Three attempts, and each found something real before failing:

        * `reset()` on the framework's own recurrent layers clears their contents and keeps their batch dimension, so a
          cache returned by a three-row pass held three-row state and the next context's fork tried to widen three rows
          to thirty-two. `fork.OWNED` fixes that and is shipped for its own sake;
        * owning that state at the full group width allocated about a gigabyte, and doing it inside the first branch
          pass put the whole allocation into the peak that pass was measured by -- divided by that pass's rows and
          multiplied by the group, a one-off gigabyte became a 24 GiB answering estimate. It is allocated while the
          context is read now, and `cache.state_bytes` counts it as held, which **admission had never done**;
        * with all of that fixed, six device tests still refuse contexts on memory, and that is **not diagnosed**.

        What the pool is worth is measured: five documents of the same length, one group each, replaying from the third
        at 33.4 ms against 109 eagerly with identical answers. What it costs is still unknown, which is why a fourth
        attempt should begin by measuring the held memory of a pooled engine rather than by writing more of it.
        """
        room = self.room_for(tokens)
        cache = build_cache(
            self.config, room + WIDTHS[-1], self.group, self.dtype, self.device, WIDTHS[-1], paged=self.paged
        )
        self._made_caches += 1
        return cache, room

    def _release_cache(self, prefill) -> None:
        """Where a closed context would hand its cache back for the next one. There is no pool yet; `_claim_cache` says
        what three attempts at one cost and what is still unexplained."""
        if prefill is not None:
            self._cache_recordings.pop(id(prefill.cache), None)

    def fork(self, prefill, rows: int) -> None:
        """Put every layer back to the end of the context and widen it to `rows`.

        Its own method because a recording's warm-up and capture passes each need it, and a fork done once for three
        passes would have the second and third continuing from the first.
        """
        assert prefill.snapshot is not None
        restore_and_fork(prefill.cache, prefill.snapshot, rows, width=self.group)

    def _run_branch(self, prefill, run, ids, rows: int, width: int, positions, remaining: int = 0):
        """The pass, replayed from a recording where there is one and recorded where a second one is worth taking.

        The order is what makes this safe. A recording is not a result: under stream capture the kernels are written
        down rather than run, so the pass is executed eagerly for its answer *first* and recorded afterwards.

        **Whether to record is arithmetic, not a habit.** `graphs.pays_from` says three passes at this shape must still
        be coming, because a recording costs 151 ms plus two proving replays and each replay after that saves 79. Two
        things can supply that number, and neither is a guess: `remaining` is how many more groups of this shape the
        call in progress will run, which the caller has already told the engine by handing over all its questions at
        once; and the count of times this shape has come back on this cache, which is evidence about a session asking
        group after group. A shape that has neither records nothing, so a caller asking one group about a document it
        will not revisit pays nothing for machinery it never uses.
        """
        if not self.graphs or self.torch_device.type != "cuda":
            self.fork(prefill, rows)
            return run(ids)

        # Keyed on the cache rather than on the context, because the cache is what a recording holds the addresses of.
        # A context that closes returns its cache to the pool with its recordings attached, so the next context of the
        # same size replays instead of recording again.
        store = self._recordings_for(prefill.cache)
        key = (rows, width)
        recorded = store["taken"].get(key)
        if recorded is not None:
            # The bindings first, then the fork: the fork must write the context into the tensors the recording reads,
            # and after the last pass those are not the ones the layers point at.
            recorded.before_fork(prefill.cache)
            self.fork(prefill, rows)
            wrong = recorded.usable(prefill.cache)
            if wrong is None:
                self.replays[key] = self.replays.get(key, 0) + 1
                return recorded.replay(prefill.cache, ids)
            # A recording that no longer describes the cache is discarded rather than replayed. The alternative is a
            # plausible answer, and this package treats that as the worst outcome available.
            del store["taken"][key]
            self.declined_recordings[key] = wrong
            return run(ids)

        self.fork(prefill, rows)
        # Copied, and this is not defensive housekeeping. A recording replays into buffers the allocator may have handed
        # out for this pass's own output, so a replay can overwrite the answer that was just computed -- which made the
        # first version of the check below compare a tensor against itself and pass every time, and would have returned
        # the replay's values as this call's answer. The bug was found by a replay that gave a wrong answer while the
        # check reported agreement to zero.
        # Timed, because whether a recording can pay is a question about this shape on this card and the answer is not a
        # constant. See `_worth_keeping`.
        started = _now(self.torch_device)
        hidden = run(ids).clone()
        eager_ms = _since(started, self.torch_device)
        store["seen"][key] = store["seen"].get(key, 0) + 1
        expected = max(remaining, store["seen"][key] - 1)
        if expected >= pays_from() and key not in self.declined_recordings:
            taken, why = record(run, prefill.cache, ids, fork=lambda: self.fork(prefill, rows), keep=(positions,))
            if taken is None:
                # Remembered so it is attempted once per shape rather than once per group, and reported rather than
                # retried in silence.
                self.declined_recordings[key] = why or "unknown"
            else:
                # Replayed once and checked against the pass it was taken from, before it is allowed to answer anything.
                #
                # Not a formality. A recording that reproduced the eager pass perfectly on a fresh engine stopped doing
                # so once other passes at other row counts had run first -- measured, and the cause is not yet
                # identified. Checking the bindings it holds was not enough to catch that, so the check is the thing
                # itself: if the first replay does not reproduce the answer already in hand, the recording is discarded.
                # A wrong answer that looks right is the worst outcome available here, and this is what makes it
                # impossible rather than unlikely.
                moved, replay_ms = self._replay_disagreement(taken, prefill, ids, rows, hidden)
                self.verified_recordings[key] = moved
                self.replay_cost[key] = (round(eager_ms, 1), round(replay_ms, 1))
                slow = keeping_pays(eager_ms, replay_ms, expected)
                if slow is not None:
                    del store["taken"][key]
                    self.declined_recordings[key] = slow
                elif moved > REPLAY_TOLERANCE:
                    self.declined_recordings[key] = (
                        f"a replay moved a hidden state by {moved:.3e}, above {REPLAY_TOLERANCE:.0e}"
                    )
                else:
                    store["taken"][key] = taken
                # The recording ran the pass again while writing itself down, which left the cache where that pass
                # left it -- not where the eager pass above left it. They are the same state by construction, and
                # `hidden` was read before any of it, so the answer this call returns is the eager one.
        return hidden


def row_constant(seen: int | None, per_row: int, at_tokens: int, per_token: int) -> int:
    """One row's context-free transient, taken from an observed per-row peak by subtracting the part that scales.

    Separate and pure because it is the rule admission depends on, and it can then be checked without a device.

    Ratcheted upwards: admission is deciding whether a pass will fit, so the largest constant seen is the one to
    budget. Ratcheting this rather than a prediction is what keeps the estimate from drifting towards refusing work --
    this quantity does not depend on the context length, so an observation at any length is comparable with any other.

    Floored at zero. A measured peak below the arithmetic slope means the two never overlapped the way the slope
    assumes, and a negative constant would budget less than the slope alone.
    """
    constant = max(0, per_row - per_token * at_tokens)
    return constant if seen is None else max(seen, constant)


def _load_processor(model: str):
    """The model's processor, or None. Absent is the ordinary case for a text-only checkpoint, not a failure."""
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model)
    except Exception:  # noqa: BLE001 - a checkpoint without one simply cannot take images, which `encode` reports
        return None


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


def _put_back_conv_states(cache, boundaries) -> None:
    """Replace each convolution state with the per-document tails recorded during the pass.

    The tails arrive in the order the convolutions ran, which is layer order, and the count is checked against the
    layers that keep a convolution state. A mismatch raises: a wrong window is a plausible answer, and an assertion is
    the only thing between the two.
    """
    wanted = [(layer, key) for layer in cache.layers for key in (getattr(layer, "conv_states", None) or {})]
    if len(wanted) != len(boundaries.conv_tails):
        raise PrismyraError(
            f"{len(boundaries.conv_tails)} convolution windows were recorded during the pass and {len(wanted)} layers "
            f"keep one. Without one each, a document's rows would convolve from another document's tail."
        )
    for (layer, key), tails in zip(wanted, boundaries.conv_tails, strict=True):
        held = layer.conv_states[key]
        if held is not None and tails.shape[1:] != held.shape[1:]:
            raise PrismyraError(
                f"a recorded convolution window is {tuple(tails.shape)} and the layer keeps {tuple(held.shape)}"
            )
        layer.conv_states[key] = tails.to(dtype=held.dtype) if held is not None else tails
