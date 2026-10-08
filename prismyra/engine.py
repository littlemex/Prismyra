"""The engine: load a model, read a context once, answer typed questions about it.

`open_context` is the primitive and `ask` is sugar over it. The distinction matters: a follow-up against an open
context costs a branch, while calling `ask` again re-reads the context. A library that only offered `ask` would hide
the thing it exists to provide.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - the framework's cache type, for the annotation only
    from transformers.cache_utils import Cache

from . import interleave, kernels, onepass, varlen
from .cache import build_cache, cache_bytes, join_bytes_per_token
from .calibration import Calibration
from .fork import (
    WIDTHS,
    Prefill,
    TooWide,
    branch_ids,
    build_suffixes,
    pick,
    restore_and_fork,
    restore_and_fork_many,
    round_width,
    snapshot,
    snapshot_bytes,
)
from .graphs import keeping_pays, pays_from, record
from .heads import Heads
from .kernels.autotune import pin as pin_autotunes
from .kernels.nvfp4 import tactics_status as _nvfp4_tactics_status
from .media import Encoded, encode, position_offset
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
#: A packed branch group is rounded up to a multiple of this many tokens. Eight keeps the matmul tiles aligned
#: without giving back much of what packing saves.
PACK_ALIGN = 8
#: The question count above which `ask()` widens this one document's own branch-row
#: capacity to `WIDE_GROUP` instead of leaving it at `self.group`. A document with more questions than
#: `WIDE_GROUP_FROM` pays for `ceil(questions / self.group)` branch passes today, each re-streaming the whole routed
#: expert weight set; one pass of up to `WIDE_GROUP` rows removes that re-stream for the common two-pass case
#: (33-64 questions), measured on L40S, fp8-36l, interleaved. Below this threshold nothing
#: changes -- a request narrower than one `self.group`-wide pass already uses only as many rows as it has questions.
WIDE_GROUP_FROM = 32
WIDE_GROUP = 64

#: For a 64-question document, the second branch group is also routed through the layer-interleaved path, matching
#: the first, below this token limit: widening to `WIDE_GROUP` for `interleaved_fork` costs 36 layers' worth of wider
#: GDN state buffers, paid once whatever the context holds; it is worth it only while the restream it removes is
#: still the larger of the two. An alternating length sweep on L40S, fp8-36l, 64 questions
#: (`tools/s4c_64q_length_sweep.py`) found the sign flip between 7,068 context tokens (-4.06% vs the two-pass
#: baseline, widening still wins) and 9,044 (+0.95%, widening now loses) -- a linear interpolation puts the crossing
#: at about 8,669 tokens. This constant sits below the measured losing point with a margin, not at the interpolated
#: crossing itself, because the sweep's own two neighbouring points already bracket it to within ~2,000 tokens and
#: this only needs to stay on the winning side of that bracket, not pinpoint it exactly. A context this long or
#: longer keeps using the non-widened interleaved path instead of falling all the way back to two-pass: that path
#: never regressed at any length tested, including 20,064 tokens (`tools/s4c_16q_length_check.py`, -3.19%) and is
#: what measurement found for 64 non-widened questions at long context (-0.7%) -- strictly better than paying the
#: widening's cost and strictly better than giving up the fusion altogether.
INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT = 8192

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

#: Free device memory a new recording's capture is refused below. 2 GiB, which is small next to the 44 GiB this was
#: measured on -- it is a check that the ordinary eager pass every call still runs first has not already been
#: starved, not a budget for the capture itself. Raising it to 8 GiB was tried and made every GPU test that takes a
#: recording fail by refusing the capture outright: this model's weights alone leave about 10 GiB free on a 44 GiB
#: card, and 8 GiB of margin is most of that before admission's own budget or an ordinary pass gets any of it. The
#: number that needed changing was not this one -- see `MAX_KEPT_RECORDINGS`.
GRAPH_MEMORY_MARGIN = 2 * 1024**3

#: How many distinct shapes one cache may hold a kept recording for at once. A kept recording's pool is never freed,
#: so unlike everything else admission budgets, this is not something a size-based margin can bound on its own: an
#: open-loop run against real traffic, with the decline-retry above in place and only a memory margin to stop it,
#: drove free device memory from several gigabytes to a few megabytes across a handful of distinct shapes each
#: keeping one recording, and into a tight allocate-fail-retry loop that was not even inside a recording attempt --
#: it was the ordinary eager pass every call still runs first. A margin only ever asks "is there room for one more";
#: it does not know how expensive the attempts already granted turned out to be, so raising it cannot bound a count
#: of unknown-sized things. A small fixed count can. Two, because the recordings measured so far during the paged
#: branch pass's remainder-bucket work moved a probability by nothing and a replay cost half an eager pass -- worth
#: having at all -- and this package has not yet measured what one recording actually costs in isolation, which it
#: would need before this number could be raised with evidence instead of nerve.
MAX_KEPT_RECORDINGS = 2

#: How long `empty_cache()` is skipped after it was tried and *still* left free memory at or below the margin --
#: on this card, meaning the margin is genuinely tight right now, not just fragmented. Comparing this reclaim call
#: present/absent at arrival rates 40/60 found it costs 2-4% of questions/second (60.3-61.4 q/s with it vs 63.0-63.5
#: without, the dispatcher narrowing in `_enable_batch_invariance` measured as unrelated noise by the same
#: comparison) -- a real cost from
#: a real CUDA synchronisation, paid every time admission runs while the margin stays tight, which an open-loop
#: run under load does repeatedly. 1 second, long enough that a tight stretch is not retried every single pass
#: (the thing costing the 2-4%) and short enough that genuine headroom returning (a resident dropped, a recording
#: declined) is noticed within a couple of passes rather than held stale for the life of the engine.
RECLAIM_COOLDOWN_S = 1.0

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
class Shelved:
    """One document on a shelf: where it is, how long it is, and the state a branch forks from."""

    handle: int
    tokens: int
    snapshot: dict = field(repr=False)
    position_from: int = 0
    #: Bytes `snapshot` actually holds -- the recurrent state's clone, not the pages. Measured at `put_many` time,
    #: because it does not depend on `tokens` at all: a recurrent layer's state is the same size whatever the
    #: document was, so a shelf's non-page cost grows with how many documents it holds, not with how long they are.
    #: See `schedule.Batcher._make_room`, which is what this field exists for.
    snapshot_bytes: int = 0


@dataclass
class Shelf:
    """Documents kept on the device across requests, answered a batch at a time.

    The difference from `Batch` is what outlives a request. A batch owns its cache, so a second question about the same
    document reads it again; a shelf owns the cache and the documents are what come and go. That is the shape a server
    wants, and it is what makes `Pool`'s page reuse load-bearing: a dropped document gives its pages back for the next
    one.

    What a shelf holds per document is its pages and **a snapshot of the recurrent state its read ended in**. The pages
    are the attention layers' share and live in the pool; the recurrent layers keep one state per row rather than per
    document, so a document's state has to be kept aside and copied into its rows when it is answered. That is the same
    snapshot a single context has always taken -- `fork.snapshot` -- and the only new part is that several are alive at
    once.
    """

    _engine: Prismyra
    _cache: Cache | None
    room: int
    #: Which engine lane this shelf's passes run under (lock, stream, `fork.OWNED` buffer) -- see `Prismyra.ask`'s
    #: lane plumbing and `prismyra.schedule.Batcher(lanes=2)`, the only caller that opens a shelf with anything but
    #: the default. Each lane owns its own `Shelf`/`Pool`/cache (never shared between lanes), so this is "which
    #: lock does *this* shelf's own pass take", not a routing decision made per document.
    lane: int = 0
    #: Documents on the shelf, by the handle `put` returned.
    documents: dict[int, Shelved] = field(default_factory=dict)
    _next_handle: int = 0

    # ---------------------------------------------------------------- putting documents on
    def put(self, context: str) -> int:
        """Read one document onto the shelf and return its handle."""
        return self.put_many([context])[0]

    def put_many(self, contexts: list[str]) -> list[int]:
        """Read several documents in one pass and return their handles, in order.

        One pass, because reading is mostly fixed cost -- 110 ms plus 11 ms a thousand tokens -- so reading five
        documents together costs about what reading one costs.
        """
        if self._cache is None:
            raise PrismyraError("this shelf has been closed")
        if not contexts:
            raise PrismyraError("putting nothing on a shelf is not an operation")
        engine = self._engine
        encoded = [engine.encode_context(one) for one in contexts]
        if any(one.has_media for one in encoded):
            raise PrismyraError(
                "a shelf is text only for now, for the same reason a batch is: media move the positions"
            )
        handles = list(range(self._next_handle, self._next_handle + len(encoded)))
        self._next_handle += len(encoded)

        started = _now(engine.torch_device)
        with torch.inference_mode():
            # The read must start from no recurrent state, and the shelf's layers are carrying whatever the last pass
            # left. Zeroed rather than reset: reset would clear the pages, which is where the documents already on the
            # shelf live.
            _forget_recurrent_state(self._cache)
            # Same reason `_read` pads a solo read: the borrowed chunked recurrent kernel picks its own
            # configuration by this read's *total* length, not by what any one document in it is, so a document put
            # on the shelf alongside different company at different times can otherwise end its read at a different,
            # real-but-inconsistent state. See `Prismyra._pad_context_lengths`.
            lengths = [one.tokens for one in encoded]
            pad_ids, lengths = engine._pad_context_lengths(lengths, [one.input_ids for one in encoded])
            # The padding needs a handle of its own, same as every real document (`_write_context`'s own check), one
            # that nothing on this shelf is using. `self._next_handle` is reserved for the *next* `put_many` call and
            # has not been handed out yet, which is exactly what makes it free to borrow for the length of this one.
            pad_handle = self._next_handle + len(encoded)
            has_pad = pad_ids.shape[1] > 0
            begin_handles = [*handles, pad_handle] if has_pad else handles
            for layer in self._cache.layers:
                begin = getattr(layer, "begin_documents", None)
                if begin is not None:
                    begin(begin_handles)
            ids = torch.cat([*(one.input_ids for one in encoded), pad_ids], dim=1)
            with varlen.reading(lengths, engine.device) as boundaries:
                engine.backbone(
                    input_ids=ids,
                    position_ids=boundaries.positions(engine.device),
                    use_cache=True,
                    past_key_values=self._cache,
                )
                _put_back_conv_states(self._cache, boundaries)
                engine._check_batched_read(self._cache, boundaries)
            taken = snapshot(self._cache)
            if has_pad:
                # The padding answers nothing and keeps no row on this shelf, so its pages go back now rather than
                # sitting here as a document nobody will ever ask about or drop.
                for layer in self._cache.layers:
                    release = getattr(layer, "release_document", None)
                    if release is not None:
                        release(pad_handle)
        engine._note_read(_since(started, engine.torch_device), len(encoded))
        for at, (handle, one) in enumerate(zip(handles, encoded, strict=True)):
            piece = pick(taken, at)
            self.documents[handle] = Shelved(
                handle=handle,
                tokens=one.tokens,
                snapshot=piece,
                position_from=one.tokens,
                snapshot_bytes=snapshot_bytes(piece),
            )
        return handles

    def drop(self, handle: int) -> None:
        """Take a document off the shelf and give its pages back."""
        if self._cache is None:
            raise PrismyraError("this shelf has been closed")
        if handle not in self.documents:
            raise PrismyraError(f"this shelf does not hold document {handle}")
        # This shelf's own lane's lock, because a pass running on another thread *of this lane* has rows naming
        # this document's pages and the run would be handed to the next one underneath it. A different lane's
        # pass never names this shelf's pages at all -- each lane owns its own `Shelf`/`Pool` -- so it does not
        # need to wait for this.
        with self._engine._lock_for(self.lane):
            for layer in self._cache.layers:
                release = getattr(layer, "release_document", None)
                if release is not None:
                    release(handle)
            del self.documents[handle]

    # ---------------------------------------------------------------- answering
    def ask(self, asked: dict[int, list[Question]], lane: int | None = None) -> dict[int, Result]:
        """Answer questions about documents already on the shelf, in one pass. No reading happens here.

        `lane` names which of the engine's locks/streams/`fork.OWNED` buffers this pass uses, defaulting to this
        shelf's own `self.lane` -- see `prismyra.schedule.Batcher(lanes=2)`, the only caller that sets `self.lane`
        to anything but 0.
        """
        if lane is None:
            lane = self.lane
        if self._cache is None:
            raise PrismyraError("this shelf has been closed")
        if not asked:
            raise PrismyraError("a pass needs at least one document to answer about")
        missing = [handle for handle in asked if handle not in self.documents]
        if missing:
            raise PrismyraError(f"this shelf does not hold {missing}; put the document on it or use its handle")
        engine = self._engine
        handles = list(asked)
        prefills = [
            Prefill(
                snapshot=self.documents[handle].snapshot,
                cache=self._cache,
                room=None,
                tokens=self.documents[handle].tokens,
                last_position=torch.tensor([self.documents[handle].tokens - 1], device=engine.device),
                position_from=self.documents[handle].position_from,
            )
            for handle in handles
        ]
        results = engine._answer_batch(
            prefills, [asked[handle] for handle in handles], context_ms=0.0, rows_for=handles, lane=lane
        )
        return dict(zip(handles, results, strict=True))

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._cache is not None:
            self._engine._forget_recordings(self._cache)
            self._cache = None
            self.documents.clear()

    def __enter__(self) -> Shelf:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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
        short_graphs: bool | None = None,
        heads: str | list | None = None,
        pin_autotune: bool = True,
        wide_group: bool = False,
        interleaved_fork: bool = True,
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
        # **Opt-in, off by default**: widening a document's own branch-row capacity
        # past `self.group` when it is asked more than `WIDE_GROUP_FROM` questions (see `_group_for`).
        # A `torch.equal` gate first measured this as **not** bit-identical for 33 and 40 questions
        # and only bit-identical for the one case that divides evenly by
        # `self.group` (64 = 32+32) -- the row-count-dependent GDN causal convolution took a different code path
        # (the framework's own fallback, not the fast kernel) whenever a branch pass's row count was not exactly
        # 1, which the leftover group from an uneven split always was and the full 32/64-row group never was.
        # Re-running that gate across the whole 33-64 range, after a branch-pass
        # convolution batch fix (`prismyra/kernels/qwen3_moe.py`'s `_install_conv`) closed that fallback for every
        # row count, not only one: all 32 widths, both context lengths tested, both supported cards, come back
        # `torch.equal` now, including with the batch-invariance claim below monkeypatched off (so that claim was
        # never what fixed this). The mismatch this flag was switched off for is gone, but it stays off anyway: the
        # widened pass measures slower than the two-pass path it replaces at every width and document length
        # measured, on at least one supported card, so the default stays off for speed rather than correctness.
        self.wide_group = wide_group
        # **On by default as of this release**: `ask()`
        # with more than one question goes through `interleave.read_and_branch` instead of
        # `open_context(...).ask(...)` -- one layer-interleaved pass fusing the context's and the first branch
        # group's dense/MoE compute per layer, instead of two full passes. This used to stay off by default
        # pending `torch.equal` confirmation and the construction-flag-dependent one-pass answer tracked at
        # `_enable_batch_invariance`'s own call site below; both are now closed. `torch.equal` against the
        # two-pass path is confirmed on real hardware across both supported cards (L40S/fp8-36l,
        # RTX PRO 4500/nvfp4-36l), two context lengths an order of magnitude apart, and every question count this
        # project asks it to handle differently (16, 32, 33, 64 -- see `tests/test_gpu.py`'s
        # `test_the_layer_interleaved_fused_path_answers_as_the_two_pass_path_did`
        # for the real-hardware confirmation). Measured faster on real RACE documents at every width above
        # one question on both cards; `False` remains available for a caller that
        # wants to rule the fused path out while debugging. `self.wide_group` is a separate switch and stays off
        # by default -- it widens a different range (33-63 questions in the *un-fused* path), which this flag
        # does not touch. That range's own `torch.equal` mismatch has since been
        # closed too (see `self.wide_group`'s own comment above), independently of this flag; its
        # default stays off for the same speed reason, a separate decision from this flag's.
        # Requires the borrowed kernels (`self._borrowed_kernel`, decided below, after the kernels are applied)
        # -- see `ask`'s guard.
        self.interleaved_fork = interleaved_fork
        # The caches are held by the engine and mutated in place, so two threads asking at once would interleave one
        # another's branches. The lock makes that safe; `prismyra.queue.Worker` is still what makes it fast.
        self._lock = threading.RLock()
        # Lane=2: a second lane needs its own lock, because the point
        # is for two passes to actually run at once rather than queue behind one `RLock`. Lane 0 keeps `self._lock`
        # itself (identity, not a copy) so every existing single-lane caller is unaffected; a lane's own cache/Pool
        # (a separate `Shelf`, built by `prismyra.schedule.Batcher(lanes=2)`) is what makes the lock sufficient --
        # two lanes never touch the same Pool, only the same model weights (read-only) and lane-tagged `fork.OWNED`
        # scratch (see `_owned`'s `lane` argument).
        self._locks: dict[int, threading.RLock] = {0: self._lock}
        #: One CUDA stream per lane beyond the first, so a lane's kernels can actually run concurrently with
        #: another lane's rather than just being launched from a different Python thread onto the one stream every
        #: `Prismyra` has used until now (lane 0 keeps that stream -- `None` here means "the current/default
        #: stream", not "no stream"). Built lazily, once per lane, because building one costs nothing worth paying
        #: for an engine that is never asked for a second lane.
        self._streams: dict[int, "torch.cuda.Stream | None"] = {0: None}
        #: Monotonic deadline before `empty_cache()` is tried again, once it has been tried and still left free
        #: memory at or below `GRAPH_MEMORY_MARGIN`. See `RECLAIM_COOLDOWN_S`.
        self._reclaim_cooldown_until = 0.0
        #: How many times this engine has actually called `torch.cuda.empty_cache()`
        #: from `_run_recorded`'s margin check, and how many milliseconds those calls cost in total. Measurement
        #: only -- nothing reads these to make a decision -- kept so an open-loop run can report the real count
        #: and cost instead of the proxy "how many passes reached the below-margin branch" used before this.
        self._empty_cache_calls = 0
        self._empty_cache_ms = 0.0
        #: Whether to record a branch pass and replay it. Off by default, and the reason is memory rather than doubt: a
        #: recording holds a private allocator pool, and this engine refuses a context by name from a budget it
        #: measures, so a feature that quietly takes device memory behind that budget would make the refusal wrong.
        #: Measured worth: a recorded pass replays in 28.9 ms against 107.7 eagerly, and recording costs 151.3 ms once.
        self.graphs = graphs
        #: Whether the attention read goes through a page table. Off by default. Its point is not memory -- that was
        #: measured at zero -- but that the read's shape stops depending on the context's length, so one recorded graph
        #: serves every context instead of one per open context. See `prismyra/paged.py`. Assigned through the `paged`
        #: property below (set once self.applied exists, further down this method) rather than here as a plain
        #: attribute: see that property's own docstring for why.
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
        #: Shapes a recording was attempted on and refused, with the reason. Reported through `stats()` rather than
        #: retried in silence -- but retried, once there is reason to think the answer would be different. See
        #: `_economics_needed` for which declines that applies to and why.
        self.declined_recordings: dict = {}
        #: The passes-still-expected count an economics decline (`keeping_pays` returned a reason) would need to flip
        #: to a keep, for shapes currently in `declined_recordings` for that reason. Nothing else is in here: a
        #: decline from a stale cache (`Recording.usable` failing) or a capture failure (`graphs.record` returning
        #: `None`) is not a question of *how many more passes*, so there is no number here that retrying it later
        #: would change.
        #:
        #: Exists because the alternative -- declining once and never asking again -- was found to be the reason a
        #: shape judged capable of a 1.94x replay was never once kept across 368 documents. The attempt gate below
        #: fires at `expected == pays_from()` (a shape's eighth sighting, under the gate's own threshold), and the
        #: shape's own measured ratio almost always needs a few more than that -- so the very first attempt, the only
        #: one that was ever going to happen, arrived already below the bar that would keep it. Four more sightings
        #: were often all that was missing, and nothing was watching for them.
        self._economics_needed: dict = {}
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
        #: Why an attempt at a homogeneous shape never got as far as
        #: `graphs.record` -- the attempt gate (`expected < pays_from()`), a standing decline already in
        #: `declined_recordings`, or `room_to_record` failing for one of its own two reasons (the memory margin,
        #: or `MAX_KEPT_RECORDINGS` already full). `room_to_record` failing was previously invisible: the gate's
        #: `and` short-circuits before `declined_recordings` is ever written to for that reason, so a card that
        #: never has two free recording slots at once would show zero declines and zero replays with no record of
        #: why. Reported through `stats()`.
        self._skip_reasons: dict[str, int] = {}
        #: The fastest read this engine has served, in milliseconds. An estimate of a pass's fixed cost, which is what
        #: it is for: reading is that fixed cost plus a slope in the tokens, so the shortest document seen is the
        #: closest
        #: thing to the intercept available without fitting a line. `None` until something has been read, and a caller
        #: that would make somebody wait on this figure should do nothing until it exists.
        self.fastest_read_ms: float | None = None
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
        #: The largest transient any read has cost, whatever its length. A read's cost has a constant part, and a budget
        #: that is only a slope in the tokens under-states a short read and -- when the slope is learned *from* a short
        #: read -- wildly over-states a long one.
        self._observed_reading_floor: int | None = None

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
        #
        # Experimental: routed experts in NVFP4, selected with PRISMYRA_EXPERTS=nvfp4 (prismyra/kernels/nvfp4.py).
        # `model` must then be a checkpoint directory whose index leaves the routed-expert weights out (the FP8
        # experts are never loaded; `tiny_experts` makes the framework's loader accept the missing keys). Converted
        # here, before `kernels.apply` below, so the adapter below finds `FusedExpertsFp4` already in place and
        # only has to recognise it rather than build it.
        use_nvfp4_experts = os.environ.get("PRISMYRA_EXPERTS") == "nvfp4" and on_cuda and fast_kernels
        if use_nvfp4_experts:
            from .kernels import nvfp4 as _nvfp4

            with _nvfp4.tiny_experts():
                self.backbone = AutoModel.from_pretrained(model, dtype=self.dtype, device_map=self.device)
            decoder_cfg = getattr(self.config, "text_config", self.config)
            _nvfp4.convert(self.backbone, decoder_cfg.num_experts_per_tok, self.torch_device)
        else:
            self.backbone = AutoModel.from_pretrained(
                model, dtype=self.dtype, device_map=self.device if on_cuda else None
            )
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
        #: Whether *this* engine currently holds a claim on the process-wide batch-invariance registration -- see
        #: the `paged` property. Must exist before the first assignment to `self.paged` below, which reads it.
        self._invariance_claimed = False
        # A *second*, permanent claim, independent of `self._invariance_claimed`
        # (which only tracks the `paged` property's own on/off toggle -- see that property's docstring). An engine
        # built with `interleaved_fork=True, wide_group=True`
        # answered a single question (n=1, `_ask_in_one_pass` -- a method with no branch on either flag) up to
        # 0.208 away from the default-off engine's answer to the identical document/question, on the same card.
        # Root cause, isolated on real hardware one axis at a time (`tools/diag_onepass_capture.py`,
        # `tools/diag_autotune_gap.py`, `tools/diag_on_reproducible.py`): `__init__` records the one-pass CUDA
        # graphs (`onepass.record_all`, below) *after* whatever this constructor's flags did to the
        # registration -- a graph capture bakes in whichever matmul kernel the dispatcher hands it at that exact
        # moment, so an engine built with `interleaved_fork=True` (which used to claim the registration here)
        # permanently replays the batch-invariant Triton kernel for n=1, while the default engine permanently
        # replays whatever cuBLAS/cuBLASLt's own heuristic happened to pick that construction. Triton-autotuner
        # racing was ruled out (`diag_autotune_gap.py`: every autotuner is already pinned to one candidate, zero
        # cache entries, by the time construction finishes) and re-recording under a *held* registration state
        # round-trips exactly (`diag_on_reproducible.py`: three independent re-recordings under "on" agreed to
        # the bit) -- re-recording under the *default* state did not (two re-recordings under "off" differed by
        # 0.0081, the same order as the on/off gap itself). That makes cuBLAS/cuBLASLt's own un-pinned heuristic
        # the thing that was never actually deterministic, construction to construction, on the plain default
        # path -- `interleaved_fork`'s old conditional claim was not adding a new side effect so much as it was
        # the only existing way to opt into the one fix (the dispatcher override) that happens to close it.
        # Claiming it unconditionally, for every CUDA engine, removes the construction-flag dependence by
        # removing the non-deterministic default path entirely, for every caller, not only ones that opted into
        # `paged` or `interleaved_fork` sharing a pass. A separate flag because `self._invariance_claimed` is
        # the `paged` property's own on/off switch (`tests/test_gpu.py::engine_paged` flips it both ways on a
        # shared engine, by design) -- reusing it here would let `engine.paged = False` drop this permanent
        # claim as a side effect, reopening the exact cross-test leakage already fixed once for a
        # different pair of ops. Both flags add to the same reference count (`_enable_batch_invariance`'s own
        # module-level counter), so holding both at once is safe and costs nothing extra -- the registration
        # itself is only ever done once, no matter how many claims are outstanding.
        self._invariance_base_claimed = False
        if on_cuda:
            try:
                _enable_batch_invariance()
                self._invariance_base_claimed = True
            except ImportError as e:
                self.applied.notes.append(
                    f"batch-invariant mode not available ({e}); every answer on this engine may move by who "
                    "else shares a pass, and a single question's answer may differ from one construction of "
                    "this engine to the next -- see diag_onepass_capture.py"
                )
        # Through the property, not `self._paged = paged`: construction is one of the two places a caller can turn
        # paging on (`Prismyra(..., paged=True)`), and the property is what makes that have the same effect as the
        # other one (`engine.paged = True` after construction, which `tests/test_gpu.py::engine_paged` and any other
        # caller flipping the flag between contexts does, by this file's own module docstring, "rather than loading
        # a second copy"). See the `paged` property for why both have to run `_enable_batch_invariance()`. Redundant
        # with the unconditional claim just above whenever that one succeeded (the registration is already on, so
        # this is a second, independent claim on the same global count, not a second registration) -- kept so that
        # `paged`'s own on/off toggle keeps working exactly as before even on a CUDA-unavailable `ImportError` path
        # where the claim above did not run.
        self.paged = paged
        # `interleaved_fork` used to need its own claim here for
        # exactly the reason `paged` already had one -- `read_and_branch`'s one `mlp` call runs the context's and
        # the branch's rows through the *same* routed-expert pass, and `fused_moe`'s own tile-size choice keys on
        # that call's total row count unless `VLLM_BATCH_INVARIANT=1` is set (`_enable_batch_invariance`'s own
        # docstring). That claim is now subsumed by the unconditional one above (every CUDA engine already holds
        # it), so this block only still runs for the same reason the `paged` block above does: covering the
        # `ImportError` path where the unconditional claim above did not succeed.
        if interleaved_fork and not self._invariance_claimed and not self._invariance_base_claimed and on_cuda:
            try:
                _enable_batch_invariance()
                self._invariance_claimed = True
            except ImportError as e:
                self.applied.notes.append(
                    f"batch-invariant mode not available ({e}); open_batch/Batcher/read_and_branch may still "
                    "move an answer by who else shares the pass"
                )
        self.unembedding = load_unembedding(model, self.hidden_size, self.device, self.dtype)
        # Off unless asked for. It is a change to what a probability means, and whether it is an improvement is a
        # measured question rather than an obvious one -- `evals/run.py` compares the two.
        self.calibration = Calibration() if calibrate else None
        #: Every timing-based Triton autotuner in the process held to one configuration (see `kernels.autotune`), so
        #: that answers do not depend on which candidate happened to win the race at a process's first call. Before the
        #: one-pass recordings are taken: a recording keeps the configuration it captured, so one picked by timing
        #: would be replayed for the life of the engine.
        self.autotune = pin_autotunes(self.torch_device, enabled=pin_autotune)
        #: Learned read-outs registered by option list (see `prismyra.heads`). A question whose options match none of
        #: them is read from the output embedding exactly as without heads.
        self.heads = Heads(heads, self.hidden_size, self.device)
        #: The one-pass read of a short request, recorded per length bucket. On by default wherever the one-pass path
        #: is the path a single question takes -- CUDA with the borrowed kernels, no calibration, no paged storage --
        #: because that read is host-bound at short lengths and a replay removes the wait (docs/PERFORMANCE.md). The
        #: memory it holds is measured when it is taken and admission counts it, which is what keeping the branch
        #: recordings off by default was protecting.
        self._one_pass = None
        wanted = short_graphs if short_graphs is not None else (on_cuda and self._borrowed_kernel)
        if wanted:
            if not on_cuda:
                raise PrismyraError(f"short_graphs records CUDA graphs and this engine is on {self.device}")
            if not (calibrate or paged):
                self._one_pass = onepass.record_all(self, self._pad_id(), self._read_one_pass)

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
            rendered = branch_ids(chosen.text, self.tokenizer)
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
        return cache_bytes(
            self.config,
            self.room_for(context_tokens) + WIDTHS[-1],
            self.group,
            self.dtype,
            WIDTHS[-1],
            paged=self.paged,
        )

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
        floor = self._observed_reading_floor or 0
        if self._observed_reading_per_token is None:
            return int(floor * ANSWERING_MARGIN)
        # The larger of the floor and the slope's prediction, not their sum: the floor was measured at a short read and
        # already contains whatever proportional part that read had, so adding them would count it twice.
        return int(max(floor, self._observed_reading_per_token * context_tokens) * ANSWERING_MARGIN)

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

    @property
    def paged(self) -> bool:
        """Whether the attention read goes through a page table. See `__init__`'s own comment on the attribute."""
        return self._paged

    @paged.setter
    def paged(self, value: bool) -> None:
        """Turning paging on is also the one condition `_enable_batch_invariance()` is gated on (`__init__`'s
        comment on `self.applied`), and a caller can turn it on two ways: at construction (`Prismyra(...,
        paged=True)`) or afterward, by assigning this attribute directly -- which this file's own module
        docstring recommends ("flipping the flag between contexts rather than loading a second copy") and which
        `tests/test_gpu.py::engine_paged` does, to share one set of weights between a joined-storage test and a
        paged one. Before this property existed, only the first path ran the invariance setup: `self.paged =
        paged` was a plain attribute, so `engine.paged = True` after construction left `open_batch`/`Batcher`
        running with no protection at all against the row-count-chosen-GEMM-algorithm effect
        `_enable_batch_invariance` exists for, silently, since nothing about assigning a bool raises or warns.

        This explains a 0.008346 residual on
        `test_open_batch_matches_ask_bit_for_bit_whatever_the_companions_total_length` (`tests/test_gpu.py`): a
        synthetic, kernel-level reproduction of every op that test touches came back bit-identical across row
        counts in isolation, with `_enable_batch_invariance()`'s dispatcher registered by hand first -- evidence
        against a second, separate, deeper cause for the residual, not conclusive on its own. Reproducing
        `engine_paged`'s exact
        two lines instead -- `Prismyra(MODEL)` then `engine.paged = True` -- showed `_BATCH_INVARIANT_DISPATCH_LIB`
        stayed `None`: the fixture's "flip the flag" path never ran the
        dispatcher registration at all, in an engine constructed exactly as every `engine_paged`-based test in
        this file constructs one. Fixing this property closed the residual on the real checkpoint to exactly
        0.0 (`measure_residual_after_fix.py`, both the short and the long companion); reverting to the plain
        attribute on the same weights, same process, same run reproduced a non-zero residual again (max
        7.657e-05 here, a different run with the same sign and the same cause as the 0.008346 residual above),
        which is the before/after pair that makes this the actual cause rather than a correlate of it.

        Reference-counted (`_enable_batch_invariance`/`_disable_batch_invariance`, both in this module), not a
        one-shot: the first version of this fix left the dispatcher registered for the rest of the process once
        any engine turned paging on, which broke two *other*, unrelated tests on this same test
        module's plain `engine` fixture (`test_a_short_question_replays_exactly_as_it_reads_eagerly` and
        `test_a_headed_question_replays_exactly_as_it_reads_eagerly`) -- `onepass.py`'s one-pass CUDA graph
        recording, left running under a dispatcher it was never recorded or verified against, once an earlier
        `engine_paged` test flipped this attribute and never flipped it back off in the sense of undoing the
        registration (`engine_paged`'s own `finally: engine.paged = was` restored this attribute but, before
        this fix, nothing noticed and nothing reversed the dispatcher). Production code cannot hit that
        collision -- `ask()`'s one-pass shortcut explicitly requires `not self.paged`, so one call never takes
        both paths -- but this test module's shared-weights fixture does, by design (its own docstring: "the
        same weights with the paged storage, by flipping the flag between contexts rather than loading a second
        copy"). This setter now claims and releases one count per *instance* (`self._invariance_claimed`), so
        turning paging back off on the engine that turned it on actually undoes the registration (confirmed by
        hand that dropping a `torch.library.Library`'s last reference and `gc.collect()`-ing restores the
        original op) once nothing else still needs it, instead of leaving it on for the rest of the process.
        """
        self._paged = value
        if value and self.torch_device.type == "cuda" and not self._invariance_claimed:
            try:
                _enable_batch_invariance()
                self._invariance_claimed = True
            except ImportError as e:
                self.applied.notes.append(
                    f"batch-invariant mode not available ({e}); open_batch/Batcher may still move an answer by "
                    "who else shares the pass"
                )
        elif not value and self._invariance_claimed:
            _disable_batch_invariance()
            self._invariance_claimed = False

    def open_context(
        self, context: str, *, images: list | None = None, videos: list | None = None, group: int | None = None
    ) -> Context:
        """Read a context and keep it open. The expensive half happens here, once.

        `images` and `videos` take anything the model's processor accepts -- a `PIL.Image`, a path, an array of frames
        -- and are read into the context alongside the text. This is where the design pays best: a frame costs the
        vision tower once and then behaves like any other context token, so the questions after it are nearly free.

        `group` overrides `self.group` for this one document's own branch-row capacity.
        `None` (the default, and every call site before this parameter existed) keeps the engine's own `self.group`.
        See `ask`'s `_group_for` for the one caller that sets it, and `_read` for why it must be decided before the
        read rather than widened later: the fork buffers it sizes are allocated once, while the context is read, and
        a later widen would move that allocation's cost into the first branch pass that needed it instead.

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
                    prefill = self._read(encoded, group=group)
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
        """Read a context and answer questions about it. Sugar for `open_context(...).ask(...)`.

        One question about a text context is answered in **one pass** over the context and the question together. The
        fork exists so that many questions share one read; with a single question there is nothing to share, and the
        separate branch pass it would cost is a whole traversal of the model (docs/PERFORMANCE.md). Only when nothing
        needs the fork: media (whose positions are worked out during the read), calibration (whose priors are measured
        through branch passes) and the paged storage (whose pages belong to a pool this path does not draw from) keep
        the forked path. The two paths read the same tokens, but not through the same kernels -- the paged fork's
        branch read goes through `unified_attention` against the page pool rather than the one-pass read's single
        FlashAttention-2 call over context and question together (`prismyra/kernels/qwen3_moe.py`'s
        `FlashAttention.forward`) -- so their answers agree only to a measured bound, not bit-for-bit:
        `PAGED_VS_JOINED_MOVEMENT` in the device tests, for this one-question case specifically (`tools/
        audit_sm120.py`'s `paged_vs_joined` section measures it directly, both supported cards). `open_context(...)
        .ask(...)` with one question still forks, and so does `ask()` itself once `self.paged` is `True` or there is
        more than one question -- that second case is measurably worse (`docs/PERFORMANCE.md`) and not bounded by
        the constant above.
        """
        self.validate(questions)
        if len(questions) == 1 and not images and not videos and self.calibration is None and not self.paged:
            return self._ask_in_one_pass(context, questions[0])
        group = self._group_for(len(questions)) if self.wide_group else None
        # The same restrictions `_ask_in_one_pass` and `open_batch` already state for the same
        # reasons -- media move the positions during the read, calibration measures its priors through branch
        # passes of its own shape, and the paged storage's pages belong to a pool this path does not draw from.
        if self.interleaved_fork and not images and not videos and self.calibration is None and not self.paged:
            # At the default `self.group` (32), 64 questions pack into exactly two 32-row
            # groups -- the first takes the fused path below, the second still falls back to `self._branch`
            # (`_ask_interleaved`'s own docstring). Widening to one 64-row group here puts every row through the
            # fused path instead of half of them. Deliberately **not** the general `self.wide_group` switch
            # above, kept as its own narrower condition rather than merged into it, even though both now verify
            # bit-identical: `self.wide_group` also covers 33-63 questions, a range a `torch.equal` gate first
            # measured *not* bit-identical -- that mismatch's real cause (the
            # GDN layer's causal convolution falling back to a different implementation whenever a branch
            # pass's row count was not exactly 1, `prismyra/kernels/qwen3_moe.py`'s `_install_conv`) was closed
            # by a branch-pass convolution batch fix, independently of this flag, and 64-exact was simply
            # the one width this gate happened to measure before that fix landed (33-64
            # re-verified bit-identical on both cards, with and without the batch-invariance
            # claim). torch.equal-gated against the two-32-row-group path before being wired in here.
            #
            # Widening every one of 36 layers' GDN state buffers to 64 rows costs more than fusing the
            # second group saves once the context itself, not the restream, dominates a pass -- measured
            # this as a win at ~5,016 tokens (-9.6%) and a loss at ~20,064 (+6.0%). `_ask_interleaved` now
            # decides whether to widen *after* it has `encoded.tokens` (`INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT`,
            # measured crossover ~8,669 tokens), rather than this call site guessing blind before the context is
            # even tokenized. A caller that asks 64 questions about a long document still gets the fused path --
            # just at `self.group` (32) rather than widened to 64 -- because the non-widened fused path never
            # regressed at any length tested (`tools/s4c_16q_length_check.py`, down to -3.19% at
            # 20,064 tokens; measured for a 64-question non-widened document: -0.7% at long context),
            # which is strictly better than falling all the way back to the two-pass baseline.
            return self._ask_interleaved(context, questions, group)
        with self._lock, self.open_context(context, images=images, videos=videos, group=group) as opened:
            assert opened._prefill is not None
            answered = self._answer(opened._prefill, questions, opened.tokens, context_ms=opened.context_ms)
        return answered

    def _group_for(self, n_questions: int) -> int | None:
        """`None` (keep `self.group`) unless this request needs more rows in one pass than `self.group` gives it.

        A document asked more than `WIDE_GROUP_FROM` questions pays for
        `ceil(n_questions / self.group)` branch passes under the engine's own default (32); widening to
        `WIDE_GROUP` (64) turns the common 33-64 question case into one pass instead of two, which removes one
        whole re-stream of the routed expert weights. Only ever widens, never narrows: a caller who built this
        engine with `group=128` already gets one pass up to 128 rows and this must not shrink that back to 64.
        """
        if n_questions > WIDE_GROUP_FROM and self.group < WIDE_GROUP:
            return WIDE_GROUP
        return None

    def _ask_interleaved(self, context: str, questions: list[Question], group: int | None) -> Result:
        """`ask()`'s path for the context and the first branch group through one layer-interleaved pass
        (`interleave.read_and_branch`) instead of a separate `open_context` read and `_branch` pass. See that
        module's docstring for what is fused and why; `torch.equal` and interleaved
        speed measurements confirmed this before `interleaved_fork=True` was wired in here.

        A second (or further) group of questions -- more than `group` of them -- runs through `self._branch`,
        unmodified, from the `Prefill` the interleaved pass returns; see `interleave.read_and_branch`'s own
        docstring for why that `Prefill`'s snapshot is interchangeable with one `_read` would have taken.

        **The exactly-64-questions widen decision is made here, not at the call site**, because only
        here is `encoded.tokens` known without tokenizing the context twice. See
        `INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT`'s own comment for the length sweep this threshold comes from and why
        a long document keeps the (never-regressed) non-widened fused path at `self.group` instead of losing the
        fusion altogether. This only ever widens past what `group` (`self.wide_group`'s own, separately-gated
        33-64 mechanism, still off by default) already asked for -- a caller running both `wide_group` and
        `interleaved_fork` together already gets 64 from `group` itself before this method is reached, and nothing
        here narrows that back down; this length guard is scoped to the width `interleaved_fork` decides
        on its own, not to the two features used together, a combination that has not been measured.

        Admission is not updated from this path's own peak (unlike `_answer`'s `_observe_peak` and
        `open_context`'s `_observe_reading`): the peak this pass reaches is the context's read and the first
        branch group's answer at once, which is not the shape either of those two counters means to describe,
        and feeding it to either would mis-calibrate admission for the ordinary two-pass path too. `_check_fits`
        below still runs, from whatever either counter already holds -- this path does not admit anything the
        two-pass path's own figures would have refused, it just does not sharpen them.
        """
        if not context.strip():
            raise PrismyraError("a context cannot be empty")
        encoded = encode(context, None, None, self.processor, self.tokenizer, self.device)
        self._check_fits(encoded.tokens)
        effective_group = group or self.group
        if (
            len(questions) == WIDE_GROUP
            and effective_group < WIDE_GROUP
            and encoded.tokens < INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT
        ):
            effective_group = WIDE_GROUP
        plans = [plan(q, self.tokenizer) for q in questions]
        token_ids = [p.token_ids for p in plans]
        width = self._width_for(plans)
        groups = self._packed_groups(plans, width, group=effective_group)
        first_members, first_width = groups[0]
        first_texts = [plans[i].text for i in first_members]
        first_padded = _round_rows(len(first_members), effective_group)

        start = _now(self.torch_device)
        with self._lock, torch.inference_mode():
            try:
                by_index: dict[int, torch.Tensor] = {}
                hidden0, prefill = interleave.read_and_branch(
                    self, encoded, first_texts, width=first_width, group=effective_group, padded_rows=first_padded
                )
                scored0 = self.heads.apply(
                    hidden0,
                    [questions[i].options for i in first_members],
                    score(hidden0, self.unembedding, [token_ids[i] for i in first_members], None),
                )
                by_index.update(zip(first_members, scored0, strict=True))
                for n, (members, group_width) in enumerate(groups[1:], start=1):
                    chunk = [plans[i].text for i in members]
                    padded = _round_rows(len(members), prefill.group or self.group)
                    same_shape_left = sum(
                        1
                        for m, w in groups[n + 1 :]
                        if _round_rows(len(m), prefill.group or self.group) == padded and w == group_width
                    )
                    hidden = self._branch(prefill, chunk, len(chunk), group_width, remaining=same_shape_left)
                    scored = self.heads.apply(
                        hidden,
                        [questions[i].options for i in members],
                        score(hidden, self.unembedding, [token_ids[i] for i in members], None),
                    )
                    by_index.update(zip(members, scored, strict=True))
                probabilities = [by_index[i] for i in range(len(questions))]
            except torch.OutOfMemoryError as e:
                raise PrismyraError(
                    f"ran out of memory answering {len(questions)} questions about {encoded.tokens} context "
                    f"tokens at group={effective_group} (interleaved_fork). Ask fewer questions at a time, or "
                    f"build the engine with a smaller group; the context itself is held once and is not what "
                    f"grew."
                ) from e
        readout_ms = _since(start, self.torch_device)

        answers = {
            q.id: _answer_for(q, p.tolist(), self.heads.name_for(q.options))
            for q, p in zip(questions, probabilities, strict=True)
        }
        return Result(
            answers=answers,
            model=self.model_name,
            context_tokens=encoded.tokens,
            scoring="raw",
            timing=Timing(context_ms=0.0, readout_ms=readout_ms),
        )

    def _shelf_ask_interleaved(self, shelf, context: str, questions: list[Question]) -> tuple[Result, int, "Shelved"]:
        """`Batcher._answer`'s path for one *fresh* document, read into `shelf` and answered in the
        same layer-interleaved pass instead of `Shelf.put_many` followed later by `Shelf.ask`. See
        `interleave.read_and_branch_shelf` for what the paged cache needed that `_ask_interleaved`'s joined-cache
        version did not; a `torch.equal` gate confirmed this before it was
        wired into `schedule.Batcher._answer`.

        Scoped by the caller to exactly one document whose own questions already fit one group (every request
        `Batcher` admits does, by `schedule.Limits.questions`) -- there is no second group to fall back to
        `Shelf.ask` for here the way `_ask_interleaved` falls back to `self._branch`, so this returns a finished
        `Result` plus the `(handle, Shelved)` pair the caller stores on the shelf, rather than a `Prefill` a
        second call would still need.

        `self.wide_group`: the same 33-`WIDE_GROUP`-question, under-`INTERLEAVE_WIDE_GROUP_
        TOKEN_LIMIT`-tokens widen `_ask_interleaved` already applies to the joined-cache path, applied here for
        the shelf one. Tokenising once more to make that decision (`self.encode_context`, cheap next to the
        forward pass this guards) rather than widening unconditionally: widening measured to cost more
        than it saves past that token threshold on the *joined* path, and nothing about the shelf's own paged
        pool changes that -- the extra GDN state a wider pass carries scales with context length the same way
        either way. `shelf` must already have been opened with `group=WIDE_GROUP` room in its own paged pool
        for this to be more than a decision with nowhere to act on it -- `schedule.Batcher` is the caller that
        opens it that way when `self.wide_group`, see `Batcher._on_shelf`.
        """
        plans = [plan(q, self.tokenizer) for q in questions]
        token_ids = [p.token_ids for p in plans]
        width = self._width_for(plans)
        group = self.group
        if self.wide_group:
            widened = self._group_for(len(questions))
            if widened is not None and self.encode_context(context).tokens < INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT:
                group = widened
        padded_rows = _round_rows(len(questions), group)
        texts = [p.text for p in plans]

        start = _now(self.torch_device)
        with self._lock, torch.inference_mode():
            try:
                hidden, handle, shelved = interleave.read_and_branch_shelf(
                    self, shelf, context, texts, width=width, padded_rows=padded_rows
                )
                scored = self.heads.apply(hidden, [q.options for q in questions], score(hidden, self.unembedding, token_ids, None))
            except torch.OutOfMemoryError as e:
                raise PrismyraError(
                    f"ran out of memory reading and answering {len(questions)} questions about a fresh document "
                    f"on a shelf (interleaved_fork). Ask fewer questions at a time, or build the engine with a "
                    f"smaller group."
                ) from e
        readout_ms = _since(start, self.torch_device)

        answers = {
            q.id: _answer_for(q, p.tolist(), self.heads.name_for(q.options))
            for q, p in zip(questions, scored, strict=True)
        }
        result = Result(
            answers=answers,
            model=self.model_name,
            context_tokens=shelved.tokens,
            scoring="raw",
            timing=Timing(context_ms=0.0, readout_ms=readout_ms),
        )
        return result, handle, shelved

    def _shelf_ask_interleaved_many(
        self, shelf, contexts: list[str], questions_per_doc: list[list[Question]]
    ) -> list[tuple[Result, int, "Shelved"]]:
        """`_shelf_ask_interleaved`'s own job for several *fresh* documents at once: every
        document in `formed.jobs` is fresh (`schedule.Batcher._answer`'s own generalised fusion condition --
        see that function), so one layer-interleaved pass reads and answers all of them, instead of diluting
        across `len(fresh)` separate single-document fused passes or falling back to `Shelf.put_many`+
        `Shelf.ask`. See `interleave.read_and_branch_shelf_many` for what changed to carry `N` documents
        instead of one; a `torch.equal` gate confirmed this.

        Each document's own `_round_rows(len(questions), group)` is computed here, independently, before the
        fused call -- not recombined with any other document's count inside it (`interleave.
        read_and_branch_shelf_many`'s own docstring is why that order matters). `group` is `self.group` unless
        `self.wide_group` widens *that one document's own* count the same way `_shelf_ask_interleaved` does --
        a mixed bin can have some documents at `self.group` and one at `WIDE_GROUP`, which is safe
        for the same reason the per-document independence already is: nothing here depends on a companion's
        own count, widened or not.
        """
        plans_per_doc = [[plan(q, self.tokenizer) for q in qs] for qs in questions_per_doc]
        width = self._width_for([p for plans in plans_per_doc for p in plans])
        groups_per_doc = []
        for context, plans in zip(contexts, plans_per_doc, strict=True):
            group = self.group
            if self.wide_group:
                widened = self._group_for(len(plans))
                if widened is not None and self.encode_context(context).tokens < INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT:
                    group = widened
            groups_per_doc.append(group)
        padded_rows_per_doc = [
            _round_rows(len(plans), group) for plans, group in zip(plans_per_doc, groups_per_doc, strict=True)
        ]
        texts_per_doc = [[p.text for p in plans] for plans in plans_per_doc]

        start = _now(self.torch_device)
        with self._lock, torch.inference_mode():
            try:
                hidden, handles, shelved_list = interleave.read_and_branch_shelf_many(
                    self, shelf, contexts, texts_per_doc, width=width, padded_rows_per_doc=padded_rows_per_doc
                )
                token_ids = [p.token_ids for plans in plans_per_doc for p in plans]
                options = [q.options for qs in questions_per_doc for q in qs]
                scored = self.heads.apply(hidden, options, score(hidden, self.unembedding, token_ids, None))
            except torch.OutOfMemoryError as e:
                raise PrismyraError(
                    f"ran out of memory reading and answering {len(contexts)} fresh documents in one "
                    f"layer-interleaved pass (interleaved_fork). Ask about fewer documents at a time."
                ) from e
        readout_ms = _since(start, self.torch_device)

        results = []
        at = 0
        for questions, shelved in zip(questions_per_doc, shelved_list, strict=True):
            answers = {
                q.id: _answer_for(q, p.tolist(), self.heads.name_for(q.options))
                for q, p in zip(questions, scored[at : at + len(questions)], strict=True)
            }
            at += len(questions)
            results.append(
                Result(
                    answers=answers,
                    model=self.model_name,
                    context_tokens=shelved.tokens,
                    scoring="raw",
                    timing=Timing(context_ms=0.0, readout_ms=readout_ms),
                )
            )
        return list(zip(results, handles, shelved_list, strict=True))

    def _ask_in_one_pass(self, context: str, question: Question) -> Result:
        """The context and the question as one sequence, read once, answered at its last position.

        The same tokens at the same positions as the forked path -- the context as `encode` tokenises it, then the
        question exactly as `build_suffixes` renders a branch -- through the same read the context pass uses, so the
        kernels are the ones a context read runs. What is skipped is the fork: no snapshot, no widening to the group,
        no branch pass. The same refusals apply: an empty context, a question wider than a branch may be, and a read
        that admission says will not fit.
        """
        if not context.strip():
            raise PrismyraError("a context cannot be empty")
        planned = plan(question, self.tokenizer)
        self._width_for([planned])  # the forked path's refusal of an over-wide question, so both paths refuse alike
        suffix = branch_ids(planned.text, self.tokenizer)
        with self._lock:
            encoded = encode(context, None, None, self.processor, self.tokenizer, self.device)
            tokens = encoded.tokens + len(suffix)
            bucket = self._one_pass.bucket_for(tokens) if self._one_pass is not None else None
            if bucket is None:
                # A replay needs no admission: everything it touches was allocated when it was recorded.
                self._check_fits(tokens)
            start = _now(self.torch_device)
            before = None if bucket is not None else self._peak_baseline()
            try:
                with torch.inference_mode():
                    ids = torch.cat([encoded.input_ids, torch.tensor([suffix], device=self.device)], dim=1)
                    if bucket is not None:
                        hidden = onepass.replay(bucket, ids, self._pad_id())
                    else:
                        hidden = self._read_one_pass(ids)
                    (probabilities,) = self.heads.apply(
                        hidden, [question.options], score(hidden, self.unembedding, [planned.token_ids], None)
                    )
                    values = probabilities.tolist()
                    del hidden, ids
            except torch.OutOfMemoryError as e:
                raise PrismyraError(
                    f"ran out of memory reading a context of {encoded.tokens} tokens with its question; a shorter "
                    f"context is the only knob"
                ) from e
            if before is not None:
                self._observe_reading(before, tokens)
            elapsed = _since(start, self.torch_device)
        return Result(
            answers={question.id: _answer_for(question, values, self.heads.name_for(question.options))},
            model=self.model_name,
            context_tokens=encoded.tokens,
            scoring="raw",
            # The question is read inside the context pass, so the whole request is `context_ms`.
            timing=Timing(context_ms=elapsed, readout_ms=0.0),
        )

    def _read_one_pass(self, ids: torch.Tensor) -> torch.Tensor:
        """The eager one-pass read: one row of context and question, the hidden state at its last token.

        Its own method because two things must run exactly this: a request no recording holds, and the proof each
        recording is held to (`onepass.prove`), which compares a replay against it.
        """
        tokens = ids.shape[1]
        # One row. The branch room is kept at its usual size rather than zero, because the attention layer sizes its
        # context room as the total minus the branch room; at one row it is a few megabytes.
        cache = build_cache(self.config, self.room_for(tokens) + WIDTHS[-1], 1, self.dtype, self.device, WIDTHS[-1])
        with torch.inference_mode():
            out = self.backbone(input_ids=ids, use_cache=True, past_key_values=cache)
            hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[0, -1:]
        # Released before returning, inside the caller's lock: the output holds the cache, and the next request must
        # not find this one's memory still allocated when it is admitted.
        del out, cache
        return hidden

    def _pad_id(self) -> int:
        """What a recorded one-pass read is padded with. Any token will do -- nothing after the last real token reaches
        it, and each bucket is proved on that -- so the tokenizer's own pad token, or the end of text without one."""
        for name in ("pad_token_id", "eos_token_id"):
            value = getattr(self.tokenizer, name, None)
            if isinstance(value, int):
                return value
        return 0

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
            "autotune": self.autotune.as_dict(),
            # `{"source": None, "pinned": None}` when this process never converted an NVFP4 MoE layer (the usual
            # case on fp8-36l); otherwise which of PRISMYRA_NVFP4_TACTICS/the bundled per-card table/a fresh,
            # this-process-only timing search chose this process's GEMM tactic. See `kernels/nvfp4.py`'s
            # `autotune_tactics()`.
            "nvfp4_tactics": _nvfp4_tactics_status(),
            "scoring": self.calibration.mode if self.calibration else "raw",
            "storage": "paged" if self.paged else "joined",
            # Counted by the layers themselves rather than taken from the flag. A previous version of the paged path
            # reported itself as installed and never ran, and this is the number that would have said so.
            "paged_reads_served": self._paged_reads(),
            "graphs": self.graphs,
            "graphs_declined": dict(self.declined_recordings),
            # Only the economics declines still eligible for a second attempt -- the bar `expected` has to clear.
            # Shrinks as shapes are reopened and either kept or declined again under a fresh measurement.
            "graphs_retry_at": dict(self._economics_needed),
            "graphs_verified": dict(self.verified_recordings),
            "graphs_replays": dict(self.replays),
            "graphs_cost": dict(self.replay_cost),
            "graphs_skip_reasons": dict(self._skip_reasons),
            # The one-pass recordings: which buckets serve, how often each has, what proving each measured and what
            # they hold. Empty when the one-pass read runs eagerly.
            "short_graphs": self._one_pass.stats() if self._one_pass is not None else {},
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
            "reading_bytes_floor": self._observed_reading_floor or 0,
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
        # Less what the one-pass recordings hold: their private pool is reserved and mostly unallocated between
        # replays, and none of it can be handed to a read. The figure is the whole growth of the reservation while they
        # were taken, so it also counts tensors already outside `spare`; refusing a little early is the safe direction.
        if self._one_pass is not None:
            spare -= self._one_pass.held_bytes
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

    def _read(self, encoded, group: int | None = None) -> Prefill:
        # A solo read and `open_batch`'s joint read of several documents go through the *same* borrowed chunked
        # recurrent kernel, and that kernel's own configuration is chosen by the *total* length of the varlen run
        # it is given -- not by any one document's content in it (measured decisively:
        # two companions of the identical length but different content left a
        # target's extracted state bit-identical; the same target alone, at a different total length, did not).
        # So a document read alone and the same document read alongside others can legitimately end up at two
        # different total lengths, and therefore two different -- but each internally consistent -- recurrent
        # states, which is exactly the open_batch/`ask()` mismatch this engine's device tests measure against.
        # Padding every read, solo or batched, up to the same small set of total-length buckets (`_round_rows`,
        # reused rather than duplicated: "round up to the next power of two, capped" is the identical decision for
        # a row count and for a token count) removes the difference instead of chasing it. Only when the borrowed
        # kernels this depends on are installed, the engine is paged (unpaged storage has no per-document state to
        # keep separate in the first place) and there is no media (whose positions a flat multi-segment run has not
        # been taught to carry, same restriction `open_batch` already states).
        if self.paged and not encoded.has_media and not self._missing_batched_read_kernels():
            pad_ids, lengths = self._pad_context_lengths([encoded.tokens], [encoded.input_ids])
            if pad_ids.shape[1] > 0:
                return self._read_padded(encoded, pad_ids, lengths)
        cache, room = self._claim_cache(encoded.tokens, group=group)
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
        effective_group = group or self.group
        restore_and_fork(cache, taken, effective_group, width=effective_group)
        return Prefill(
            snapshot=taken,
            cache=cache,
            room=room,
            tokens=encoded.tokens,
            last_position=torch.tensor([encoded.tokens - 1], device=self.device),
            position_from=position_from,
            group=effective_group,
        )

    def _missing_batched_read_kernels(self) -> set[str]:
        """Which of the two borrowed replacements a batched (multi-segment, `cu_seqlens`) read depends on are not
        installed, empty when both are. Shared by `open_batch` and `_read`'s padding, which depend on the same
        thing for the same reason: the framework's own gated-delta-rule and convolution have no argument for where
        one document ends inside a flat run.
        """
        return {"convolution", "gated_delta_rule"} - {swap.name for swap in self.applied.swaps}

    def _pad_context_lengths(self, lengths: list[int], ids: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        """Round a varlen read's total length up to the bucket `_round_rows` would pick, as one more segment appended
        after the real documents, and the padding ids to fill it -- a harmless repeat of the last document's own
        tokens, because the measurement behind this is that a chunked recurrent kernel's config is chosen by *total*
        length and does not care what the padding is.

        Returns the padding ids (shape `(1, 0)`, not `(1, pad)` carrying nothing, when the total is already at a
        bucket) and `lengths` with the pad segment appended only when there is one -- `varlen.reading` refuses a
        zero-length document, and a run that is already at a bucket has nothing to add.
        """
        total = sum(lengths)
        padded = _round_rows(total, self.longest_context)
        pad = max(0, padded - total)
        if pad == 0:
            return ids[0].new_zeros((1, 0)), lengths
        last = ids[-1]
        reps = -(-pad // last.shape[1])
        pad_ids = last.repeat(1, reps)[:, :pad]
        return pad_ids, [*lengths, pad]

    def _read_padded(self, encoded, pad_ids: torch.Tensor, lengths: list[int]) -> Prefill:
        """`_read`'s single-document path, through the same joint-read machinery `open_batch` uses for several --
        one real document and one padding segment, so the kernel sees the same total length a later `open_batch`
        sharing this document would round it to. Only the real document's row of the result is kept; the padding's
        is discarded exactly as a padded branch pass already discards its extra rows (`_round_rows`'s own note).
        """
        cache, room = self._claim_cache(sum(lengths))
        ids = torch.cat([encoded.input_ids, pad_ids], dim=1)
        with torch.inference_mode():
            for layer in cache.layers:
                begin = getattr(layer, "begin_documents", None)
                if begin is not None:
                    begin([0, 1])  # the real document, then its padding -- one handle each, or `_write_context` refuses
            with varlen.reading(lengths, self.device) as boundaries:
                self.backbone(input_ids=ids, position_ids=boundaries.positions(self.device), use_cache=True, past_key_values=cache)
                _put_back_conv_states(cache, boundaries)
                self._check_batched_read(cache, boundaries)
            taken = pick(snapshot(cache), 0)
            for layer in cache.layers:
                release = getattr(layer, "release_document", None)
                if release is not None:
                    release(1)
        # Not `restore_and_fork`: its `begin_branches()` defaults every row to "the last document written" (its own
        # docstring), which was always correct when the only document ever written to a solo cache was the real one
        # -- here the padding was written after it. `rows_for=[0] * self.group` says the same thing this cache's
        # only remaining document already implies, explicitly rather than by relying on write order.
        restore_and_fork_many(cache, [(taken, self.group)], width=self.group, rows_for=[0] * self.group)
        return Prefill(
            snapshot=taken,
            cache=cache,
            room=room,
            tokens=encoded.tokens,
            last_position=torch.tensor([encoded.tokens - 1], device=self.device),
            position_from=encoded.tokens,
        )

    @property
    def longest_context(self) -> int:
        """The largest context a cache is built for. What a scheduler needs to know before it assembles a batch."""
        return CONTEXT_SIZES[-1]

    def _note_read(self, ms: float, documents: int) -> None:
        """Remember the fastest read, in per-document terms so a batch and a single read are comparable."""
        each = ms / max(1, documents)
        if self.fastest_read_ms is None or each < self.fastest_read_ms:
            self.fastest_read_ms = each

    def encode_context(self, context: str) -> Encoded:
        """Tokenise a document without reading it, so a caller can do that work off the request path.

        The scheduler does: it encodes on the caller's thread when a request is submitted, and hands the result to
        `open_batch`. Without this the scheduler tokenised each document twice on the one thread that owns the device --
        once to count its tokens while forming a pass and once inside the read -- and both were on the critical path.
        """
        return encode(context, None, None, self.processor, self.tokenizer, self.device)

    def open_shelf(self, room: int | None = None, lane: int = 0, group: int | None = None) -> Shelf:
        """One cache held open, with documents put on it and taken off as callers come and go.

        A `Batch` reads its documents, answers them and drops the cache, so asking twice about one document reads it
        twice -- and page reuse has nothing to reuse pages for, since nothing outlives a batch. A shelf is the other
        shape: the pages stay, a document stays until it is dropped, and a second question about a document already on
        the shelf costs a branch pass and no read at all.

        `room` is how many context tokens the shelf holds altogether, rounded up to a bucket. Default is the largest
        bucket that admission will accept, because a shelf that holds two documents is barely a shelf.

        `lane` is which engine lane this shelf's own passes run under (see `Shelf.lane`). Each call builds a brand
        new cache (`_claim_cache` has no pool to reuse from yet), so two shelves -- one per lane -- never share a
        `Pool`/page table; the only thing two lanes still share is the model weights (read-only) and, if `lane`
        differs, nothing else at all. Measured directly: a second
        shelf is not free -- opening one at `room=4096` plus 5 documents on each cost about 2.1 GiB total from a
        freshly loaded model's 8.6 GiB of free device memory -- so a second lane's `room` should be set with that
        in mind rather than left at the default (which is sized for *one* shelf being the only one).

        `group` (combining `wide_group` with the Shelf/Batcher path): overrides
        `self.group` for *this shelf's own* paged attention pool capacity, the same override `_claim_cache`
        already gives the joined-cache path. Built in at open time because the pool's page table is sized once,
        not per call -- a document later asked up to `WIDE_GROUP` questions needs the pool to have room for that
        many branch rows from the start, or `begin_branches` refuses the same way admitting more rows than a
        pool holds always has (`"N rows asked for and this pool holds M"`, the crash this
        capacity-guard fix exists to keep out of the *fused* path; this is the same ceiling for the shelf's own
        paged pool underneath it). `None` keeps today's single behaviour (`self.group`).
        """
        if not self.paged:
            raise PrismyraError(
                "a shelf needs the paged storage: documents share the pages and each row's table names its own. Build "
                "the engine with Prismyra(..., paged=True)."
            )
        wanted = room if room is not None else self._largest_shelf()
        self._check_fits(wanted)
        cache, held = self._claim_cache(wanted, group=group)
        return Shelf(_engine=self, _cache=cache, room=held, lane=lane)

    def _largest_shelf(self) -> int:
        """The biggest bucket this engine can hold a shelf of, from its own admission figures.

        Asked rather than assumed, because the answer is a fact about the card and the weights on it. Falls back to the
        smallest bucket, which admission will then refuse by name if even that does not fit -- a refusal naming the
        figures beats a shelf that appears to exist and fails on its first document.
        """
        for size in reversed(CONTEXT_SIZES):
            try:
                self._check_fits(size)
            except PrismyraError:
                continue
            return size
        return CONTEXT_SIZES[0]

    def open_batch(self, contexts: list[str] | list[Encoded]) -> Batch:
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
        if missing := self._missing_batched_read_kernels():
            raise PrismyraError(
                f"a batch of documents needs the borrowed {' and '.join(sorted(missing))}: the framework's own has no "
                "argument for where one document ends, so a batched read would scan across the boundary and answer "
                f"about a document that was never written. What the adapter reported: "
                f"{self.applied.skipped or self.applied.summary()}."
            )
        encoded = [one if isinstance(one, Encoded) else self.encode_context(one) for one in contexts]
        if any(one.has_media for one in encoded):
            raise PrismyraError(
                "a batch of documents is text only for now: media widen a context's positions by a grid rather than by "
                "a token count, and a flat run of several would need each document's own offset threaded through."
            )
        lengths = [one.tokens for one in encoded]
        total = sum(lengths)
        pad_ids, lengths = self._pad_context_lengths(lengths, [one.input_ids for one in encoded])
        cache, room = self._claim_cache(total + pad_ids.shape[1])
        started = _now(self.torch_device)
        with torch.inference_mode():
            # One pass over all of them. Reading is 110 ms of fixed cost plus 11 ms per thousand tokens on this
            # model, so what this removes is that fixed cost paid per document rather than per batch.
            # Named even though a batch's handles are its positions, so the layer's "already held" check runs rather
            # than being skipped on the one path that could get away with skipping it.
            # The padding is one more document to the pages, same as to the recurrence: `begin_documents` wants one
            # handle per entry in `lengths`, pages and all, or the paged attention layer refuses the read outright
            # (`_write_context`'s own check). `len(encoded)` is free because nothing is held in a batch's fresh cache
            # yet.
            pad_handle = len(encoded)
            has_pad = pad_ids.shape[1] > 0
            handles = list(range(len(encoded))) + ([pad_handle] if has_pad else [])
            for layer in cache.layers:
                begin = getattr(layer, "begin_documents", None)
                if begin is not None:
                    begin(handles)
            ids = torch.cat([*(one.input_ids for one in encoded), pad_ids], dim=1)
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
            if has_pad:
                # The padding answers nothing and keeps no row, so its pages go back now rather than sitting in this
                # batch's cache until it closes.
                for layer in cache.layers:
                    release = getattr(layer, "release_document", None)
                    if release is not None:
                        release(pad_handle)
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
        context_ms = _since(started, self.torch_device)
        self._note_read(context_ms, len(contexts))
        return Batch(_engine=self, _prefills=prefills, _room=room, context_ms=context_ms)

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

    def _answer_batch(
        self,
        prefills: list[Prefill],
        asked: list[list[Question]],
        context_ms: float,
        rows_for: list[int] | None = None,
        lane: int = 0,
    ) -> list[Result]:
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
        # The real check is on the *padded* total, not this one -- `_branch_across` now
        # pads each document to its own bucket before summing (see there), which can need more rows than the real
        # total alone would. Checked here too, before any tokenising, so a batch that cannot fit fails with one
        # clear reason instead of a confusing one from deeper in the pass.
        padded_total = sum(_round_rows(c, self.group) for c in counts)
        if padded_total > self.group:
            raise PrismyraError(
                f"{sum(counts)} questions across {len(asked)} documents round up to {padded_total} rows once each "
                f"document is padded to its own bucket independently of its companions, and a group of {self.group} "
                f"cannot hold that many. Ask fewer documents or questions together, or build the engine with a "
                f"larger group; the independent-padding rule is what makes two documents answered together give "
                f"the same reduction order each one would alone (see `_round_rows`)."
            )

        flat = [q for questions in asked for q in questions]
        plans = [plan(q, self.tokenizer) for q in flat]
        # Tightened the same way `_packed_groups` tightens a single document's groups: a batch answering several
        # documents in one pass is exactly one group of that function's own kind, every row in it. Using the
        # untightened WIDTHS bucket here instead -- which is what this line did before -- pads every row wider than
        # `ask()` would and is not a rounding difference: the extra padding columns reach the recurrent layers'
        # kernels and move the hidden state at the real last token, not just the padding's own. Confirmed by
        # bisecting shelf against `ask()` on the same document, same questions, same order: matching only this
        # width made the two bit-exact; matching only the question order changed nothing. See
        # tests/test_gpu.py::test_a_shelf_matches_ask_bit_for_bit.
        width = _round_pack_align(max(len(branch_ids(p.text, self.tokenizer)) for p in plans), self._width_for(plans))
        # Which document each row answers about, named by the handle the **cache** knows it as. A batch admits its
        # documents in order so the handles are the positions; a shelf holds whatever was put on it, which is why this
        # is given rather than derived.
        names = rows_for if rows_for is not None else list(range(len(counts)))
        if len(names) != len(counts):
            raise PrismyraError(f"{len(names)} document names for {len(counts)} lists of questions")
        rows_for = [names[at] for at, count in enumerate(counts) for _ in range(count)]

        start = _now(self.torch_device)
        with self._lock_for(lane), self._stream_for(lane), torch.inference_mode():
            try:
                hidden = self._branch_across(prefills, counts, rows_for, [p.text for p in plans], width, lane=lane)
                read = score(hidden, self.unembedding, [p.token_ids for p in plans], None)
                probabilities = self.heads.apply(hidden, [q.options for q in flat], read)
            finally:
                # Whether it answered or raised, no row is set up to read anything now, so a document nobody is
                # reading can be dropped. In a `finally`: a failed pass must not leave a shelf unable to drop anything.
                for layer in prefills[0].cache.layers:
                    finish = getattr(layer, "finish_branches", None)
                    if finish is not None:
                        finish()
        readout_ms = _since(start, self.torch_device)

        results, at = [], 0
        for handle, questions in enumerate(asked):
            answers = {
                q.id: _answer_for(q, values.tolist(), self.heads.name_for(q.options))
                for q, values in zip(questions, probabilities[at : at + len(questions)], strict=True)
            }
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

    def _branch_across(
        self, prefills, counts: list[int], rows_for: list[int], texts: list[str], width: int, lane: int = 0
    ):
        """The pass itself. Every row's positions start at its own document's end, which is per row not per batch.

        Goes through `_run_recorded`, the same decision `_run_branch` uses for a single document. It did not used to:
        this called the backbone directly, so `graphs=True` recorded nothing here, which is the gap `docs/PERFORMANCE.md`
        names under "Recording the batched pass is not the next thing, and why". Wiring it is what that section says it
        would take -- a few lines -- once the other half, `graphs.Recording` accepting a remainder bucket instead of an
        exact context length, makes a recording survive the batch's documents changing between passes.
        """
        # Batch-invariant: see `_round_rows`. The row count is the only thing that otherwise differs between "two
        # documents answered together" and "either one answered alone", once the width is matched (`_round_pack_align`)
        # -- and that alone moved an answer by up to 0.29 and flipped decisions. Padding every pass sharing documents
        # to the same row count a solo document would be padded to removes the difference entirely: `pad` extra rows
        # repeat the first document, discarded at the end exactly as `build_suffixes` already discards padded columns.
        #
        # An earlier version of this: a single `padded_rows = _round_rows(rows, self.group)` rounded the
        # *combined* total, which closes the context-length axis (an 80-document
        # benchmark) but not the question-count one: the same two real questions about the same document land in
        # a 2-row pass alone and a 4-row pass once a one-question companion is added, even though neither document's
        # own rows changed -- `tests/test_gpu.py`'s documented 0.0128/0.024 residual, confirmed on sm_120 too
        # (`audit_sm120.py`). Padding *each document to its own bucket* first, then laying the
        # padded blocks end to end, makes a document's own width a function of its own row count alone -- a
        # document's padded width must not depend on which other documents share the batch with it.
        # `restore_and_fork_many` already
        # takes an arbitrary per-document row count in `parts` (it was only ever used with padding on the last
        # document because that is all `_branch_across` built), so nothing downstream of this needed to change to
        # accept it.
        padded_counts = [_round_rows(c, self.group) for c in counts]
        padded_rows = sum(padded_counts)

        # `texts` is flat and real-only, in document order (`_answer_batch`'s `flat`). Insert each document's own
        # pad entries right after its own real ones -- repeating that document's own first real text, the same
        # convention `build_suffixes` already uses for the single-document case, just scoped per document instead
        # of globally so a different document's padding can never be mistaken for this one's.
        padded_texts: list[str] = []
        real_row_at: list[int] = []  # index into `padded_texts`/the eventual padded rows for each real, flat row
        at = 0
        for count, padded in zip(counts, padded_counts):
            block = texts[at : at + count]
            real_row_at.extend(range(len(padded_texts), len(padded_texts) + count))
            padded_texts.extend(block)
            if padded > count:
                padded_texts.extend([block[0]] * (padded - count))
            at += count
        ids, read_at, _ = build_suffixes(padded_texts, self.tokenizer, self.device, padded_rows, width)

        # One start position per *padded* row: every row of a document, real or padding, starts at that document's
        # own context end -- there is one such position per document, not per question, so repeating it for a
        # document's pad rows is the same arithmetic as for its real ones, not a special case.
        starts = [
            prefills[at].position_from or prefills[at].tokens
            for at, padded in enumerate(padded_counts)
            for _ in range(padded)
        ]
        offsets = torch.tensor(starts, device=self.device).unsqueeze(1)
        positions = offsets + torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        cache = prefills[0].cache

        # Each document gets its own padded count directly -- no more "extend the last document's block", because
        # every document now carries its own padding rather than borrowing room at the tail.
        parts = [(prefills[at].snapshot, padded_counts[at]) for at in range(len(counts))]
        rows_for_padded: list[int] = []
        at = 0
        for count, padded in zip(counts, padded_counts):
            name = rows_for[at]  # every real row of one document already names the same document
            rows_for_padded.extend([name] * padded)
            at += count

        def fork() -> None:
            restore_and_fork_many(cache, parts, width=self.group, rows_for=rows_for_padded, lane=lane)

        def run(suffix: torch.Tensor, suffix_positions: torch.Tensor) -> torch.Tensor:
            # A branch pass, same reasoning as `_branch`'s `run` below.
            with varlen.branching():
                out = self.backbone(
                    input_ids=suffix, position_ids=suffix_positions, use_cache=True, past_key_values=cache
                )
            return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        # One document's remainder is all a paged recording's bucket key carries (see `_run_recorded`), so a pass
        # naming more than one document is not a shape graphs may generalise over yet.
        homogeneous = len(set(rows_for)) <= 1
        hidden = self._run_recorded(
            cache, fork, run, ids, positions, padded_rows, width, homogeneous=homogeneous, lane=lane
        )
        # Gather the real rows back out in the caller's original flat order -- they are no longer a contiguous
        # prefix now that each document's own padding sits right after its own real rows instead of at the tail.
        real_idx = torch.tensor(real_row_at, device=self.device)
        return hidden[torch.arange(padded_rows, device=self.device), read_at][real_idx]

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
        plans = [plan(q, self.tokenizer) for q in questions]
        token_ids = [p.token_ids for p in plans]
        width = self._width_for(plans)
        # This document's own branch-row capacity, set by `ask`'s `_group_for` at read time and
        # carried on `prefill` ever since -- `None` (every prefill from before this field existed, and every one
        # `open_batch` still builds) means "use `self.group`, as always".
        groups = self._packed_groups(plans, width, group=prefill.group)

        # Before the clock starts, and outside the lock's timed section: a prior is cached per question, so charging
        # the first request for every later one's correction would report a cost that is not there.
        priors = self.calibration.priors(self, questions, plans) if self.calibration else None

        start = _now(self.torch_device)
        probabilities: list[torch.Tensor] = []
        widest_chunk = 0
        with self._lock, torch.inference_mode():
            before = self._peak_baseline()
            try:
                by_index: dict[int, torch.Tensor] = {}
                for n, (members, group_width) in enumerate(groups):
                    chunk = [plans[i].text for i in members]
                    widest_chunk = max(widest_chunk, len(chunk))
                    # How many more groups of this exact shape this call will run, which is what decides whether
                    # recording the pass can pay for itself. Compared on the padded row count `_branch` actually
                    # runs at (`_round_rows`), not the raw member count: two groups of 5 and 7 real questions run
                    # the identical padded-to-8 pass now, so they are the same shape for this count too.
                    padded = _round_rows(len(members), prefill.group or self.group)
                    same_shape_left = sum(
                        1
                        for m, w in groups[n + 1 :]
                        if _round_rows(len(m), prefill.group or self.group) == padded and w == group_width
                    )
                    hidden = self._branch(prefill, chunk, len(chunk), group_width, remaining=same_shape_left)
                    scored = self.heads.apply(
                        hidden,
                        [questions[i].options for i in members],
                        score(
                            hidden,
                            self.unembedding,
                            [token_ids[i] for i in members],
                            [priors[i] for i in members] if priors else None,
                        ),
                    )
                    by_index.update(zip(members, scored, strict=True))
                probabilities = [by_index[i] for i in range(len(questions))]
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

        answers = {
            q.id: _answer_for(q, p.tolist(), self.heads.name_for(q.options))
            for q, p in zip(questions, probabilities, strict=True)
        }
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
        # The floor, from any read. A read has a constant part and dividing it by the tokens is what went wrong: at
        # 3,040 and 24,327 tokens the transient is 1.96e-4 GiB a token both times, so a constant is invisible there --
        # and at thirty tokens the same division said 20 MiB a token, refusing a 1,024-token context at 23 GiB.
        floor = self._observed_reading_floor
        self._observed_reading_floor = transient if floor is None else max(floor, transient)
        # The slope, only from reads long enough for the constant not to dominate. One bucket is the bar: below it the
        # division is a measurement of the constant divided by an arbitrary number.
        if context_tokens < CONTEXT_SIZES[0]:
            return
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
        widest = max(len(branch_ids(p.text, self.tokenizer)) for p in plans)
        try:
            return round_width(widest)
        except TooWide as e:
            raise PrismyraError(str(e)) from e

    def _packed_groups(self, plans: list, width: int, group: int | None = None) -> list[tuple[list[int], int]]:
        """Which questions share a branch pass, and how wide each pass is.

        A pass is as wide as its longest question, and every shorter row pays for the difference in padding that each
        layer computes and discards. With one width for the whole request -- the widest question, rounded to a bucket --
        questions of mixed length spend much of a branch pass on padding (docs/PERFORMANCE.md measures 53%). Sorting by
        length before grouping and sizing each group to its own longest row (to a multiple of `PACK_ALIGN`) leaves the
        questions in every group of similar length. Which rows travel together changes the reduction order, so a
        near-tie can move by as much as batching already allows; see `COMPANION_MOVEMENT` in the device tests.

        Only when nothing depends on the width being shared: the calibration measures its priors at the request's width,
        so `calibrate` keeps it. Recordings do not need it -- each group's (rows, width) is its own shape, and a caller
        asking the same questions again produces the same groups, so each shape recurs and is recorded on its own.
        """
        n = len(plans)
        group = group or self.group
        if self.calibration is not None:
            return [(list(range(lo, min(n, lo + group))), width) for lo in range(0, n, group)]
        lengths = [len(branch_ids(p.text, self.tokenizer)) for p in plans]
        order = sorted(range(n), key=lambda i: lengths[i])
        groups = []
        for lo in range(0, n, group):
            members = order[lo : lo + group]
            longest = max(lengths[i] for i in members)
            # The same rule whether recording is on or not. A recording must answer exactly as the eager pass it was
            # taken from, and a different width is a different reduction: rounding to the pinned buckets only under
            # `graphs` moved a probability by 3e-4 between the two.
            groups.append((members, _round_pack_align(longest, width)))
        return groups

    def _branch(self, prefill: Prefill, texts: list[str], rows: int, width: int, remaining: int = 0) -> torch.Tensor:
        # The snapshot is taken on the first branch, when the cache holds exactly the context, so restoring it also
        # puts every layer's token count back to the end of the context. One mechanism, not a state restore plus a
        # separate rewind: two of them can disagree, and the one that is wrong answers plausibly.
        if prefill.snapshot is None:
            prefill.snapshot = snapshot(prefill.cache)

        # Batch-invariant: see `_round_rows`. Every row buffer downstream is already sized for `prefill.group`
        # (`self.group` unless `ask`'s `_group_for` widened it for this document), so padding up to it costs
        # nothing to allocate -- only the padded rows' own compute, which `remaining` below also now measures
        # economics against at this padded shape rather than the raw one.
        padded_rows = _round_rows(rows, prefill.group or self.group)
        ids, read_at, _ = build_suffixes(texts, self.tokenizer, self.device, padded_rows, width)
        # From where the model thinks the context reached, which is past its token count when media widened it.
        start = prefill.position_from or prefill.tokens
        positions = torch.arange(start, start + ids.shape[1], device=self.device).expand(padded_rows, -1)

        def run(suffix: torch.Tensor, suffix_positions: torch.Tensor) -> torch.Tensor:
            """One branch pass, with the fork done by the caller.

            The fork is deliberately **not** in here, and that was found by measurement rather than reasoned out. With
            it inside, one replay disagreed with the eager pass and two replays disagreed with each other -- the
            signature of a buffer a recording both reads and writes without resetting. The layer rebinds its recurrent
            state to a new tensor on every pass, and a rebinding is Python: a recording keeps the tensor it saw and a
            replay cannot repeat the assignment. So the fork runs eagerly every time, at the cost of a few copies per
            layer, and the recording covers only what is pure device work.

            `suffix_positions` is a parameter rather than the closed-over `positions`, and a recording's own static
            buffer rather than a kept constant -- `graphs.record` says why: a different document starts its branch at
            a different position, and a recording that answered every document at the position its first one needed
            would be wrong rather than slow.

            Wrapped in `varlen.branching()` so the borrowed gated-delta-rule kernel is told
            `output_final_state=False` -- see that function's docstring for why nothing downstream of a branch
            ever reads the state it would otherwise write. A CUDA-graph capture of this call bakes in whichever
            kernel launches ran during capture, and capture always goes through this same `run`, so a replay gets
            the skip too without needing to know about it.
            """
            with varlen.branching():
                out = self.backbone(
                    input_ids=suffix, position_ids=suffix_positions, use_cache=True, past_key_values=prefill.cache
                )
            return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        hidden = self._run_branch(prefill, run, ids, padded_rows, width, positions, remaining)
        # Each row is read at its own last real token, which is why the padding cannot reach an answer.
        return hidden[torch.arange(padded_rows, device=self.device), read_at][: len(texts)]

    def _replay_disagreement_on(self, taken, cache, fork, ids, positions, reference) -> tuple[float, float]:
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
            taken.before_fork(cache)
            fork()
            replayed = taken.replay(cache, ids, positions)
            worst = max(worst, float((replayed.float() - reference.float()).abs().amax()))
        return worst, _since(started, self.torch_device) / REPLAY_CHECKS

    def _lock_for(self, lane: int) -> threading.RLock:
        """The lock a lane's whole pass is serialised under. Lane 0 is `self._lock` itself; a further lane gets its
        own, built once and kept -- see `self._locks`."""
        lock = self._locks.get(lane)
        if lock is None:
            lock = threading.RLock()
            self._locks[lane] = lock
        return lock

    def _stream_for(self, lane: int):
        """The CUDA stream a lane's pass runs on, as a context manager -- `contextlib.nullcontext()` for lane 0,
        which keeps using whatever stream was already current (today's single-lane behaviour, unchanged); a real
        `torch.cuda.stream(...)` for any further lane, built once. CPU, or no CUDA stream support needed, falls
        back to the same `nullcontext()` lane 0 uses."""
        import contextlib

        if lane == 0 or self.torch_device.type != "cuda":
            return contextlib.nullcontext()
        stream = self._streams.get(lane)
        if stream is None:
            stream = torch.cuda.Stream(device=self.torch_device)
            self._streams[lane] = stream
        return torch.cuda.stream(stream)

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

    def _claim_cache(self, tokens: int, group: int | None = None):
        """A cache sized for a bucket rather than for this context, reused if one is free.

        `group` overrides `self.group` for this one cache's branch-row capacity -- the
        caller already knows, before any pages exist, that this document's own branch passes will need more rows
        than the engine's construction-time default allows (see `ask`'s `_group_for`). `None` keeps today's single
        behaviour.

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
            self.config, room + WIDTHS[-1], group or self.group, self.dtype, self.device, WIDTHS[-1], paged=self.paged
        )
        self._made_caches += 1
        return cache, room

    def _forget_recordings(self, cache) -> None:
        """Drop the recordings taken on this cache. They hold its addresses, and the cache is about to go."""
        self._cache_recordings.pop(id(cache), None)

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
        # `prefill.group`: this document's own fork buffers were sized at read time for `prefill.group` rows
        # (`self.group` unless `ask`'s `_group_for` widened it) -- `width` must match whatever that was, or `_owned`
        # reallocates a new buffer here, inside the branch pass, which is exactly the "surprise allocation lands in
        # the wrong budget" failure `_read`'s own docstring describes for the read side.
        restore_and_fork(prefill.cache, prefill.snapshot, rows, width=prefill.group or self.group)

    def _run_branch(self, prefill, run, ids, rows: int, width: int, positions, remaining: int = 0):
        """The single-document branch pass, through the shared recording machinery. See `_run_recorded`."""
        return self._run_recorded(
            prefill.cache, lambda: self.fork(prefill, rows), run, ids, positions, rows, width, remaining
        )

    def _bucket_key(self, cache, rows: int, width: int) -> tuple:
        """The shape a recording is kept under: `(rows, width)`, widened with a remainder bucket when the cache is
        paged storage.

        The joined storage's read is shaped `(rows, context + suffix, heads, dim)`, so two contexts of different
        lengths are two different graphs and `(rows, width)` is already as coarse as it can be. The paged storage's
        read is shaped by the pool, not by the context, so its only remaining dependence is the one `graphs.Recording`
        checks: the branch write's offset, which is the context length modulo the page size. Bucketing the key by that
        remainder, rather than by the exact length, is what lets one recording answer about every document that shares
        it -- sixteen recordings instead of one per length ever seen.
        """
        if not self.paged:
            return (rows, width)
        from .paged import BLOCK as page_block

        remainder = next(
            (held % page_block for layer in cache.layers if (held := getattr(layer, "context_length", None)) is not None),
            None,
        )
        return (rows, width) if remainder is None else (rows, width, remainder)

    def _run_recorded(
        self,
        cache,
        fork,
        run,
        ids,
        positions,
        rows: int,
        width: int,
        remaining: int = 0,
        homogeneous: bool = True,
        lane: int = 0,
    ):
        """The pass, replayed from a recording where there is one and recorded where a second one is worth taking.

        Shared by the single-document branch pass (`_run_branch`) and the batched one (`_branch_across`): both fork a
        cache into the shape a pass needs, run the backbone, and may record it, and the only thing that differs
        between them is *how* the fork is done -- one document's snapshot widened, or several documents' snapshots
        laid out by row. `cache` and `fork` carry that difference in; everything from here down is the same decision.

        The order is what makes this safe. A recording is not a result: under stream capture the kernels are written
        down rather than run, so the pass is executed eagerly for its answer *first* and recorded afterwards.

        **Whether to record is arithmetic, not a habit.** `graphs.pays_from` says three passes at this shape must still
        be coming, because a recording costs 151 ms plus two proving replays and each replay after that saves 79. Two
        things can supply that number, and neither is a guess: `remaining` is how many more groups of this shape the
        call in progress will run, which the caller has already told the engine by handing over all its questions at
        once; and the count of times this shape has come back on this cache, which is evidence about a session asking
        group after group. A shape that has neither records nothing, so a caller asking one group about a document it
        will not revisit pays nothing for machinery it never uses.

        `homogeneous` is false for a pass answering about more than one document, and that turns graphs off for this
        call entirely -- found by an open-loop measurement answering wrong questions after this was shipped without
        it. The paged storage's remainder bucket is sound for *one* document's remainder: a batch's `context_length`
        is the *longest* document in it (`PagedForkLayer.begin_branches`), so two batches sharing that one number can
        still disagree, row for row, on every other document's remainder -- which is exactly what decides where that
        row's own branch write lands. A recording bakes that address in. Making the key or the check carry every
        row's remainder would fix it properly; until that is built, a mixed batch is not a shape graphs generalises
        over at all, and the honest thing is to say so rather than key it coarser and answer some rows wrong.
        """
        # Lane=2: CUDA graph capture is a stream-scoped operation, and recording's own
        # claim to "a private allocator pool for the life of the engine" has never been checked against a second
        # lane capturing on a second stream at the same time. Rather than find out by trusting it, a non-zero lane
        # always takes the eager path -- the concurrency this lane exists for is between lanes' eager passes, which
        # does not need graphs at all; the cost given up is lane 0's already-measured graphs economics (judged not
        # worth paying for and so rarely paid for even on lane 0 in practice).
        if not self.graphs or self.torch_device.type != "cuda" or not homogeneous or lane != 0:
            fork()
            return run(ids, positions)

        # Keyed on the cache rather than on the context, because the cache is what a recording holds the addresses of.
        # A context that closes returns its cache to the pool with its recordings attached, so the next context of the
        # same size replays instead of recording again. `_bucket_key` widens the key with a remainder bucket for the
        # paged storage, so a recording survives a change of document and not only a change of group.
        store = self._recordings_for(cache)
        key = self._bucket_key(cache, rows, width)
        recorded = store["taken"].get(key)
        if recorded is not None:
            # The bindings first, then the fork: the fork must write the context into the tensors the recording reads,
            # and after the last pass those are not the ones the layers point at.
            recorded.before_fork(cache)
            fork()
            wrong = recorded.usable(cache)
            if wrong is None:
                self.replays[key] = self.replays.get(key, 0) + 1
                return recorded.replay(cache, ids, positions)
            # A recording that no longer describes the cache is discarded rather than replayed. The alternative is a
            # plausible answer, and this package treats that as the worst outcome available.
            del store["taken"][key]
            self.declined_recordings[key] = wrong
            return run(ids, positions)

        fork()
        # Copied, and this is not defensive housekeeping. A recording replays into buffers the allocator may have handed
        # out for this pass's own output, so a replay can overwrite the answer that was just computed -- which made the
        # first version of the check below compare a tensor against itself and pass every time, and would have returned
        # the replay's values as this call's answer. The bug was found by a replay that gave a wrong answer while the
        # check reported agreement to zero.
        # Timed, because whether a recording can pay is a question about this shape on this card and the answer is not a
        # constant. See `_worth_keeping`.
        started = _now(self.torch_device)
        hidden = run(ids, positions).clone()
        eager_ms = _since(started, self.torch_device)
        store["seen"][key] = store["seen"].get(key, 0) + 1
        expected = max(remaining, store["seen"][key] - 1)
        # An economics decline is reopened once enough more passes have arrived to clear the bar it was declined by
        # -- `_economics_needed` holds that bar, set only for this one kind of decline. Everything else in
        # `declined_recordings` (a stale cache, a capture failure) stays closed: those are not "not enough passes
        # yet" and more passes would not change the answer.
        if key in self._economics_needed and expected >= self._economics_needed[key]:
            del self.declined_recordings[key]
            del self._economics_needed[key]
        # `pays_from()` here only gates whether a recording is *attempted* -- the decision whether to *keep* one, a
        # few lines down, already measures this exact shape's own eager and replay cost and does not borrow a ratio
        # from anywhere. See `graphs.PAGED_REPLAY_MS` for why swapping this gate's ratio was tried and reverted: the
        # short one-pass ratio (0.268) makes an *attempt* easier to justify than the paged branch pass's own ratio
        # (0.515) would, because a better ratio needs fewer future passes to pay back the same recording cost -- so
        # using the paged ratio here would make attempts rarer, which is the opposite of what was wanted. The gate
        # being generous is exactly why most attempts arrive one or two sightings short of the keep bar and need the
        # reopening above to get a second chance rather than none.
        # A kept recording holds a private allocator pool for the life of the engine -- nothing here ever frees one --
        # and the retry above means a shape declined once can now be kept later, so the number of pools this engine
        # ends up holding is not bounded by anything written down. Measured the hard way, twice: an open-loop run
        # against real traffic, with the retry in place, drove free device memory from several gigabytes to a few
        # megabytes and into a tight allocate-fail-retry loop that made no further progress -- and raising the
        # memory margin alone did not stop it recurring, because the margin only ever asks "is there room for one
        # more", never "how many are there already". `MAX_KEPT_RECORDINGS` asks the second question; the margin
        # stays as a check the first still answers usefully once the count is bounded.
        free, _ = (
            torch.cuda.mem_get_info(self.torch_device) if self.torch_device.type == "cuda" else (1 << 62, 1 << 62)
        )
        # Measured the gap directly --
        # after a rate=10 burst, `mem_get_info`'s free number was 2.16 GiB (below this margin) while PyTorch's own
        # `reserved - allocated` gap was 7.6 GiB of cached-but-unallocated blocks the caching allocator was simply
        # not returning to the driver, not memory any live tensor (recording, shelf snapshot, or anything else)
        # actually needed. `empty_cache()` recovered essentially all of it (free: 2.16 -> 9.54 GiB) in one call. A
        # margin check that only ever looks at the driver's free number declines on exactly this kind of transient
        # fragmentation, so one reclaim attempt happens here before giving up -- cheap because it only runs on the
        # already-below-margin path, not on every pass.
        # The reclaim call itself, not just reaching it, costs 2-4% of
        # questions/second under load (measured by isolating it from the dispatcher-narrowing
        # change). `RECLAIM_COOLDOWN_S` skips the call (not the margin check -- `room_to_record` below still comes
        # out `False` on the stale, still-tight `free`) while a previous attempt is still within its cooldown.
        if self.torch_device.type == "cuda" and free <= GRAPH_MEMORY_MARGIN:
            now = time.monotonic()
            if now >= self._reclaim_cooldown_until:
                # An earlier version of this check predicted
                # that `empty_cache()` would raise `mem_get_info`'s free number by `reserved - allocated` and
                # added that prediction straight into `free`, without ever calling `empty_cache()` or re-reading
                # `mem_get_info` to confirm it actually moved. `prismyra/schedule.py`'s `_make_room` carried the
                # identical fix and a bisection
                # found it broke `tests/test_gpu.py::test_a_shelf_evicts_on_memory_pressure_even_with_tokens_
                # to_spare` -- a case where the externally reported free number is genuinely tight for a reason
                # this process's own allocator slack does not explain, exactly the scenario this margin exists
                # to catch (see this check's own comment above, "97 documents... exhausted a 44 GiB card"). No
                # test exercises this copy the same way, but the reasoning is identical, so the same correction
                # applies here: `cached_slack == 0` is the one case the prediction can never be wrong about
                # (nothing cached to give back, so the call could not help and skipping it is a true no-op);
                # every other case now calls `empty_cache()` and re-reads the real number instead of predicting it.
                allocated = torch.cuda.memory_allocated(self.torch_device)
                reserved = torch.cuda.memory_reserved(self.torch_device)
                cached_slack = max(0, reserved - allocated)
                if cached_slack == 0:
                    pass
                else:
                    started = time.perf_counter()
                    torch.cuda.empty_cache()
                    self._empty_cache_calls += 1
                    self._empty_cache_ms += (time.perf_counter() - started) * 1e3
                    free, _ = torch.cuda.mem_get_info(self.torch_device)
                    if free <= GRAPH_MEMORY_MARGIN:
                        self._reclaim_cooldown_until = now + RECLAIM_COOLDOWN_S
        fits_memory_margin = free > GRAPH_MEMORY_MARGIN
        fits_kept_count = len(store["taken"]) < MAX_KEPT_RECORDINGS
        room_to_record = fits_memory_margin and fits_kept_count
        if expected < pays_from():
            self._skip_reasons["economics_gate"] = self._skip_reasons.get("economics_gate", 0) + 1
        elif key in self.declined_recordings:
            self._skip_reasons["already_declined"] = self._skip_reasons.get("already_declined", 0) + 1
        elif not fits_memory_margin:
            self._skip_reasons["memory_margin"] = self._skip_reasons.get("memory_margin", 0) + 1
        elif not fits_kept_count:
            self._skip_reasons["max_kept_recordings"] = self._skip_reasons.get("max_kept_recordings", 0) + 1
        if expected >= pays_from() and key not in self.declined_recordings and room_to_record:
            taken, why = record(run, cache, ids, positions, fork=fork)
            if taken is None:
                # Remembered so it is attempted once per shape rather than once per group, and reported rather than
                # retried in silence. Not an economics decline, so `_economics_needed` does not get an entry and this
                # one stays closed: a capture failure is about this shape, not about how many more passes are coming.
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
                moved, replay_ms = self._replay_disagreement_on(taken, cache, fork, ids, positions, hidden)
                self.verified_recordings[key] = moved
                self.replay_cost[key] = (round(eager_ms, 1), round(replay_ms, 1))
                slow = keeping_pays(eager_ms, replay_ms, expected)
                if slow is not None:
                    # Nothing to remove: the recording is only stored below, once it has been judged worth keeping.
                    self.declined_recordings[key] = slow
                    # The bar this shape needs to clear, in the same units as `expected` -- so the gate above can
                    # reopen this exact decline once enough more sightings have arrived, instead of never asking
                    # again. `pays_from` is the only thing `keeping_pays` computed `slow` from, so this is not a
                    # second measurement, only the number the first one already produced.
                    self._economics_needed[key] = pays_from(replay_ms, eager_ms)
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


def _round_rows(rows: int, cap: int) -> int:
    """How many rows a branch pass actually runs at: the next power of two at or above `rows`, never past `cap`
    (`self.group`, which every row buffer is already allocated for).

    Found by bisection, not supposition: two documents sharing a pass moved a probability by up to 0.29 and
    flipped decisions, with the question width matched and the question order ruled out, and the same move
    reproduced on **one** document with two unrelated dummy questions appended -- no second document, no
    varlen boundary, just a different row count. Comparing the row count against `ask()`'s own, with the
    borrowed `gated_delta_rule` kernel disabled (`PRISMYRA_WITHOUT=gated_delta_rule`), the same change in row
    count moved things by 0.084 instead of 0.19 and flipped nothing -- smaller, so some of this is ordinary
    floating-point reduction order, but the larger share is that kernel's own row-count-dependent tiling.

    A kernel chosen by shape cannot be told to ignore the shape, so this changes the shape instead: pads every
    pass to one of a small fixed set of row counts, so two passes that would have run at 5 and 7 real rows both
    run at 8, with the extra rows a harmless repeat of an existing row (discarded the same way `build_suffixes`
    already discards padded columns). Two documents or one document asked twice then see the identical kernel
    dispatch their row count would get alone, which is what makes answering together stop moving an answer.

    The power-of-two bucketing above closes most of the row-count effect but not all of
    it, because it rounds the *pass's total* row count, not each document's own -- a target alone and the same
    target plus a one-question companion can combine to totals either side of a power-of-two boundary even though
    the target's own row count never changed (the documented residual: `tests/test_gpu.py`'s `COMPANION_MOVEMENT /
    20`, measured 0.0128 on L40S; `audit_sm120.py` found the matching case on sm_120 at 0.057, i.e. this is not a
    different bug per card, it is the same one at a different magnitude). `PRISMYRA_ROUND_ROWS_TO_GROUP=1` rounds
    every pass straight to `cap` instead -- the simplest version of "every pass runs at the same width", so a
    document's own padding can no longer depend on who shares the pass -- at the cost of always paying for a
    full-width pass; the measured speed cost of that trade should guide whether this becomes the default.
    """
    if os.environ.get("PRISMYRA_ROUND_ROWS_TO_GROUP") == "1":
        return cap
    if rows >= cap:
        return cap
    return min(cap, 1 << (max(rows, 1) - 1).bit_length())


def _round_pack_align(longest: int, bucket: int) -> int:
    """How wide a branch pass actually needs to be: the longest row it carries, rounded up to `PACK_ALIGN`, never
    wider than `bucket` (a pinned `WIDTHS` entry, which is as wide as a recording may ever be asked to be).

    One function, because `_packed_groups` (one document, several groups) and `_answer_batch` (several documents,
    one group) are the same shape of decision -- a group of rows sharing one pass -- and having two copies of this
    arithmetic is how they drifted: `_answer_batch` used to call `self._width_for(plans)` and stop, which is `bucket`
    with no tightening. The two padding widths are not an equivalent rounding of each other; the padding columns
    reach the recurrent layers' kernels and move the hidden state at a row's own last token, not just the padding's.
    A document asked once through `ask()` and once through a shelf, with nothing else different, answered a
    different option (0.339 against 0.512 on the deciding one) until both went through this.
    """
    return min(bucket, -(-longest // PACK_ALIGN) * PACK_ALIGN)


def _answer_for(question: Question, values: list[float], read_by: str | None = None) -> Answer:
    """One question's answer from its probabilities, in its declared option order. Shared by every path that answers.

    `read_by` names the head that produced the probabilities, or is None when the output embedding did."""
    best = max(range(len(values)), key=lambda i: values[i])
    option = question.options[best]
    return Answer(
        id=question.id,
        kind=question.kind,
        value=question.value_of(option),
        option=option,
        probabilities=dict(zip(question.options, values, strict=True)),
        read_by=read_by,
    )


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


def _enable_batch_invariance() -> None:
    """Make the operations that actually move with a pass's row count row-independent -- no more of them.

    Found decisively: a document read alongside two companions of identical
    length but entirely different content was bit-identical through every layer-0 operation except one -- the
    router's own `F.linear(x, self.gate.weight)` inside `kernels.qwen3_moe.FusedExperts._route`. That one op's
    result for a real row moved with the pass's total row count, the row-count-chosen-GEMM-algorithm effect
    `onepass.py`'s islands already defend against during a *recording* (`rows_exact`) but nothing defends against
    during an ordinary read or branch pass. `kernels.qwen3_moe._ROUTER_LINEAR` calls vLLM's `linear_batch_invariant`
    directly at `_route`'s one call site, and `fused_experts`' own kernel gets `VLLM_BATCH_INVARIANT=1` (set below)
    so its tile-size choice stops keying on `M` too (`fused_moe.py`'s own guard, independent of any dispatcher
    override). An attempt that stopped at the router alone found 2/80 real-document mismatches
    left (largest move 0.189) -- not from attention or shared-expert projections as first suspected (that
    diagnosis was from a single synthetic two-document companion pair, which a fuller, real-document sweep
    across every layer later showed does not generalise), but from the borrowed
    chunked gated-delta-rule kernel (`vllm.third_party.flash_linear_attention`) itself: on the real mismatching
    batch (RACE validation, batch 4 of the 80-document benchmark), layer 0's `linear_attn` block alone moved by
    0.0039 between a solo read and the real 8-document batch with router+tiling already fixed -- before the
    router, before any MoE op, inside the recurrence this engine depends on for its batching to be fast at all.

    Two candidates were measured head-to-head at full scale (a single-layer signal, and separately the decisive 80-document/11-batch count):
    disabling TF32 and bf16/fp16 reduced-precision matmul reduction closed layer 0's gap (0.0039 -> 0.0) but
    *reopened* it at layer 4 once checked across all 40 layers on the real batch -- a precision change shrinks the
    chunk-boundary reduction-order effect enough to round to zero at shallow layers, not remove it, so it does not
    survive the full benchmark (still 2/80 mismatches). Registering vLLM's fixed-tile Triton matmul on
    `aten::mm`/`addmm`/`matmul`/`linear` (`enable_batch_invariant_mode`'s dispatcher step, SM80-family path) does
    survive it: 0/80 mismatches, confirming the borrowed kernel's inter-chunk state carry runs through one of
    those four ops. `enable_batch_invariant_mode()` does three more things this does not need: monkeypatching
    `torch.bmm` (measured alone: still 0.0039, no effect on this kernel), the TF32/reduced-precision flags just
    shown insufficient alone and unneeded once the dispatcher override is in, and preferring the `cublaslt` BLAS
    backend. None of the three changed the measured 0/80 result when added or removed alongside the dispatcher
    registration, so this function registers only the dispatcher override plus the MoE tiling env var, rather
    than calling `enable_batch_invariant_mode()`/`init_batch_invariance()` wholesale.

    Measured cost against the full five-lever version (`docs/PERFORMANCE.md`-style sweep, a solo-document measurement
    and the `open_batch` documents=8/16/32, group=64 sweep): unchanged within measurement noise -- the dispatcher
    registration this keeps was already the expensive part (every plain bf16/fp32 matmul in the decoder funnelled
    through a slower fixed-tile Triton kernel instead of cuBLAS), and `torch.bmm`/TF32/`cublaslt` were measured to
    cost nothing extra on this model (it has no plain `torch.bmm` call site and no FP32 tensors for TF32 to touch).
    Dropping them is a correctness simplification -- fewer process-wide side effects for whatever future kernel
    might actually use `torch.bmm` or care which BLAS backend is preferred -- not a speed win in this measurement.

    Kept only when a pass can actually share rows across more than one logical document (`paged` on CUDA, where
    `open_batch`/`Batcher` share a pass across several documents) or across a context and a branch in one call
    (`interleaved_fork` on CUDA -- see this `__init__`'s own call site): plain `ask()`/`open_context()`
    on a joined, non-interleaved cache never shares a pass across anything, so it has nothing this buys.
    The early return below used to leave sm_120 (RTX PRO 4500, the only card
    `nvfp4-36l` serves on) with *no* process-wide protection at all -- not even vLLM's own fallback for
    non-SM80 CUDA. Reading `vllm.model_executor.layers.batch_invariant.enable_batch_invariant_mode` (the function
    this one was written to narrow, not to replace) shows it has an explicit branch for exactly this case: "Hopper
    (SM90) and Blackwell (SM100): the only source of batch variance is split-k, which we disable via the cuBLAS
    workspace config" (`CUBLAS_WORKSPACE_CONFIG`/`CUBLASLT_WORKSPACE_SIZE`). This function's own `else: return`
    dropped that branch by omission, not by measurement -- there is no comment or diag log claiming it was tried
    and found insufficient on sm_120, only the comment that it "was not measured here". `audit_sm120.py`
    is the first time it has been.

    Reference-counted (`_BATCH_INVARIANT_REFCOUNT`), where it used to be a one-shot
    guarded only by `_BATCH_INVARIANT_DISPATCH_LIB is None`. Needed once `prismyra/engine.py`'s `paged` property
    started calling this on *every* `engine.paged = True`, not only at construction (see that property's own
    docstring for why): `tests/test_gpu.py` shares one engine object between `engine_paged` tests, which flip
    `paged` on and restore it to `False` in a `finally`, and a *second*, unrelated feature on the very same
    shared object -- `onepass.py`'s one-pass CUDA graph recording, which production code never combines with a
    paged engine (`Prismyra.ask`'s own `not self.paged` guard on the one-pass shortcut) but this test module's
    shared-weights fixture does. With the dispatcher registration left permanently on after the first
    `engine_paged` test, `test_a_short_question_replays_exactly_as_it_reads_eagerly` and
    `test_a_headed_question_replays_exactly_as_it_reads_eagerly` (both on the plain, non-paged `engine` fixture)
    started failing: a graph recorded and replayed consistently under the persistent-tile Triton matmul this
    function installs is not bit-identical to the one it was eager-compared against before -- a real,
    separate incompatibility between the two features, not an ordering artifact (confirmed by registering the
    dispatcher from the very first line of the `engine` fixture instead of from a later `engine_paged` test: the
    same two tests failed with the identical values either way). Production code structurally cannot hit this
    (paged and one-pass are mutually exclusive on one `ask()` call), so the correct fix is for the dispatcher
    registration to go away again once nothing paged still needs it, which is what the refcount buys: `paged`
    going back to `False` calls `_disable_batch_invariance()`, which drops the count and, at zero, drops the
    `torch.library.Library` and lets it be garbage-collected -- confirmed by hand that `del lib; gc.collect()`
    actually restores the original dispatch (`torch.library.Library`'s own documented behaviour, not assumed).
    """
    import os

    import torch
    from vllm.model_executor.layers.batch_invariant import (
        addmm_batch_invariant,
        linear_batch_invariant,
        matmul_batch_invariant,
        mm_batch_invariant,
    )
    from vllm.platforms import current_platform

    global _BATCH_INVARIANT_REFCOUNT
    _BATCH_INVARIANT_REFCOUNT += 1

    # fused_moe.py's own guard (`get_default_config`): picks a fixed MoE tiling config instead of one keyed by the
    # pass's row count M, independent of anything registered on the dispatcher below. Reset on every call (cheap,
    # idempotent) rather than only on the first, so a caller that disabled and re-enabled sees it reapplied too.
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

    if _BATCH_INVARIANT_REFCOUNT > 1:
        return  # already registered by an earlier claim; nothing left to do

    if not current_platform.is_cuda():
        return

    # "global" (the default, unchanged) registers the four aten ops below for
    # every plain bf16 matmul in the whole model -- measured at +14.5ms
    # (6.8%) on a solo `ask()` and +18.9ms (1.8%) on a 32-document `open_batch`, on an L40S, almost all of it in
    # `matmul_kernel_persistent` calls that are not at one of the
    # three call sites that are actually row-count dependent: the router (`_route`, now narrowed) and the
    # two `GatedDeltaNet` gate projections (`in_proj_a`/`in_proj_b`, narrowed by `_patch_gdn_gates`,
    # unconditionally -- see `qwen3_moe.Qwen3MoeAdapter.replace`). "narrow" skips this registration (and the
    # Hopper/Blackwell cuBLAS workspace change below) and relies on those two narrow fixes plus the
    # `VLLM_BATCH_INVARIANT` env var above instead.
    #
    # An earlier claim that "narrow" was measured safe on `audit_sm120.py`'s own
    # companion matrix did not hold up once the measurement actually ran: on sm_120, "narrow" leaves 150/4,392
    # probability checks non-exact
    # against "global"'s 72/4,392, and -- more to the point than the raw count -- it breaks question-counts
    # {2, 3, 31, 32} that "global" does not touch at all (`global`'s 72 are 100% question-count=1, a different,
    # already-tracked residual). That means at least one more plain `F.linear`/
    # `torch.mm` call this project has not yet narrowed is still row-count dependent on sm_120, and the global
    # dispatcher registration is the only thing currently catching it. "narrow" stays an opt-in diagnostic switch
    # for exactly this reason -- it is not a candidate default until whatever call site the 2→150 jump comes from
    # is found and narrowed too.
    if os.environ.get("PRISMYRA_INVARIANCE_SCOPE", "global") == "narrow":
        return

    # The Triton persistent matmul this registers (`mm_batch_invariant` et al.) is not
    # itself gated to SM80 anywhere in its own implementation -- `linear_batch_invariant` just calls
    # `matmul_persistent`, a plain Triton kernel, unconditionally. `_ROUTER_LINEAR` (`kernels/qwen3_moe.py`) was
    # already calling it directly on sm_120 (RTX PRO 4500, Blackwell) for the FP8 router with no crash and no
    # family check at all, which is the evidence that it runs there -- "vLLM's Triton version on sm_120" was a
    # question `audit_sm120.py`'s own code answers by example, not something that needed a separate try. An
    # earlier version of this function treated SM80 and "everything else" as needing *different* fixes (dispatcher
    # vs. cuBLAS workspace config only); registering the dispatcher everywhere closes a residual the cuBLAS-only
    # fix left (measured: `audit_sm120.py` non-exact 258/4392 before this change).
    lib = torch.library.Library("aten", "IMPL")
    key = current_platform.dispatch_key
    lib.impl("aten::mm", mm_batch_invariant, key)
    lib.impl("aten::addmm", addmm_batch_invariant, key)
    lib.impl("aten::matmul", matmul_batch_invariant, key)
    lib.impl("aten::linear", linear_batch_invariant, key)
    # Kept alive while the refcount is above zero (matching `enable_batch_invariant_mode`'s own module-level
    # singleton while it is wanted at all): letting it be garbage-collected is now how `_disable_batch_invariance`
    # un-registers these four, deliberately, rather than something to avoid happening by accident.
    global _BATCH_INVARIANT_DISPATCH_LIB
    _BATCH_INVARIANT_DISPATCH_LIB = lib

    if not current_platform.is_device_capability_family(80):
        # Belt and suspenders on every other family (Hopper/Blackwell, including sm_120): vLLM's own fallback for
        # these cards disables cuBLAS split-K the same way, for anything that reaches cuBLAS directly rather than
        # through one of the four aten ops above (e.g. a custom op that calls `at::cuda::blas::gemm` itself).
        # Original values saved on the module (`_BATCH_INVARIANT_SAVED_BACKENDS`) so `_disable_batch_invariance`
        # can put them back rather than guessing torch's defaults.
        global _BATCH_INVARIANT_SAVED_BACKENDS
        _BATCH_INVARIANT_SAVED_BACKENDS = (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.preferred_blas_library(backend="cublaslt")


def _disable_batch_invariance() -> None:
    """The other half of the refcount `_enable_batch_invariance` keeps: drops this caller's claim
    and, only once nothing else still holds one, actually reverses the registration -- un-registering the
    dispatcher override by dropping the last reference to its `torch.library.Library` (confirmed by hand that
    `del lib; gc.collect()` restores the original op, which is what makes doing this safe at all) and restoring
    the cuBLAS/TF32 backend flags `_enable_batch_invariance` saved before overwriting them. Does not touch
    `VLLM_BATCH_INVARIANT`: unlike the dispatcher registration, nothing has shown that env var alone breaks
    anything un-paged, and leaving a stray env var set is a smaller risk than mis-timing when fused_moe.py reads
    it.
    """
    global _BATCH_INVARIANT_REFCOUNT, _BATCH_INVARIANT_DISPATCH_LIB, _BATCH_INVARIANT_SAVED_BACKENDS
    if _BATCH_INVARIANT_REFCOUNT == 0:
        return
    _BATCH_INVARIANT_REFCOUNT -= 1
    if _BATCH_INVARIANT_REFCOUNT > 0:
        return
    if _BATCH_INVARIANT_DISPATCH_LIB is None:
        return  # never actually registered (e.g. this process is not on CUDA) -- nothing to undo
    import gc
    import os

    import torch

    _BATCH_INVARIANT_DISPATCH_LIB = None
    gc.collect()
    if _BATCH_INVARIANT_SAVED_BACKENDS is not None:
        cublas_cfg, cublaslt_size, fp16_rpr, bf16_rpr = _BATCH_INVARIANT_SAVED_BACKENDS
        for name, value in (("CUBLAS_WORKSPACE_CONFIG", cublas_cfg), ("CUBLASLT_WORKSPACE_SIZE", cublaslt_size)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = fp16_rpr
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bf16_rpr
        torch.backends.cuda.preferred_blas_library(backend="default")
        _BATCH_INVARIANT_SAVED_BACKENDS = None


_BATCH_INVARIANT_DISPATCH_LIB = None
_BATCH_INVARIANT_REFCOUNT = 0
_BATCH_INVARIANT_SAVED_BACKENDS = None


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


def _forget_recurrent_state(cache) -> None:
    """Put a cache's recurrent layers back into the state a fresh one is in, so the next read starts from nothing.

    A shelf's cache carries whatever the last read left, and a document read on top of that would begin from the last
    one's state -- the failure this package treats most seriously, since it answers plausibly rather than raising.
    `reset` would do it and would also clear the pages, which is where the documents already on the shelf live.

    **Four things, and not one of them is optional.** The framework allocates each state once, with the shape of the
    first one it sees, and thereafter copies into it in place to keep the address stable for CUDA graphs. A shelf reads
    one document and then six, so the shape has to be allowed to change, which means clearing the "initialised" flags
    and not only the entries. Each combination short of all four fails in a different line of the framework:

    | left alone | what happens |
    |---|---|
    | the entries | one initial state for six sequences: "expected 6 rather than 1" |
    | the entries, as None | `update_recurrent_state` copies in place and needs a tensor |
    | `has_previous_state` | the layer reads an entry that is gone: `KeyError` |
    | `is_..._initialized` | nothing is reallocated, so the `KeyError` moves a line down |

    With all four, the layer is in exactly the state a cache that has never held anything is in, and a read of any
    number of documents allocates what it needs. It also means the convolution takes its prefill branch, which pads
    rather than concatenating, so a read carries no prefix from the last one.
    """
    for layer in cache.layers:
        for entries, flags in (
            ("recurrent_states", ("is_recurrent_states_initialized", "has_previous_state")),
            ("conv_states", ("is_conv_states_initialized",)),
        ):
            held = getattr(layer, entries, None)
            if isinstance(held, dict):
                held.clear()
            for name in flags:
                marked = getattr(layer, name, None)
                if isinstance(marked, dict):
                    for key in marked:
                        marked[key] = False
