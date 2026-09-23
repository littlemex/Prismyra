"""A branch pass recorded once and replayed, and the host-side bookkeeping a replay does not do.

Half of a branch pass is the device waiting to be told what to do next: 6,438 kernel launches for four rows of a short
suffix, and 55.1 ms of actual work inside 109.4 ms of wall clock. A recorded graph is one launch. Measured on the
supported model, the same forward: **107.7 ms eagerly, 28.9 ms replayed**, with the recording itself costing 151.3 ms
once.

Three things make that possible here and would not in a general engine, and all three are properties this package
already had for other reasons:

* the cache is **preallocated**, so the addresses a recording bakes in stay valid;
* the branch width is **pinned** to a bucket, so a suffix of forty tokens and one of fifty share a recording;
* the context is **held open**, so its length -- the one shape that a recording cannot generalise over -- is fixed for
  every group of questions asked about it.

**What a replay does not do is run Python.** That is the whole point and it is also the hazard. Each cache layer keeps a
host-side count of the tokens it holds, so that the framework can ask for the length during a forward without a
device-to-host copy on the request path; a replay moves the bytes and leaves that integer where it was. The next group
then advances from the wrong offset and answers **plausibly**, which is the failure this package treats most seriously.
So the host state is recorded alongside the graph, at the moment the recording was made, and restored after every
replay. `fork.LENGTH_ATTRS` is the same list of names the snapshot machinery uses, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .fork import LENGTH_ATTRS

#: What a pass costs eagerly and replayed, measured on the supported model at a short suffix. Only their **ratio** is
#: used, because the absolute figures belong to one suffix width and the ratio is what survives a change of width.
REPLAY_MS = 28.9
EAGER_MS = 107.7

#: Replays spent proving a recording before it may answer. `engine.REPLAY_CHECKS` is the same number; it is a cost here.
PROVING_REPLAYS = 2


def pays_from(replay_ms: float = REPLAY_MS, eager_ms: float = EAGER_MS, warmups: int = 3) -> int:
    """How many passes at one shape must still be coming for recording it to be cheaper than not recording it.

    Counted **in passes, not milliseconds**, and that is the whole correction. Taking a recording runs `warmups` passes
    on a side stream, then one more under capture, then `PROVING_REPLAYS` replays that answer nothing. So with a ratio
    ``r = replay_ms / eager_ms`` and R passes still to run:

        (warmups + 1) + PROVING_REPLAYS x r  <  R x (1 - r)

    which on the measured ratio of 0.268 gives R > 6.20, so **seven**.

    The first version of this said three, from a figure of 151.3 ms for "recording costs". That figure cannot be right
    and the arithmetic is what shows it: three warm-up passes at 107.7 ms each is 323 ms before the capture begins, so
    a cost of 151 ms had left the warm-ups out. Expressed in milliseconds the error was invisible; expressed in passes
    it is not expressible. Measured, at 128 questions about one context -- four groups, where three-says-yes and
    seven-says-no:

    | | questions per second |
    |---|---|
    | recording on, threshold of three | 47.21 |
    | recording off | **102.07** |

    A 2.16x loss where the arithmetic had promised a saving. That is the second time this mechanism has cost more than
    it saved, and both times the cause was a cost model with a term missing rather than a kernel behaving unexpectedly.
    """
    if eager_ms <= 0:
        raise ValueError(f"a pass cannot cost {eager_ms} ms")
    r = replay_ms / eager_ms
    if r >= 1:
        # A replay no cheaper than the pass cannot pay for itself at any count. "Never" is the honest answer, rather
        # than a large number a long enough session would eventually cross.
        return 1 << 30
    return int(((warmups + 1) + PROVING_REPLAYS * r) / (1 - r)) + 1


def keeping_pays(eager_ms: float, replay_ms: float, expected: int) -> str | None:
    """None if this recording will pay for itself, or why it will not.

    Asked **after** the recording exists, because the number it needs cannot be known before: how much a replay
    saves depends on the shape, and by more than a little. Measured on the supported model at a 3,000-token context
    and thirty-two rows:

    | suffix width | eager pass | replayed pass | ratio | passes needed to pay |
    |---|---|---|---|---|
    | 16 | 109.4 ms | 74.8 ms | 0.684 | 17 |
    | 32 | 110.7 ms | 94.7 ms | 0.855 | 40 |
    | 64 | 144.2 ms | 142.1 ms | **0.986** | 414 |
    | 128 | 268.1 ms | 267.3 ms | **0.997** | 2,190 |

    A recording removes the time the device spends waiting to be told what to do next, and **kernel launches are
    asynchronous**: once each kernel takes longer than the call that launches it, the host stays ahead of the device
    and there is no waiting left to remove. That is the whole story of the table. The 55.1 ms of kernel time inside
    a 109.4 ms pass that motivated this mechanism was a short suffix, where the pass is host-bound; at a suffix of
    128 tokens the same pass is device-bound and a recording is worth 0.8 ms of 268.

    So a fixed threshold cannot work, and two of them have now been measured failing -- three passes, then seven,
    each losing about two-fold at 128 and 256 questions per context. This replaces both. The cost of finding out is
    one recording per shape per engine, remembered in `declined_recordings` so it is paid once.
    """
    needed = pays_from(replay_ms, eager_ms)
    if expected >= needed:
        return None
    return (
        f"a replay of this shape costs {replay_ms:.1f} ms against {eager_ms:.1f} eagerly, so recording it pays "
        f"from {needed} more passes and {expected} are expected"
    )


#: Passes run before a recording is taken. Three, on a side stream, which is what the recording needs: allocators,
#: autotuners and any kernel that compiles on first use must have finished, or they happen inside the capture and it
#: fails -- `cudaErrorStreamCaptureInvalidated`, with nothing to say which of them did it.
WARMUPS = 3

#: Host-side attributes that a replay cannot update and that a wrong value in makes a wrong answer rather than an error.
#: The lengths come from `fork`; the rest is this cache's own record of where a context ended and what it last wrote.
CACHE_STATE = ("context_length", "writing_branches", "last_branch_rows")


#: State a layer holds by **rebinding** rather than by writing in place: the recurrence and the convolution
#: replace their tensors on every pass. A recording reads whatever was bound when it was taken, so those bindings
#: have to be put back before each replay -- otherwise the fork writes the context into the tensor the last pass
#: produced while the recording reads the one from before it, which is a fixed wrong answer rather than a drift.
REBOUND = ("recurrent_states", "conv_states")


def _bindings(cache) -> list[dict]:
    """Which tensor each layer's rebindable state currently points at. Identities, not values."""
    out = []
    for layer in cache.layers:
        kept: dict = {}
        for attr in REBOUND:
            held = getattr(layer, attr, None)
            if isinstance(held, dict):
                kept[attr] = dict(held)
            elif isinstance(held, list):
                kept[attr] = list(held)
        if not getattr(layer, "holds_attention", False):
            for attr in ("keys", "values"):
                value = getattr(layer, attr, None)
                if torch.is_tensor(value):
                    kept[attr] = value
        out.append(kept)
    return out


def _rebind(cache, state: list[dict]) -> None:
    for layer, kept in zip(cache.layers, state, strict=True):
        for attr, value in kept.items():
            if isinstance(value, dict):
                held = getattr(layer, attr)
                held.clear()
                held.update(value)
            elif isinstance(value, list):
                getattr(layer, attr)[:] = value
            else:
                setattr(layer, attr, value)


def _host_state(cache) -> list[dict]:
    """Every host-side counter in the cache, as data. Read after a recording and written after every replay."""
    out = []
    for layer in cache.layers:
        kept: dict = {}
        for name in (*LENGTH_ATTRS, *CACHE_STATE):
            value = getattr(layer, name, None)
            if torch.is_tensor(value):
                kept[name] = value.clone()
            elif isinstance(value, int | bool):
                kept[name] = value
        out.append(kept)
    return out


def _restore_host_state(cache, state: list[dict]) -> None:
    for layer, kept in zip(cache.layers, state, strict=True):
        for name, value in kept.items():
            current = getattr(layer, name, None)
            if torch.is_tensor(current) and torch.is_tensor(value):
                # In place, because something else holds a reference to this tensor -- that is why it exists.
                current.copy_(value)
            else:
                setattr(layer, name, value)


@dataclass
class Recording:
    """One recorded branch pass: the graph, the buffers it reads and writes, and the host state it cannot set itself."""

    graph: torch.cuda.CUDAGraph
    #: Written before every replay. The recording baked in this tensor's address, so it is filled rather than replaced.
    ids: torch.Tensor
    #: Read after every replay, and overwritten by the next one. A caller that needs it to survive must copy it.
    hidden: torch.Tensor
    rows: int
    width: int
    #: The context length the recording was taken at. A recording is valid for that length and no other, and this is
    #: checked rather than hoped for.
    #:
    #: It is not enough that the *shapes* match. A branch's tokens are written at an offset measured from the end of the
    #: context, and with pages that offset includes the context length modulo the page size -- so a context thirteen
    #: tokens longer puts the branch in different slots, and the recorded writes go to the slots from before. Measured:
    #: five documents of 251 to 302 tokens, answers wrong by up to 0.15, with the recording reporting agreement to zero
    #: because the check had only ever compared replays taken on the context the recording came from.
    context_length: int = 0
    #: Everything else the recorded pass reads that Python allocated. Held for the recording's whole life, because a
    #: graph bakes in addresses and does not keep the tensors at them alive: the position ids were a local of the call
    #: that took the recording, so they were freed when it returned and the replay went on reading memory the allocator
    #: had handed to something else. That failed in exactly the way that is hardest to catch -- the replays taken
    #: immediately after the recording were right, because the tensor was still alive, and every later one was wrong.
    kept: tuple = ()
    after: list[dict] = field(default_factory=list)
    #: The bindings the recording reads from, and the ones the pass it recorded left behind. Swapped around a replay so
    #: that the caller's fork writes into what the recording reads, and what the caller reads afterwards is the output.
    reads: list[dict] = field(default_factory=list)
    writes: list[dict] = field(default_factory=list)

    def before_fork(self, cache) -> None:
        """Point the layers back at the tensors the recording reads, so the caller's fork fills those."""
        _rebind(cache, self.reads)

    def usable(self, cache) -> str | None:
        """None if this recording still describes the cache, or what is wrong if it does not.

        The context's length is checked first and is the common case: a pooled cache is reused by the next context, and
        the next context is rarely the same length.

        Checked before every replay rather than trusted. A recording holds tensors by identity, and anything that
        rebinds one of them without going through `before_fork` -- another code path, a reallocation, a layer the
        framework changed -- leaves the recording reading bytes nobody is writing. The result of that is a plausible
        answer, which is worse than an error, so the answer is an error: this returns a reason and the caller runs the
        pass eagerly instead.
        """
        for layer in cache.layers:
            held = getattr(layer, "context_length", None)
            if held is not None and held != self.context_length:
                return f"the context is {held} tokens and this recording was taken at {self.context_length}"
        for n, (layer, want) in enumerate(zip(cache.layers, self.reads, strict=True)):
            for attr, value in want.items():
                held = getattr(layer, attr, None)
                if isinstance(value, dict):
                    for key, tensor in value.items():
                        if held is None or held.get(key) is not tensor:
                            return f"layer {n} rebound {attr}[{key}] away from the tensor the recording reads"
                elif isinstance(value, list):
                    for i, tensor in enumerate(value):
                        if held is None or i >= len(held) or held[i] is not tensor:
                            return f"layer {n} rebound {attr}[{i}] away from the tensor the recording reads"
                elif held is not value:
                    return f"layer {n} rebound {attr} away from the tensor the recording reads"
        return None

    def replay(self, cache, ids: torch.Tensor) -> torch.Tensor:
        """Run the recorded pass on new suffix tokens, and leave the cache as that pass would have left it.

        The caller forks first, after calling `before_fork`. Neither is inside the recording, and `record` says why.
        """
        if ids.shape != self.ids.shape:
            raise ValueError(f"this recording takes ids of {tuple(self.ids.shape)}, not {tuple(ids.shape)}")
        self.ids.copy_(ids)
        self.graph.replay()
        _rebind(cache, self.writes)
        _restore_host_state(cache, self.after)
        return self.hidden


def record(run, cache, ids: torch.Tensor, fork, keep: tuple = ()) -> tuple[Recording | None, str | None]:
    """Warm up, record the pass, and return something replayable -- or None and the reason it could not be recorded.

    `run` takes the static ids tensor and returns the pass's hidden states, and `fork` puts the cache back to the end of
    the context. `run` is called `WARMUPS` times on a side stream and then once inside the recording, with `fork` before
    each of them -- without that the warm-ups continue from one another and the recording is taken against a state no
    real pass is ever in.

    `keep` is every other tensor the pass reads that Python allocated -- the position ids, anything else a closure
    captured. The recording holds them so they cannot be freed, because a graph stores addresses and nothing else.

    `fork` is outside the recording on purpose. With it inside, one replay disagreed with the eager pass and two replays
    disagreed with each other, because the layer rebinds its recurrent state to a new tensor on each pass and a
    rebinding is Python that a replay cannot repeat.

    A recording is not a result. Under stream capture the kernels are written down rather than executed, so the tensors
    it returns hold whatever the warm-up left; the caller runs the pass eagerly for its answer and records afterwards.
    Treating the capture's output as an answer is the quiet version of this going wrong.

    Failure is returned rather than raised. Capture is fragile for reasons that have nothing to do with the caller --
    a kernel that compiles on first use, an allocator that reaches past its pool -- and the eager path is always
    available, so an engine that cannot record should say so and carry on.
    """
    static = ids.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(side):
            for _ in range(WARMUPS):
                fork()
                run(static)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        fork()
        reads = _bindings(cache)
        with torch.cuda.graph(graph):
            hidden = run(static)
        torch.cuda.synchronize()
        writes = _bindings(cache)
    except Exception as e:  # noqa: BLE001 - any failure here means the eager path, which is what the caller has
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    return Recording(
        graph=graph,
        ids=static,
        hidden=hidden,
        rows=ids.shape[0],
        width=ids.shape[1],
        kept=tuple(keep),
        context_length=next(
            (held for layer in cache.layers if (held := getattr(layer, "context_length", None)) is not None), 0
        ),
        after=_host_state(cache),
        reads=reads,
        writes=writes,
    ), None
