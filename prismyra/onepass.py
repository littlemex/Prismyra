"""The one-pass read of a short request, recorded once per length bucket and replayed.

A single question about a short text is read in one pass over the context and the question together, and on the
supported model that pass is **host-bound**: measured on an L40S at 232 tokens, 92 ms of wall clock around 40 ms of
kernel time, 2,564 kernel launches. The device spends most of the request waiting to be told what to do next, so a
faster kernel changes nothing, and a recorded graph -- one launch where there were thousands -- removes the wait.

What a recording cannot do is change shape, and the one-pass read's shape is its length. So the length is bucketed:
the request is right-padded to a bucket and read at its last real token. Every layer is causal -- the attention, the
convolution and the recurrence all run forward in position -- so a token after the last real one cannot reach it.

**That argument is true and not sufficient**, and the difference is the design of this file. Causality says the padding
cannot reach a real token through the model. It says nothing about the libraries: a matrix multiply may choose its
algorithm by the number of rows, and on the supported model the bf16 ones do. The router's projection gives a real
row different bits when it is multiplied as one of 640 rows instead of one of 581 -- a different reduction order, one
rounding step in a logit -- and on a question sitting at 0.44 against 0.54 that was enough to change the answer. A
check on random rows did not see it: a reduction order that differs is hidden by rounding to bfloat16 almost every
time, so equal outputs on a random probe are weak evidence of an equal algorithm.

So the recording is **piecewise**. Every projection whose algorithm the library chooses by row count -- the bf16
``F.linear`` calls, which on this model are the router, the shared expert's gate and the recurrence's two small input
projections -- is an *island*: at replay it runs eagerly on exactly the request's real rows, which is the call the eager
path makes, and its result is copied into the buffer the next recorded piece reads. Everything between islands is
recorded at the bucket's length and replayed; those kernels compute each row independently of how many rows there are.
That is a claim too, and it is proved rather than trusted: every bucket is replayed at the shortest and the longest
length it serves and compared bit for bit with the eager read of the same tokens, and a bucket that differs in any bit
is not kept.

Three more properties make a recording valid across requests, and each is checked rather than assumed:

* **a fresh state every time.** Each recording is taken from a cache that has never been read, so a replay starts the
  recurrence from nothing whatever ran before it. The second proving replay runs after the first has left its state
  behind and must agree with it to the bit;
* **its own inputs.** The ids, the position read and every island's input and output are tensors the recording holds.
  A graph stores addresses and keeps nothing alive at them, so everything it reads is held here;
* **nothing shared that persists.** The buckets share one cache and one private pool. Every tensor a recorded piece
  reads is allocated outside the pool (weights, static inputs, island outputs, the keys and values each pass writes
  before it reads) or written earlier in the same replay, so one bucket's replay cannot leave anything another reads.

The memory all of it takes is measured when it is taken (`held_bytes`), and admission subtracts it from what it counts
as available.
"""

from __future__ import annotations

import contextlib
import time
import warnings
from dataclasses import dataclass, field

import torch
from torch import nn

#: The lengths a one-pass read is recorded at, in tokens of context plus question. A request is padded to the smallest
#: that holds it, and a longer one runs eagerly. Spaced by 64 up to 1,024, where padding is a visible share of a short
#: pass, and by 128 above it, where the pass is long enough that a replay saves less of it.
BUCKETS = (*range(64, 1025, 64), *range(1152, 2049, 128))
#: Passes run before a recording, each from a fresh state. One is enough to compile what compiles on first use; a
#: capture that still fails is declined and named, and its lengths run eagerly.
WARMUPS = 1
#: Replays spent proving a bucket at each length it is proved at. Two, for the reason `graphs.PROVING_REPLAYS` gives:
#: the first replay runs in the state the recording was taken in, and only the second runs after a replay.
PROVING_REPLAYS = 2


class _Active:
    """The recording in progress, if any. Held at module level because the islands are reached from inside the
    framework's forward, where nothing can be passed down to them."""

    recorder: _Recorder | None = None


# --------------------------------------------------------------------------- islands
def rows_exact(fn, x: torch.Tensor) -> torch.Tensor:
    """``fn(x)``, except while a pass is being recorded, where ``fn`` becomes an island run at the real row count.

    ``x`` carries its tokens on its second-to-last dimension, which is how every projection on this model receives
    them. Outside a recording this is one function call and changes nothing.
    """
    recorder = _Active.recorder
    if recorder is None:
        return fn(x)
    return recorder.island(fn, x)


def _chooses_by_rows(module: nn.Module) -> bool:
    """Whether calling this module is a bf16 (or wider) ``F.linear``, whose algorithm the library picks by row count.

    The block-quantised projections are not: they run on a Triton kernel whose configuration is fixed for the shape,
    and replacing them is `kernels.qwen3_moe.Fp8Linear`'s job. Their original modules stay in the tree as that
    replacement's ``inner`` and are never called, so they are left alone.
    """
    weight = getattr(module, "weight", None)
    return (
        isinstance(module, nn.Linear)
        and isinstance(weight, torch.Tensor)
        and weight.dim() == 2
        and weight.is_floating_point()
        and weight.element_size() > 1
    )


def install_islands(decoder: nn.Module) -> int:
    """Route every row-count-sensitive projection in the decoder through `rows_exact`. Idempotent; returns how many."""
    count = 0
    for name, module in decoder.named_modules():
        if name.endswith(".inner") or not _chooses_by_rows(module):
            continue
        if getattr(module, "_prismyra_island", False):
            count += 1
            continue
        original = module.forward

        def forward(x, *args, _original=original, **kwargs):
            if args or kwargs:
                return _original(x, *args, **kwargs)
            return rows_exact(_original, x)

        module.forward = forward
        module._prismyra_island = True
        count += 1
    return count


class Islands:
    """The buffers every bucket's islands read from and write to, shared across buckets.

    Shared because buckets never run at once: a replay writes an island's input, runs the island and has its output
    read by the next piece before any other bucket runs. Held per bucket instead, they were nearly all of what the
    recordings cost -- 9.8 GiB for 24 buckets, because each kept its 126 island inputs alive at its own length.

    * one **staging** buffer for every island's input. The piece before an island copies the island's input into it,
      inside the recording, and the island reads it from there. The input itself is then an ordinary temporary of the
      recording that later pieces may reuse;
    * one **output** buffer per island, by its position in the pass, at the longest bucket's length. A bucket reads a
      view of its own length, so every bucket's recording reads the same addresses.
    """

    def __init__(self, longest: int, width: int, dtype: torch.dtype, device):
        self.longest = longest
        self.staging = torch.zeros(longest * width, dtype=dtype, device=device)
        self.outputs: list[torch.Tensor] = []

    def stage(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() > self.staging.numel():
            raise ValueError(f"an island's input of {tuple(x.shape)} is larger than the staging buffer")
        view = self.staging[: x.numel()].view(x.shape)
        view.copy_(x)
        return view

    def output(self, index: int, like: torch.Tensor) -> torch.Tensor:
        """The output buffer for the `index`-th island of a pass, as a view of `like`'s shape."""
        length = like.shape[-2]
        if index == len(self.outputs):
            shape = (*like.shape[:-2], self.longest, like.shape[-1])
            self.outputs.append(torch.zeros(shape, dtype=like.dtype, device=like.device))
        held = self.outputs[index]
        if held.shape[:-2] != like.shape[:-2] or held.shape[-1] != like.shape[-1] or held.dtype != like.dtype:
            raise ValueError(
                f"island {index} produced {tuple(like.shape)} {like.dtype} and an earlier bucket's was "
                f"{tuple(held.shape)} {held.dtype}; the islands of a pass must be the same at every length"
            )
        return held[..., :length, :]


class _Recorder:
    """Records one pass as pieces, ending a capture at each island and beginning the next after it."""

    def __init__(self, pool, islands: Islands):
        self.pool = pool
        self.islands = islands
        self.steps: list = []
        self.current: torch.cuda.CUDAGraph | None = None

    def begin(self) -> None:
        self.current = torch.cuda.CUDAGraph()
        self.current.capture_begin(pool=self.pool)

    def end(self) -> None:
        assert self.current is not None
        # Two islands in a row leave nothing to record between them, and an empty graph replays as nothing. The
        # framework warns about it because it usually means a capture on the wrong stream; here it is expected.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The CUDA Graph is empty")
            self.current.capture_end()
        self.steps.append(self.current)
        self.current = None

    def island(self, fn, x: torch.Tensor) -> torch.Tensor:
        staged = self.islands.stage(x)  # recorded: the piece ends by handing the island its input
        self.end()
        # Run once, outside any capture, for the output's shape. Its contents are meaningless -- nothing recorded so
        # far has executed -- and it is not what the next piece reads: that is the shared output buffer, whose padded
        # rows hold zeros or an earlier request's real values, either of which is finite and reaches only padding.
        out = self.islands.output(sum(1 for step in self.steps if isinstance(step, tuple)), fn(x))
        self.steps.append((fn, staged, out))
        self.begin()
        return out


# --------------------------------------------------------------------------- recordings
@dataclass
class Bucket:
    """One recorded read: its pieces and islands in order, and the static tensors it reads and writes."""

    length: int
    steps: list
    ids: torch.Tensor
    at: torch.Tensor
    hidden: torch.Tensor
    replays: int = 0

    @property
    def islands(self) -> int:
        return sum(1 for step in self.steps if isinstance(step, tuple))


@dataclass
class OnePassGraphs:
    """Recorded one-pass reads by bucket, with what taking and proving them measured."""

    buckets: dict[int, Bucket] = field(default_factory=dict)
    #: Per bucket, the worst disagreement between a replay and the eager read, over the lengths it was proved at. Every
    #: kept bucket is at zero; the figure is kept so that "it was kept" and "it agreed" stay separate claims.
    proved: dict[int, float] = field(default_factory=dict)
    #: Buckets refused, with the reason.
    declined: dict[int, str] = field(default_factory=dict)
    islands: int = 0
    held_bytes: int = 0
    record_ms: float = 0.0
    cache: object = None
    shared: Islands | None = None
    #: How much the allocator's reservation grew while each bucket was recorded, in bytes.
    growth: dict[int, int] = field(default_factory=dict)

    def bucket_for(self, tokens: int) -> Bucket | None:
        """The smallest recorded bucket that holds this many tokens, or None to read eagerly."""
        fits = [length for length in self.buckets if tokens <= length]
        return self.buckets[min(fits)] if fits else None

    def stats(self) -> dict:
        return {
            "buckets": sorted(self.buckets),
            "islands_per_pass": next(iter(self.buckets.values())).islands if self.buckets else 0,
            "row_sensitive_projections": self.islands,
            "replays": {length: b.replays for length, b in sorted(self.buckets.items())},
            "proved": dict(sorted(self.proved.items())),
            "declined": dict(sorted(self.declined.items())),
            "held_bytes": self.held_bytes,
            "growth_bytes": dict(sorted(self.growth.items())),
            "record_ms": round(self.record_ms, 1),
        }


def fresh(cache) -> None:
    """Put a one-row cache back into the state a cache that has never been read is in.

    The attention layers rewind to empty; the recurrent layers forget their state entirely, so the next read takes the
    framework's prefill branch rather than continuing a recurrence from the last one. `engine._forget_recurrent_state`
    is the same four resets and says why all four are needed.
    """
    from .engine import _forget_recurrent_state

    for layer in cache.layers:
        if getattr(layer, "holds_attention", False):
            layer._host_length = 0
            layer.cumulative_length.zero_()
            layer.context_length = 0
            layer.writing_branches = False
    _forget_recurrent_state(cache)


def record_bucket(
    engine, cache, islands: Islands, length: int, pad_id: int, pool, side: torch.cuda.Stream
) -> tuple[Bucket | None, str | None]:
    """Record the one-pass read at one bucket length, or return why it could not be."""
    device = engine.torch_device
    ids = torch.full((1, length), pad_id, dtype=torch.long, device=device)
    at = torch.zeros(1, dtype=torch.long, device=device)

    def run() -> torch.Tensor:
        # The same pinned MoE tile `engine._read_one_pass` uses for a bucket this recording never reaches
        # (longer than `BUCKETS`' top): warm-up, capture and the island's own real-row-count re-run all take this
        # path, so a kept bucket's replay and `_read_one_pass`'s eager fallback never disagree over which tile
        # the routed-expert GEMM used. See `kernels.onepass_moe_tuning` for why this is safe to bake into a
        # recording (one fixed tile, proved bit-identical at this bucket's shortest and longest length exactly
        # like every other kernel this file records).
        from .kernels import onepass_moe_tuning

        with onepass_moe_tuning.scope():
            out = engine.backbone(input_ids=ids, use_cache=True, past_key_values=cache)
        last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return last[0].index_select(0, at)

    side.wait_stream(torch.cuda.current_stream(device))
    recorder = _Recorder(pool, islands)
    try:
        with torch.inference_mode(), torch.cuda.stream(side):
            for _ in range(WARMUPS):
                fresh(cache)
                run()
            torch.cuda.synchronize(device)
            fresh(cache)
            _Active.recorder = recorder
            try:
                recorder.begin()
                hidden = run()
                recorder.end()
            finally:
                _Active.recorder = None
                # A capture left open by a failure must be closed before anything else runs on the device. It is
                # already failing, so the first error is the one reported.
                if recorder.current is not None:
                    with contextlib.suppress(Exception):
                        recorder.current.capture_end()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001 - any failure means the eager path, which is always available
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
    return Bucket(length=length, steps=recorder.steps, ids=ids, at=at, hidden=hidden), None


def replay(bucket: Bucket, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Read `ids` (one row, no longer than the bucket) through the recording; the hidden state at its last token."""
    real = ids.shape[1]
    bucket.ids[:, :real].copy_(ids)
    bucket.ids[:, real:].fill_(pad_id)
    bucket.at.fill_(real - 1)
    with torch.inference_mode():
        for step in bucket.steps:
            if isinstance(step, tuple):
                fn, x, out = step
                out[..., :real, :].copy_(fn(x[..., :real, :]))
            else:
                step.replay()
    bucket.replays += 1
    return bucket.hidden


def prove(engine, bucket: Bucket, lengths, pad_id: int, eager) -> float:
    """The worst a replay disagrees with the eager read of the same tokens, over the given lengths.

    Each probe's tokens are drawn from the vocabulary with a seed fixed by its length, so a bucket is proved on the same
    sequences on every start.
    """
    vocabulary = int(engine.unembedding.shape[0])
    worst = 0.0
    for real in lengths:
        generator = torch.Generator(device="cpu").manual_seed(real)
        probe = torch.randint(0, vocabulary, (1, real), generator=generator).to(engine.device)
        reference = eager(probe).float()
        for _ in range(PROVING_REPLAYS):
            replayed = replay(bucket, probe, pad_id).float()
            worst = max(worst, float((replayed - reference).abs().amax()))
    bucket.replays = 0
    return worst


def record_all(engine, pad_id: int, eager, lengths=BUCKETS) -> OnePassGraphs:
    """Record and prove every bucket. One that fails either is declined and named, and its lengths run eagerly."""
    from .cache import build_cache
    from .fork import WIDTHS

    held = OnePassGraphs()
    device = engine.torch_device
    decoder = getattr(engine.backbone, "language_model", engine.backbone)
    held.islands = install_islands(decoder)
    torch.cuda.synchronize(device)
    reserved = torch.cuda.memory_reserved(device)
    started = time.perf_counter()
    lengths = sorted(set(lengths))
    # One cache for every bucket, sized for the largest. Each pass writes the keys and values it reads, from position
    # zero, so a bucket never sees another's; the recurrent state is reallocated by every recording from a fresh cache.
    held.cache = build_cache(
        engine.config, engine.room_for(lengths[-1]) + WIDTHS[-1], 1, engine.dtype, engine.device, WIDTHS[-1]
    )
    held.shared = Islands(lengths[-1], engine.hidden_size, engine.dtype, device)
    pool = torch.cuda.graph_pool_handle()
    # One stream for every recording. The caching allocator keeps what a stream frees for that stream, so a stream per
    # bucket left each warm-up's activations reserved where nothing would use them again: 8.9 GiB for 24 buckets,
    # against 1.7 GiB on one stream.
    side = torch.cuda.Stream(device=device)
    below = 0
    for length in lengths:
        bucket, why = record_bucket(engine, held.cache, held.shared, length, pad_id, pool, side)
        held.growth[length] = torch.cuda.memory_reserved(device) - reserved - sum(held.growth.values())
        if bucket is None:
            held.declined[length] = why or "unknown"
            continue
        # The shortest length this bucket serves carries the most padding, and the longest carries none.
        moved = prove(engine, bucket, sorted({below + 1, length}), pad_id, eager)
        held.proved[length] = moved
        if moved != 0.0:
            held.declined[length] = f"a padded replay sat {moved:.3e} from the eager read of the same tokens"
            continue
        held.buckets[length] = bucket
        below = length
    torch.cuda.synchronize(device)
    held.record_ms = (time.perf_counter() - started) * 1e3
    held.held_bytes = max(0, torch.cuda.memory_reserved(device) - reserved)
    return held
