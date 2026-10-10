"""Reading a long context in pieces, instead of one pass over the whole thing.

A context read holds, grows and transiently touches memory of its own -- "transiently" because activations and
kernel workspace exist only for the one call that makes them and are freed the moment it returns. The context
itself (its keys and values, its gated-delta-rule state) is `use_cache=True` tracked state the model carries
forward from call to call, so a read of 16,000 tokens in one call and the same read in several calls -- each one
shorter, each one handing the model's `past_key_values` to the next -- is the identical computation seen through a
narrower window: nothing about causal attention over a growing cache, or a chunk-recurrent kernel's own carried
state, cares how many calls built it, only what state it is handed and what new tokens it is asked to extend that
state by.

That equivalence held exactly, measured `torch.equal`, comparing a 16,000-token document read as one pass against
the same document read as several 4,096-token pieces through the same cache: the hidden state at the document's
last token came back bit-identical, and the chunked read's own peak was lighter by more than a gigabyte.

The boundary placement matters for one of the two kinds of layer this model has. The borrowed gated-delta-rule
kernel does its own recurrence internally in chunks of 64 tokens; splitting a read anywhere that is not a multiple
of 64 hands that kernel a run it has to re-chunk around the carried state rather than extend evenly, and a
different internal chunking is exactly the kind of reduction-order difference this project treats as a changed
answer, not merely a slower one. Every boundary this module introduces -- including the one `interleave.
read_and_branch` hands off to its own, unchunked, layer-interleaved tail -- is therefore a multiple of
`CHUNK_TOKENS`, and `CHUNK_TOKENS` itself is a multiple of 64.

What is **not** claimed here: that a context read carrying media, or one going through the paged attention pool,
chunks the same way. Neither has been measured, and nothing in this module runs for either --
`should_chunk`/`chunk_boundary`/`read_chunked` are called only from the two text-only, joined-cache reads that
have been measured this way: `Prismyra._read`'s plain branch, and `interleave.read_and_branch`'s own context
prefix. A caller outside those two is a programming error in this package, not a configuration a user can reach.
"""

from __future__ import annotations

import torch

#: A read longer than this many tokens is split into pieces this large, down to one final, usually-shorter piece
#: a caller's own last pass takes over -- see this module's own docstring for why the length is a multiple of 64
#: and the measurement behind this particular multiple. Large enough that a 16,000-token document is three calls
#: ahead of its caller's own last one, not two hundred; still comfortably the thing that must shrink a long read's
#: peak rather than add a floor of its own.
CHUNK_TOKENS = 4096

#: The least a caller's own last pass over the leftover tail is ever asked to carry, once chunking applies at
#: all. Found necessary on real hardware, not assumed: a one-token tail (document length `CHUNK_TOKENS + 1`)
#: crashed the borrowed gated-delta-rule kernel's Triton compilation (`ssm_state_indices`/`INPLACE_FINAL_STATE`,
#: a code path that single-token calls take and multi-token ones do not -- the kernel's own decode-step
#: shortcut, which this project's cache does not feed what it wants). `chunk_boundary` keeps the tail at or
#: above this by rolling a too-small remainder back into the chunked prefix instead of leaving it on its own;
#: a multiple of 64 so a tail this small is still a whole GDN chunk, not a fragment of one.
MIN_TAIL_TOKENS = 64

#: A conservative, hardcoded floor for how much transient memory one token of a *one-shot* read costs, used only
#: to decide whether a given read should chunk -- not `Prismyra.reading_bytes`, which answers the same question
#: for admission and is honestly zero on a process that has not read anything yet (that method's own docstring).
#: Zero is the right default for admission -- refuse nothing there is no evidence against -- and the wrong one
#: here: guessing a fresh process's first long read costs nothing is exactly the case this module exists to catch
#: before it runs as one pass. 1.96e-4 GiB/token is the measured rate `reading_bytes` itself documents.
_READING_BYTES_PER_TOKEN_FLOOR = int(1.96e-4 * 2**30)

#: How much larger than its own estimate the one-shot read's cost must be allowed to run before chunking kicks in,
#: as a multiple of that estimate. Not 1.0: an estimate is a prediction, not a fact, and the margin exists so a
#: single bad one still leaves the read's *actual* peak, not this module's guess, as what `Prismyra._check_fits`
#: refuses against -- chunking is meant to be the thing that already happened by the time a close call arrives.
_SHOULD_CHUNK_MARGIN = 2


def chunk_boundary(context_tokens: int) -> int:
    """How many of `context_tokens` a chunked read peels off into whole `CHUNK_TOKENS` pieces, ahead of a
    caller's own last pass over whatever is left. Zero when there is only one piece either way -- a context no
    longer than one chunk gains nothing from being split and only pays a Python round trip for it."""
    if context_tokens <= CHUNK_TOKENS:
        return 0
    boundary = ((context_tokens - 1) // CHUNK_TOKENS) * CHUNK_TOKENS
    if context_tokens - boundary < MIN_TAIL_TOKENS:
        boundary -= CHUNK_TOKENS  # too small a tail on its own; fold it into the chunked prefix's last piece instead
    return max(boundary, 0)


def visible_free(device: torch.device) -> tuple[int, int]:
    """`torch.cuda.mem_get_info`, corrected for a process that was given less of the device than the device has.

    `mem_get_info` answers for the device, not for this process. `torch.cuda.set_per_process_memory_fraction` is a
    soft cap PyTorch's own allocator enforces on top of that -- and the CUDA call behind `mem_get_info` has never
    heard of it: on a 32 GiB card capped to 22.5 GiB, it keeps reporting the device's own free figure, not this
    process's much smaller one. Found by measuring both: a 16,000-token context's branch pass admitted and read as
    one piece under a simulated 22.5 GiB cap reached a peak within a few hundred MiB of the same pass's peak with
    no cap at all on the same card -- `should_chunk` and `Prismyra._check_fits` had both looked at the device's
    free figure, seen tens of GiB of it, and never seen the cap at all. Only the allocator itself saw it, and it
    refused the next byte past it with no warning either of them had the chance to give.

    A real card sized to the limit this is simulating needs none of this -- its own `mem_get_info` already answers
    correctly, because there `total` *is* what fits -- and a process the fraction was never set on pays nothing for
    it either: `get_per_process_memory_fraction` defaults to `1.0`, which makes `cap` wider than `total` ever is and
    the `min` below a no-op. What this corrects is specifically a process capped *tighter* than the device it is
    on, real or simulated, which `set_per_process_memory_fraction` is the one documented way to be.
    """
    free, total = torch.cuda.mem_get_info(device)
    # `get_per_process_memory_fraction` insists on an index (`cuda`, with none, is not enough -- `mem_get_info`
    # above is more forgiving and does not), so a device with none names the one CUDA already made current.
    index = device.index if device.index is not None else torch.cuda.current_device()
    fraction = torch.cuda.get_per_process_memory_fraction(index)
    if fraction < 1.0:
        cap = int(total * fraction)
        headroom = cap - torch.cuda.memory_reserved(device)
        free = min(free, max(0, headroom))
        total = cap
    return free, total


def should_chunk(engine, context_tokens: int) -> bool:
    """Whether this context's own read should go through `chunk_boundary`'s pieces rather than one pass over all
    of it.

    Automatic, from the same memory accounting `Prismyra._check_fits` already keeps -- not a flag a caller sets,
    because the two shapes are bit-identical (this module's own docstring) and nothing about *correctness* turns
    on which one runs, only on whether the one-shot read's own transient would leave this device too little
    headroom. `engine` is duck-typed (`torch_device`, `cache_bytes`, `reading_bytes`, `_one_pass`) rather than
    imported as `Prismyra` to avoid this module importing `engine.py`, which already imports this one.
    """
    if chunk_boundary(context_tokens) == 0:
        return False
    if engine.torch_device.type != "cuda":
        return False  # chunking trades device memory for Python round trips; there is nothing to trade on the CPU path
    held = engine.cache_bytes(context_tokens)
    one_shot = max(engine.reading_bytes(context_tokens), context_tokens * _READING_BYTES_PER_TOKEN_FLOOR)
    free, _total = visible_free(engine.torch_device)
    # The same "the allocator's own idle pool is spare too" adjustment `_check_fits` makes, for the same reason:
    # the device's free figure alone undercounts what this process can still use.
    spare = torch.cuda.memory_reserved(engine.torch_device) - torch.cuda.memory_allocated(engine.torch_device)
    one_pass = getattr(engine, "_one_pass", None)
    if one_pass is not None:
        spare -= one_pass.held_bytes
    free += max(0, spare)
    return held + one_shot * _SHOULD_CHUNK_MARGIN >= free


def read_chunked(engine, cache, input_ids: torch.Tensor, boundary: int) -> None:
    """Feed `input_ids[:, :boundary]` to `engine.backbone` in `CHUNK_TOKENS` pieces, growing `cache` as it goes.

    The caller reads whatever is left (`input_ids[:, boundary:]`) itself, in whichever shape its own read needs
    -- a plain last pass (`Prismyra._read`) or a layer-interleaved one carrying a branch group alongside it
    (`interleave.read_and_branch`) -- and either continues correctly from the state this leaves in `cache`,
    because that state is exactly what the same tokens would have left behind read in one pass (this module's
    own docstring).
    """
    start = 0
    while start < boundary:
        end = min(start + CHUNK_TOKENS, boundary)
        engine.backbone(input_ids=input_ids[:, start:end], use_cache=True, past_key_values=cache)
        start = end
