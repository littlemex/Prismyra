# Changelog

## 0.4.6

The 22.5 GiB budget that fell short of the 16,000-token, 32-question case by a small margin in
the previous release now fits it at the engine's full default width, with no narrower,
slower configuration needed.

- A gated-delta-net layer's branch-pass state buffer, sized at the engine's full group width, is
  now freed the moment a request that only needs one such group is done reading it, instead of
  being held until the whole cache drops. This removes a steady climb in held memory across a
  long pass; on the supported checkpoint, that climb used to add about 2.2 GiB from the first
  layer read to the last and now adds about 0.45 GiB. A multi-group request is unaffected by this
  change and keeps the prior behaviour.
- The formula admission uses to budget that same layer's held state priced its recurrent half at
  the model's own compute dtype, but the framework always keeps that half in `float32` regardless
  of the model's dtype, so the formula was pricing it at half its real width. The fix prices each
  half at its own dtype instead of a flat multiplier, which stays correct whatever dtype the model
  computes at. This is a correction to admission's own arithmetic only; every answer this engine
  gives is unaffected and bit-identical to the version before this change.

## 0.4.5

A 16,000-token context and 32 questions at once now fit on a 24 GiB class card with no extra
configuration, and on a smaller, 22.5 GiB budget with one explicit setting.

- Two opt-in weight placements, `PRISMYRA_LM_HEAD=lazy` and `PRISMYRA_EMBED_TOKENS=lazy`, keep the
  output and input embedding matrices in host, pinned memory between requests instead of on the
  device, gathering only the rows a forward pass actually names. Together they free roughly
  1.9-2.1 GiB on the supported 36-layer checkpoint. Both are off by default and have no effect
  unless set.
- A context longer than 4,096 tokens is now read in 4,096-token pieces fed through the same cache
  one after another, instead of one call over the whole length; each piece leaves the cache in
  exactly the state the same tokens would have left it in read in one pass. Whether a given read
  splits is decided per request from the engine's own memory accounting; there is no flag to set.
- That accounting had a blind spot under a per-process memory cap: a process given a software
  fraction of a card smaller than the whole of it had that fraction enforced by the allocator, but
  neither the automatic-chunking decision nor the engine's own admission check ever saw it. A new
  function corrects the free and total figures for the per-process fraction a caller may have set;
  a process with no fraction set is unaffected.
- A branch pass's read no longer copies the context once per question. The read now shares the
  context's pages across every question's table instead of copying them into it, and each
  question privately holds only its own tokens plus a copy of the context's own partial last
  page, bounded by a small, fixed page size rather than by the context's length. Where the full
  default width already fits a budget, the pass answers about 6% faster than before; combined with
  a narrower width for a tighter budget, the free memory left over roughly doubles and the latency
  cost of that narrower width falls by about half.

A 22.5 GiB budget and the engine's full default width together still fall short of the
16,000-token, 32-question case by a small margin, for a reason unrelated to the pieces above and
not addressed by this release. Building the engine with a narrower width closes that gap at the
cost of answering some of the wider requests in more than one internal pass.
