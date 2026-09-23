# Performance

Every number here was measured on one machine, which each file in
[`benchmarks/results/`](../benchmarks/results/) names. **CI cannot catch a performance regression**: the hosted runners
have no GPU, so the kernels never execute there. What CI does check is that the result files are well formed and
internally consistent, because the README's tables are generated from them.

## The shape

```
total ~= 138 ms  +  91.6 ms x ceil(questions / 32)
```

At about 5,000 context tokens. Two constants, and they answer different questions:

* **138 ms** is one traversal of the model over the context. It is paid once per context, whatever you then ask.
* **91.6 ms** is one traversal carrying a group of up to 32 questions. A traversal costs the same whether it carries one
  question or thirty-two, which is why the curve is flat to 32 and then steps.

Both constants grow with the context length. The per-question figure people usually want, 2.85 ms, exists only at a full
group; at one question per group it is 91.6 ms.

## Memory

The other half of the cost, and the one the cost model does not mention. Every branch holds its own copy of the
context's keys and values, so an open context needs

```
2 x attention_layers x group x kv_heads x (context + 512) x head_dim x bytes_per_element
```

which on the supported model -- 10 attention layers among 40, **2** key-value heads, head dimension 256, bfloat16 -- is
3.36 GiB at 5,000 tokens and the default group of 32, and 12.52 GiB at 20,000. The recurrent layers are absent from that
formula because their state is a fixed size per layer whatever the context length.

Two key-value heads is why this is affordable at all, and it is easy to get wrong: the model has 16 attention heads and
this family's linear-attention layers carry 32 value heads, neither of which is the number in this formula. Taking one
of those instead overstates the cache by 8x or 16x.

`group` is therefore a memory dial as well as a cost dial: it divides the memory exactly and multiplies the traversals
by `ceil(questions / group)`. `Prismyra.cache_bytes(tokens)` reports the figure for a given configuration, and
`open_context` refuses a context that will not fit rather than letting the allocator refuse it.

## Where it wins and where it loses

| questions | Prismyra | vLLM |
|---|---|---|
| 1 | 225.9 ms | **134 ms** |
| 2 | -- | 181 ms |
| 3 | -- | 228 ms |
| 4 | -- | 275 ms |
| 8 | 227.2 ms | 463 ms |
| 16 | 228.1 ms | 839 ms |
| 32 | 229.5 ms | 1,591 ms |
| 64 | 320.2 ms | 3,095 ms |
| 128 | 502.6 ms | 6,103 ms |

The crossover is at four questions. Below it, a general serving engine is the right tool and the README says so.

The baseline is vLLM on the same card, with `VLLM_USE_DEEP_GEMM=0`, which this architecture needs. Its context pass is
87 ms and each question costs 47 ms; the context figure and the per-question figure are measured, and the entries above
four questions multiply the latter. That is vLLM used straightforwardly. Padding contexts to block multiples and
submitting every question in one batch are untested workarounds that would narrow the gap, and they are not counted here
in either direction.

The 47 ms is the whole point. vLLM stores the context's keys and values in pages, and a question lands in the middle of
the last page, so the page is recomputed rather than shared. Prismyra copies the partial page instead. See
[FORK.md](FORK.md).

## How the context pass got from 288 ms to 138 ms

Five changes, each measured as its own paired run: the same process, the same inputs, with and without that one change.

| change | before | after | won |
|---|---|---|---|
| routed experts on a fused kernel | 288.1 | 208.3 | 79.8 |
| dense projections on a block-scaled fp8 kernel | 204.5 | 185.7 | 18.8 |
| normalisation on a faster kernel | 185.2 | 174.7 | 10.5 |
| head duplication deleted, bit-identically | 175.1 | 166.6 | 8.5 |
| convolution on a kernel written for it | 166.7 | 138.3 | 28.4 |

**The chain is deliberately not continuous.** One step's "after" and the next step's "before" differ by a few
milliseconds, and both are printed rather than smoothed. Subtracting one column from the other across rows does not
work, and `benchmarks/check_results.py` enforces the gap staying small enough to be run-to-run variation -- a large gap
would mean the steps came from different configurations and do not describe one sequence.

What each change is and what was measured and rejected on the way is in [KERNELS.md](KERNELS.md).

## The branch pass

One group of up to 32 questions: **123.5 ms to 91.6 ms**, from putting the branch pass on a variable-length attention
kernel. Attention over those ten layers went from 43.3 ms to 10.9 ms.

The cause is worth knowing because it is invisible from the outside. With a cache present the framework materialises an
additive attention mask, and the fast kernel cannot take one, so it silently falls back to a kernel built for an older
architecture. No mask is needed here: every branch's queries are the last positions of its own sequence and attend to
the context's keys plus its own, which is exactly the right-aligned causal case the fast kernel already handles.

## Against vLLM's own kernels

Same work, same card, read from profiles of both. Lower is better.

| | Prismyra | vLLM |
|---|---|---|
| convolution, 30 layers | **3.0 ms** | 3.37 ms |
| recurrence, 30 layers | **18.9 ms** | 21.0 ms |
| attention, 10 layers | **6.8 ms** | 8.13 ms |
| routed experts, 40 layers | **29.5 ms** | 29.97 ms |
| dense projections | 32.7 ms | **21.65 ms** |

Three are faster and the convolution is also more accurate than the one it replaces. The dense projections are slower
and that is the honest remaining gap.

## Concurrency, and the ceiling that actually binds

Measured on one L40S at 3,040 context tokens, eight questions each, through `prismyra.queue.Worker`:

| callers arriving together | requests per second | first answer home | median | median spent queueing | median in
  service |
|---|---|---|---|---|---|
| 1 | 1.04 | 962 ms | 962 ms | 0 ms | 962 ms |
| 2 | 1.09 | 926 ms | 1,384 ms | 463 ms | 921 ms |
| 4 | 1.08 | 934 ms | 2,324 ms | 1,398 ms | 929 ms |
| 8 | 1.08 | 927 ms | 927 ms -> 4,158 ms | 3,235 ms | 924 ms |

**Throughput does not move.** One device serves one request at a time, so it cannot: service time stays at about 925 ms
whatever arrives, and every millisecond added by concurrency is queueing. That is worth knowing before building anything
for multiple users, because it says what such work could and could not achieve -- it can decide who waits, not how many
are served.

What bound instead was memory, and that has been fixed. **Nineteen contexts of 3,040 tokens can be open at once, where
three could**, because the context is now held once rather than once per branch:

| | held per open context | contexts on one 48 GiB card |
|---|---|---|
| a copy per branch | 2.17 GiB | 3 |
| the context once | **0.37 GiB** | **19** |

The answers did not change, to the bit, and the clock barely did: 991 ms against 962 ms for one caller, and 168.2
against 167.7 ms per question on RACE. What the change costs is a copy that lives for one layer -- the context and the
branch rows joined for that layer's read, 222 MiB at these sizes -- rather than 2.17 GiB held for as long as the
context is open.

Two measurements decided how to do it. One of them turned out to be true only of the shape it was taken at, and the
correction is the more useful finding:

* **A narrow batch reclaims nothing -- at 3,040 context tokens.** A branch pass cost 245 ms at one row and 246 ms at
  thirty-two, a factor of 1.01, and that was read as the design's central premise holding. It holds at that length and
  not beyond it: see below.
* **The attention kernel tolerates aliased pages.** Rows whose page tables name the same prefix pages answer as rows
  holding their own copies, within two steps of bfloat16. That was measured on device, and it was the kill criterion for
  removing the join by handing the kernel a page table instead. The capability is real; the path built on it is not
  here, and why is in [the paged path](#the-paged-path-that-never-ran) below.

## What a pass costs, by phase and by width

The figure above -- what an open context *holds* -- is the smaller half of the story. Answering transiently allocates
several times as much, and reading the context allocates more still. `prismyra-bench widths --context-tokens N` reports
all three; peaks are the allocator's, measured above the bytes already allocated when the phase began, with the device
synchronised either side.

| context tokens | width | branch ms | held GiB | reading peak | answering peak |
|---|---|---|---|---|---|
| 3,040 | 1 | 250.1 | 0.129 | 0.731 | 0.080 |
| 3,040 | 8 | 235.5 | 0.198 | 0.802 | 0.634 |
| 3,040 | 32 | 254.1 | 0.432 | 1.035 | 2.534 |
| 24,327 | 1 | 238.3 | 0.563 | 5.330 | 0.111 |
| 24,327 | 8 | 247.8 | 0.613 | 5.381 | 1.238 |
| 24,327 | 32 | 360.4 | 0.868 | 5.631 | 4.970 |

Three things follow, and the first two overturn what this document said before.

**The premise is context-dependent.** A branch pass costs the same at any width when the context is short -- 1.02 across
widths at 3,040 tokens -- and does not when it is long: 1.51 at 24,327, 238 ms against 360 ms. The 1.01 reported above
was taken at one length and generalised to all of them. What makes the difference is per-row work proportional to the
context: each row's read is joined with its own copy of the context, and at 3,040 tokens that is small against the fixed
cost of a traversal while at 24,327 it is not. So a request with three questions should not pay for thirty-two rows of
it, and since this measurement **it does not** -- a group uses as many rows as it has questions.

**Answering costs several times what holding costs.** 4.970 GiB against 0.868 GiB held at 24,327 tokens and full width.
Admission used to compare only the held cache against free memory, which meant a context could be admitted, sit
comfortably, and run out of memory on its first question. It now budgets the work as well.

**Reading costs more than answering, and no caller can shrink it.** The reading peak is 4.78 GiB above the cache it
leaves behind at 24,327 tokens, against 2.53 GiB for the widest branch pass, and it is flat in the group -- 1.96e-4 GiB
per context token at both lengths measured. Asking fewer questions does not reduce it; only a shorter context does. So
there is a length beyond which a context cannot be opened at all, whatever the group. `prismyra-bench ceiling` finds it
and reports which of the two refused it:

| context tokens | predicted read | outcome |
|---|---|---|
| 3,264 | not yet known | read |
| 26,112 | 6.588 GiB | read |
| 52,224 | 13.177 GiB | refused by name, before allocating anything |

The observed figure is 235,582 bytes per context token. A fresh engine reads the first context unbudgeted and measures
it; from then on the refusal arrives before the allocator is asked, and it says that only a shorter context helps --
because when reading is what does not fit, a smaller group changes nothing.

### How admission budgets them

Both figures are **observed on the engine that will use them**, not derived, and both are ratcheted upwards as passes
happen. The answering figure is split in two, which is not tidiness:

    per row  =  a constant, observed  +  2,048 bytes x context tokens, arithmetic

The arithmetic part is the join, from the model's own shapes -- two key-value heads, head_dim 256, two bytes, keys and
values. It was 4,096 while the join was copied a second time to reach the layout the kernel reads, and 3,838 was
measured against that prediction, so it over-stated by 7%, which is the safe direction. The second copy is gone (see
[below](#the-second-copy-of-the-join)) and the figure halves with it. The **fallback** path -- the framework's own
attention, when the borrowed kernel is unavailable -- is budgeted at 4,096 instead, because it is handed a strided view
and whether it copies that is not something this package controls or has measured.
Keeping the largest observed *prediction* instead would ratchet the wrong way: a prediction scaled up from a short
context is larger per token than one from a long context, so the shortest context ever seen would win and be kept
forever, and a full-width pass at 24,327 tokens would be budgeted at 17.7 GiB when it needs 4.97 -- refused with four
gigabytes idle. The constant does not depend on the context length, so ratcheting it cannot do that.

The first context on a fresh engine is **unbudgeted**, because the figure is observed and there is nothing to observe
yet. That is a real hole and is closed by the allocator rather than by admission: a 48,655-token context on a fresh
engine raises a named error from the read, saying that asking fewer questions will not help and a shorter context is the
only knob. After one read the figure exists and a longer context is refused before anything is allocated.

**Per-row cost used not to be flat in width**, and admission still says it cannot vouch for a width it has not seen.
`prismyra-bench admission --context-tokens 18000` runs the sequence that shows why: a fresh engine answers one question
at 24,327 tokens, then the estimate it makes for thirty-two is compared with that pass.

| | per row at width 1 | per row at width 32 | estimate for 32 rows | that pass measured |
|---|---|---|---|---|
| head-major, copied twice | 0.111 GiB | 0.155 GiB | 4.028 GiB | 4.970 GiB, short by 1.23x |
| token-major, copied once | 0.109 GiB | 0.109 GiB | 4.028 GiB | 3.482 GiB, covered |

The second copy of the join **was** the width dependence. It is proportional to rows and to context, and it was the part
that made a narrow pass look cheaper per row than a wide one; with it gone the two agree to three decimals and an
observation at width 1 predicts width 32. That was not the reason for making the change and is not something this
document predicted.

`budget_is_evidenced()` is nevertheless still false for a pass wider than the widest yet measured, and the refusal says
"at least" rather than a figure. One measurement on one model is not a reason to promise a shape holds; the allocator
remains the authoritative gate, and `torch.OutOfMemoryError` is caught and re-raised naming the knobs that apply --
fewer questions per call, or a smaller group.

Free memory is the device's free bytes plus the allocator's reserved-but-unallocated pool. Asking the device alone
refused an 18,000-token context that had 9 GiB waiting for it inside the process, because loading these weights leaves
that pool large.

### The paged path that never ran

A paged storage path was written -- `block_table` and `seqused_k`, the context's pages named by every row's table, the
join removed entirely. It was measured at two context lengths, reported identical decisions and probability movements of
0.0000, and was 8-9% slower at 3,040 tokens and break-even at 18,003.

None of that happened. `Prismyra.__init__` stored the flag, validated that the installed kernel took the arguments, and
reported `"storage": "paged"`; `_read` calls `build_cache` with six positional arguments and the flag is the seventh.
The paged layer was never constructed. Every "paged" run was the joined path compared against itself, which is why a
path that reduces in a different order moved no probability at all and why the peaks matched to three decimals.

What found it was disbelieving the perfection. The flag had a construction-time check, a field in `stats()`, a benchmark
option, and eighteen unit tests of its page arithmetic -- and no wiring. The code is gone rather than fixed: it was
never a feature, and bringing it back correctly means doing the verification from the beginning anyway. `ForkLayer`
now records `last_branch_rows`, the row count it actually received, so a claim about how a pass ran can be checked
against the code that would have done the work.

### The second copy of the join

Half of what the join cost was not the join. The borrowed kernel wants `(total_tokens, heads, dim)`; the framework's
cache layout is head-major, `(rows, heads, tokens, dim)`; and `transpose(1, 2).reshape(...)` cannot be a view, so every
layer of every pass copied the whole join a second time to get there.

`ForkLayer` now stores token-major and returns a transposed **view**, so callers still see the framework's layout and
the kernel's own reshape is free. The attention code is unchanged. What it bought, at 24,327 context tokens:

| context tokens | | branch ms at width 32 | branch peak at width 32 | width factor, 1 to 32 |
|---|---|---|---|---|
| 24,327 | head-major, copied twice | 360.4 | 4.970 GiB | 1.51 |
| 24,327 | token-major, copied once | 312.0 | **3.482 GiB** | **1.29** |
| 3,040 | head-major, copied twice | 254.1 | 2.534 GiB | 1.02 |
| 3,040 | token-major, copied once | 252.7 | 2.534 GiB | 1.02 |

At 24,327 tokens the saving is 1.488 GiB, against 1.63 predicted by halving the slope, and 48 ms of clock. **At 3,040
tokens nothing moved at all** -- the peak is identical to three decimals, not merely close. That is worth stating
rather than averaging away: at a short context the join is not the high-water mark of the pass, so removing half of it
changes the traffic and not the peak. The change is a long-context one, and the shape of the table is how you can tell.

Answers are unchanged, and not within a tolerance: the kernel is handed the same bytes in the same order. All 137 device
tests pass.

What remains of the join is one copy per layer per row, 2,048 bytes per context token per row, and removing it is what
a page table would be for. Two cheaper things come first and neither is done: a scratch buffer the join is written into
rather than allocated per layer, and running rows through attention in chunks so the buffer is bounded by the chunk. The
first trades transient memory for held memory, which is the figure the storage work bought in the first place, so it
needs measuring rather than assuming.

## Against a serving engine

`evals/generate.py` says in its own docstring that the speed it measures is an upper bound, because "a serving engine
can batch the questions and share the article's prefix without using this read-out at all". `evals/against_vllm.py` is
that engine: vLLM, the same weights, the same card, the same RACE questions, prefix caching on, one request per
question, the output restricted to the letter tokens so both arms score declared answers and normalise over them.

| arm | answers | accuracy | questions / s | ms per context | tokens sent |
|---|---|---|---|---|---|
| this package, as first measured | 234 | 0.953 | 7.36 | 536.7 | **28,421** |
| this package, with the recurrence kernel | 234 | 0.953 | **16.44** | 220.2 | **28,421** |
| vLLM, one request per question | 234 | 0.936 | 31.64 | 125.0 | 76,556 |

**Still 1.9x behind, from 4.3x behind.** Accuracy is not a like-for-like comparison and no claim is made from it: the
arms score different tokens, since a request returning one token can only score the letters while this package scores
the option text. The token column is what this package wins, by 2.7x -- it sends the context once per document where
vLLM sends it once per question and leans on the prefix cache not to recompute it.

### What closed half the gap, and how the rest of it looks

Not a faster kernel. One branch pass, profiled:

| | as first measured | with the recurrence kernel |
|---|---|---|
| wall clock | 268.4 ms | 109.4 ms |
| sum of all kernel time | 112.7 ms, **42% of the wall clock** | 55.1 ms, **50%** |
| kernel launches | **32,873**, 820 per layer across 40 layers | **6,438**, 161 per layer |
| largest single kernel | routed experts, 16.3 ms | the fp8 dense matmul, 5.47 ms across 250 calls |

Fifty-eight percent of a pass was the device waiting to be told what to do next, and that also explains the measurement
which never made sense alone: a branch pass costing the same at width 1 and width 32. A cost that does not move with the
work is not the work.

The launches were not spread evenly. **One gated delta net layer issues 1,021 of them against a full-attention layer's
114**, and thirty of the forty layers are gated delta nets, so 93% of a pass came from them: 218 copies, 191 elementwise
kernels, 82 multiplies and 66 sums, for one layer. That is a chunked scan written in PyTorch -- and it is there as a
*fallback*. The framework decorates both of its gated-delta-rule paths to ask a kernel hub first, and the slow one runs
when the hub has nothing installed. vLLM ships the kernel the hub would have provided.

Borrowing it took a branch pass from **268.4 ms to 111.8 ms**, and the whole read-out from 137.5 to 57.6 ms per
question. It is a numerical change, not a bit-identical one, because the framework promotes to float32 and scans in
chunks of sixty-four and the kernel does neither. What that costs, measured over 472 RACE questions with
`PRISMYRA_WITHOUT=gated_delta_rule` running the other implementation in the same harness:

| | ms per question | accuracy | decisions identical |
|---|---|---|---|
| the borrowed kernel | **57.6** | 0.9407 | -- |
| the framework's own scan | 137.5 | 0.9364 | **465 / 472 = 98.5%** |

Seven decisions moved out of 472 and accuracy did not fall. Two video tests had asserted an exact integer on a question
whose top two options are 0.35 apart from each other and one of the seven; they now assert what they were written to
catch, which is that the clip's timing reached the model at all. `tests/test_gpu.py` says so at the assertion.

### The rest of the gap

The same diagnosis still applies, one layer down. What remains is launch count, and the routes to it are ordered by what
they cost to build:

* **the convolution kernel covers only the context pass.** A branch pass arrives as many rows and keeps the framework's
  path, which is recorded where it is installed and is the next cheap thing to look at.
* **CUDA graphs are shipped, behind `Prismyra(graphs=True)`.** The earlier note said a recording faulted on replay and
  the reason was never found. The reason was what was being recorded: capturing `ask` fails immediately --
  `cudaErrorStreamCaptureInvalidated` -- because the read-out converts probabilities to host values and the suffix ids
  are copied in from pageable memory. Capturing **only the forward**, with the ids in a static buffer and three warm-up
  passes on a side stream:

  | group | eagerly | with recording |
  |---|---|---|
  | 1 | 147.1 ms | 114.6 ms |
  | 2 | 115.1 ms | 605.8 ms, the recording |
  | 3 | 113.8 ms | **32.8 ms** |
  | 4 | 113.6 ms | **32.5 ms** |
  | 5 | 113.4 ms | **32.6 ms** |

  **3.5x from the third group, and every group answers identically** -- not within a tolerance, the same probabilities
  to six decimals. The recording is taken lazily on a shape's second use, so a caller asking one group about a
  document it will not revisit pays nothing. The 491 ms the recording costs above an eager pass is repaid by the
  seventh group.

  Off by default, and the reason is memory rather than doubt: a recording holds a private allocator pool, and this
  engine refuses a context by name from a budget it measures, so a feature that quietly takes device memory behind
  that budget would make the refusal wrong.

  **A graph stores addresses and keeps nothing alive at them.** That sentence cost most of a day. The recorded pass
  reads the position ids, and those were a local of the call that took the recording, so they were freed when it
  returned and every later replay read memory the allocator had since handed to something else. The recording now
  holds every tensor the pass reads that Python allocated, and the answers are identical again -- 0.0 across four
  groups and four repeats, with three different warm-up histories.

  The shape of that failure is why it took so long. **The replays taken immediately after the recording were right**,
  because the tensor was still alive; every later one was wrong. So a check that replayed once and compared against the
  pass it was taken from passed every time, and so did a check that replayed twice. Two earlier attempts at a check were
  worse than nothing: one compared the eager output against itself, because the replay had overwritten the buffer
  holding it, and one counted paged reads by summing an attribute that never existed, so it could only ever return
  zero. **A check that can only return the answer you hope for reads as evidence and is not.**

  The check is still there and still runs -- two replays against a copy of the pass, with `stats()["graphs_verified"]`
  reporting what it measured and `graphs_declined` naming anything refused. It did not catch this, so it is a guard
  rather than a proof.

  What it took to get right is worth more than the speed-up. A replay **runs no Python**, and two separate things
  depended on Python running. The cache's host-side length -- the integer each layer keeps so the framework can ask how
  many tokens it holds without a device read -- is not advanced by a replay, so it is recorded alongside the graph and
  restored after. And the recurrence **rebinds** its state to a new tensor on every pass, which a replay cannot repeat:
  after a recording the layers point at the tensor that pass produced, while the recording goes on reading the one bound
  when it was taken. Both bindings are kept and swapped around each replay, so the fork writes the context into what the
  recording reads. Getting either wrong gives a plausible answer rather than an error, which is why the device test
  compares probabilities across four groups rather than decisions on one.
* **the paged path is back, and it runs.** Its memory saving was measured and was zero, so it was deleted; the reason
  to restore it is that the read's shape stops depending on the context's length, which is what a recording needs. On a
  fingerprint-verified tree, over nine groups across three contexts of different lengths:

  | | decisions changed | largest probability move | ms per group | reads served |
  |---|---|---|---|---|
  | joined against paged | **0** | **0.0000** | 108 -> 112 | 90 |

  Two things to take from that. The paged read is **bit-identical** to the joined one, which corrects what the deleted
  version of the file assumed -- it said a paged read reduces in a different order and would move answers, and it does
  not. And `reads served = 90` is nine groups times ten attention layers, which is the number that says the path ran:
  the first version of this file reported itself installed and served none, and its measurements were the joined path
  compared with itself.

  It is 4 ms per group **slower**, which is expected: the join was never the bottleneck. The value is entirely the
  constant shape, and collecting it needs one more step. A recording holds the addresses of a cache, and a cache is
  built per open context, so a recording still cannot cross contexts even with the shape settled. Reusing one page pool
  across contexts would finish the job and trades against holding nineteen contexts open at once, which is a decision
  rather than an optimisation.

* **what the shape constancy is worth**, and the graph measurement is what gives that a number. Its memory saving was
  measured and was zero, which is why deleting it was right on the evidence available. The reason to want it back is
  not memory: a page pool is a fixed allocation with the lengths carried in `seqused_k` as data, so **one graph would
  serve every context length** rather than one per open context -- which is the difference between a session workload
  getting 3.7x and every workload getting it.

## What a recording can and cannot outlive

A recording is worth 32.5 ms against 114, so the question of how long one lasts is the question of how often that
applies. Two things bind it, and only one of them turned out to be the allocation.

**A recording belongs to one context length.** Not to one shape -- to one length. The shapes are already arranged to be
constant: the widths are pinned to buckets and the allocation is now bucketed too, so a context of 900 tokens and one of
1,000 get the same 1,024-token cache. That is not enough, and the reason is arithmetic rather than bookkeeping: a
branch's tokens are written at an offset measured from the end of the context, and with pages that offset carries the
context length modulo the page size. A context thirteen tokens longer puts the branch in different slots, and the
recorded writes go to the slots from before.

Measured, on five documents of 251 to 302 tokens with one group each:

| | answers against eager | what the engine reported |
|---|---|---|
| before the length was checked | wrong by up to **0.15** | agreement to zero, nothing refused |
| after | **0.0 everywhere** | `the context is 276 tokens and this recording was taken at 263` |

That is the fourth time a check here reported agreement while the answers were wrong, and the pattern in all four is the
same: the check compared something narrower than the claim. This one only ever replayed on the context the recording
came from. `Recording.usable` now tests the length first, because a pooled cache's next context is rarely the same
length.

**Pooling the caches works and is not shipped.** A closed context returning its cache to a pool, reset for the next one,
is what would let a recording outlive one document -- and it was implemented and measured: five documents, one group
each, replaying from the third at 33.4 ms against 109 eagerly, answers identical. Then ten device tests failed.
`reset()` on the framework's own recurrent layers clears their contents and keeps their batch dimension, so a cache
returned by a three-row pass still holds three-row state and the next context's fork tries to widen three rows to
thirty-two. Making that work means owning the shape of state this package deliberately hands to the framework, so it
is its own change. The bucketed allocation stayed, because it costs nothing and is half of what the sharing needs.

So today a recording pays for a session asking many groups about one document, and for a sequence of documents that
happen to be the same length. It is refused by name otherwise.

## How these numbers were taken

Every figure above was measured on one borrowed L40S with the weights on it, which means a deployment step, and that
step produced three wrong answers before it produced a right one. What it takes to trust one of these numbers:

* **the source has to be the source.** The deployment untarred over an existing directory, so files deleted from the
  repository stayed on the machine -- a test file removed two commits earlier was still being collected. It now extracts
  into a fresh directory, and the fingerprint of every `.py` is compared on both sides before a number is believed.
* **the import has to resolve to it.** The package is installed editable against one directory while the clean copy sits
  in another, and `python3 script.py` does not put the working directory on the search path. Half a day of measurements
  were taken against the stale tree while the test suite, which runs with the working directory on the path, read the
  fresh one. Every measurement script now prints `prismyra.__file__` as its first line.
* **errors have to be visible.** `kubectl cp` was failing on file ownership with its output discarded, so the machine
  kept an older copy while reporting success. Suppressing that is what kept the first two invisible for as long as they
  were.

None of this is interesting engineering. It is here because two measurements in this document's history were taken
against code that does not exist in this repository, and a reader deserves to know which discipline produced the rest.

## Concurrency, as first measured

Eight requests arriving together, from a run predating the kernel work -- the shape is what matters, not the levels:

| | interleaved | queued |
|---|---|---|
| median latency | 2,113 ms | **945 ms** |
| first answer home | 2,113 ms | **214 ms** |
| requests per second | 3.8 | **4.8** |

Requests entering the model together are correct but slow in a particular way: they share one stream, so all of them
crawl and all finish late. A queue lets the first one leave first. `prismyra.queue.Worker` is that queue, and it is in
the core rather than in the server because this is a property of using one device from several threads.

Answers did not change at any concurrency tried: 0 of 8 differed.

**Not shipped, and recorded here because it says what the ceiling is.** Four users with different contexts and two
questions each cost 660.4 ms one at a time. Packing their contexts into one traversal brought that to 429.9 ms, and the
packed context pass alone is 90.1 ms against 4 x 90 ms separately; all 8 of 8 choices were unchanged against each user
run alone. `ask_many` in this package does none of that -- it answers requests one after another -- so the figures are
the size of the opportunity, not a claim about the code. The kernels already accept the sequence boundaries packing
needs.

## Reproducing this

On the machine the result file names:

```bash
pip install -e ".[fast,dev]"
prismyra-bench sweep --require-kernels
```

`--require-kernels` refuses to measure when a kernel is missing, so a slow run cannot be filed as a result. To check
against what is recorded:

```bash
prismyra-bench compare --against benchmarks/results/qwen3_6_35b_a3b_fp8__rtx_pro_6000.json
```

That exits non-zero on any point more than 10 per cent slower. It is one-sided: getting faster is not a regression, but
it is reported, because an unexplained improvement is usually a measurement that stopped measuring the same thing.

To refresh a file after a deliberate change:

```bash
prismyra-bench sweep --require-kernels --update benchmarks/results/<file>.json
```

That merges only the keys the harness measured. The baseline engine's numbers and the per-kernel profiles come from runs
against that other engine and are edited by hand, so a refresh cannot invent a baseline.

## What the harness measures, and what it does not

It measures the whole job end to end, in process: median of seven with the first three discarded. Device time only --
there is no transport in these figures. The first traversal of a shape pays for allocation and kernel selection, which a
served request does not, which is why the early samples go.

It does not measure the baseline engine, the per-kernel profiles, or anything about a second GPU. Those are separate
runs, recorded by hand, and the file says which is which.
