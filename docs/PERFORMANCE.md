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
| convolution on a kernel written for it | 166.7 | 138.3 | 28.4, and see below |

**The chain is deliberately not continuous.** One step's "after" and the next step's "before" differ by a few
milliseconds, and both are printed rather than smoothed. Subtracting one column from the other across rows does not
work, and `benchmarks/check_results.py` enforces the gap staying small enough to be run-to-run variation -- a large gap
would mean the steps came from different configurations and do not describe one sequence.

**The convolution row above is wrong, and what replaced it is more interesting than the number.** The replacement is
installed on the framework's module-level name and acts only on weights this adapter tagged, and the tag was a Python
attribute -- but the layer does not pass the weight, it passes `weight.squeeze(1)`. A view is a new object carrying none
of the original's attributes, so the wrapper deferred to the framework on **every call** while `stats()` reported the
kernel as applied. The shape the wrapper saw, `(8192, 4)` against `nn.Conv1d`'s `(8192, 1, 4)`, is the whole diagnosis.

Tagged by data pointer now, which a view shares, and measured again on a 961-token context:

| | read |
|---|---|
| the borrowed convolution | 110.8 ms |
| the framework's own | 112.2 ms |
| the borrowed one again | 111.8 ms |

**It is worth 1.0 ms, not 28.4.** Which framework version the original figure belongs to is not known and is not worth
chasing; what is worth saying is that the figure survived in this document while describing a kernel that was not
running, and that the check which would have caught it -- a witness that the replacement actually served a call -- exists
for the paged path and did not exist here.

The kernel stays, and its value moved from milliseconds to capability: it takes `seq_starts`, so it does not convolve
across a document boundary, which is what makes [reading several documents in one pass](#reading-several-documents-in-one-pass)
possible. The framework's own has no such argument.

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

## Reading several documents in one pass

One card served 1.08 requests a second whatever arrived, and the cause was not the device. **The batch's whole width went
to questions about one document**, so a second caller waited for the first. vLLM's concurrency comes from the opposite
arrangement: one engine, one forward pass at a time, many requests' tokens packed into it.

`Prismyra(paged=True).open_batch([...])` does that here. Two halves, and they were built in that order because the second
one is where most of the time was:

* **answering** several documents in one pass. Rows are named per page, so a row's table names its own document's pages
  and nothing else. This is the third price of the paged path, after a memory saving that measured zero and the shape
  constancy a recording needs;
* **reading** them in one pass. The documents are concatenated into one flat run and the boundaries are carried as data.

The second half is where the numbers are, and the reason is that reading is almost all fixed cost:

| tokens | read |
|---|---|
| 289 | 113.2 ms |
| 545 | 113.2 ms |
| 1,057 | 116.0 ms |
| 2,081 | 132.5 ms |

which fits **110.1 ms plus 10.8 ms per thousand tokens**. At 289 tokens, 97% of reading a document is paying for kernel
launches rather than for the document, and reading eight of them one at a time pays that eight times.

Measured, eight documents of about 290 tokens with four questions each:

| | separately | together | |
|---|---|---|---|
| reading | 0.989 s | **0.154 s** | 6.4x |
| reading and answering | 1.968 s | **0.372 s** | 5.3x |
| documents per second | 4.06 | **21.52** | |
| questions per second | 13.72 | **72.64** | |

Answering alone was 1.77x; reading is what took it to 5.3x.

Three kernels have to be told where a boundary is, and **all three already take an argument for it**, which is why this
is wiring rather than a rewrite. The attention kernel takes `cu_seqlens_q` and `cu_seqlens_k`. The borrowed recurrence
takes `cu_seqlens`, and with it returns one final state per document -- so the argument that makes a batched read possible
is the same one that makes it useful. The convolution is this package's own kernel and takes `seq_starts`.

### Three things that had to be right, and two would not have raised

* **The convolution state was one row for the whole batch.** The framework builds it by slicing the end of the pass's
  input, and the end of a flat run is the end of the last document, so every row of every earlier document would have
  started its branch convolution from the last document's tail. The per-document tails are recorded as the convolutions
  run and written back afterwards, and a read that leaves any state with the wrong number of rows is refused by name.
* **The convolution replacement had never run at all.** See [above](#how-the-context-pass-got-from-288-ms-to-138-ms): the
  tag was a Python attribute and the layer passes a view. It is worth 1.0 ms rather than 28.4, and it is kept because it
  takes the boundaries.
* **The shared page region had no room for staging pages.** A document reserves one page for a leftover that does not
  fill a page, and the allocation counted only whole pages: two documents of 67 tokens need five pages each, and nine is
  what 134 tokens rounds up to.

The boundaries are ambient -- a module-level window the kernels read -- because the kernels are reached through the
framework's own module-level names and there is no argument to thread down from the engine. The window is a context
manager so it cannot be left open, and "no boundaries" is the only default, so a pass that forgets to open one reads a
single document, which is the behaviour that was already there.

A batch is text only for now: media widen a context's positions by a grid rather than by a token count, and a flat run of
several would need each document's own offset threaded through. It refuses by name when the borrowed convolution or
recurrence is not installed, because the framework's own have no argument for a boundary and would scan across it.

**What is not measured is a scheduler.** A batch that answers eight documents in one pass is not the same thing as a
server that fills one from arrivals, and the throughput above is what the mechanism allows rather than what a queue would
achieve.

## The scheduler

A batch is not a scheduler. `prismyra.schedule.Batcher` is one: a queue, one thread that owns the device, and a rule for
which of the waiting requests go into the next pass. Callers arriving together, each with its own document and four
questions, each waiting for its own answer -- the same closed-loop shape the figures above were taken with:

| callers | arm | seconds | requests / s | first answer | median | last answer |
|---|---|---|---|---|---|---|
| 1 | one at a time | 0.407 | 2.46 | 406.7 ms | 406.7 | 406.7 |
| 1 | batched | 0.262 | **3.82** | 261.7 ms | 261.7 | 261.7 |
| 2 | one at a time | 0.482 | 4.15 | 239.7 ms | 360.8 | 481.9 |
| 2 | batched | 0.289 | **6.92** | 288.8 ms | 288.8 | 288.9 |
| 4 | one at a time | 0.949 | 4.22 | 236.4 ms | 592.6 | 948.1 |
| 4 | batched | 0.297 | **13.45** | 296.2 ms | 296.9 | 297.0 |
| 8 | one at a time | 1.926 | 4.15 | 248.1 ms | 1,092.7 | 1,924.6 |
| 8 | batched | 0.387 | **20.68** | 383.1 ms | 385.4 | **386.3** |

**5.0x at eight callers.** The median answer goes from 1,093 ms to 385, and every caller's answer lands within 3 ms of
every other, because they are all in the same pass.

The first answer is later -- 383 ms against 248 -- and the obvious explanation for that is wrong, so it is worth being
precise. It is not the wait, which is two milliseconds. It is that the first caller now travels in a pass carrying eight
documents, which takes 386 ms, instead of being answered alone in a pass that takes 262. **The first request pays for the
throughput of the other seven.** That is the trade this scheduler makes, and it is the right one when callers are waiting
and the wrong one when the first caller's latency is the product.

### The wait, and the argument it overturned

The first version had no wait, on the reasoning that a batch fills from what has already arrived so waiting only helps
when the queue is nearly empty -- and that is when the device is not the bottleneck. **The measurement disagreed.** Eight
callers arriving together produced passes of one and seven documents, because requests that arrive together do not arrive
at the same instant: the first is some microseconds ahead, finds nothing waiting, and starts a pass alone. The queue was
not nearly empty; it was about to be full, and the scheduler could not tell the difference.

| | passes formed | eight callers |
|---|---|---|
| no wait | 1 document, then 7 | 640 ms, 1.85x |
| a wait | 8 | **387 ms, 5.0x** |
| the same eight in one pass, in a loop | 8 | 367 ms |

`linger_ms` is not how long a pass waits. It is **how long a pass waits with nothing arriving**, and it refreshes each
time a request joins, so a lone request waits two milliseconds and a burst is collected for as long as the burst lasts.
Bounded above by `Limits.worth_waiting_ms`, which is a pass's fixed cost taken from the fastest read this engine has
served: waiting longer than that cannot pay, because by then the pass could have run. Nothing waits at all before the
engine has read anything, because before then there is no evidence for any wait.

**A fixed wait was tried and broke, and what broke it is the useful part.** Tokenising each document once instead of twice
is an obvious saving -- 0.44 ms a document, twice a document, 7 ms a pass on the one thread there is only one of -- and
the obvious way to do it is on the caller's thread when the request is submitted. That made throughput **worse**: 20.37
requests a second fell to 13.44 and the passes split into two of four again. Encoding copies the token ids to the device,
and eight caller threads take their turns at that, so the arrivals spread wider than the wait. The work moved to where
there were more threads to do it on rather than to where it is done once. Encoding on the scheduler's thread, cached on
the request, gives 20.68 -- and a wait that a 0.44 ms change can break is tuned rather than derived, which is why the wait
now refreshes rather than counting down.

### Recording the batched pass is not the next thing, and why

A recorded pass replays for 0.684 of its cost at a short suffix, the batched answer pass is 289 ms of a 444 ms batch, and
the page pool exists partly so that a recording's addresses survive a change of document. So this looked like the next
thing. Two findings, and the second closes it for now.

**The batched pass never reaches the recording machinery at all.** `_answer_batch` calls the backbone directly rather than
through `_run_branch`, so `graphs=True` records nothing on this path: measured over eight batches, `graphs_verified` empty.
That is a wiring gap and wiring it is a few lines.

**But a recording would not be reusable across batches even wired.** A branch's tokens are written at an offset measured
from the end of its own document, and with pages that offset is the document's length modulo the page size. Those offsets
are computed in Python inside the forward pass, so a recording bakes in **one tuple of per-row remainders**. Two batches
share a recording only if every row's document has the same length modulo sixteen, which for eight rows is one arrangement
in 16^8.

Making the remainders constant means padding every document to a page boundary, and padding a context is not free: adding
whitespace to a context moved answers by 0.075 to 0.108 in a separate measurement, which is larger than the gap between
options on some questions. Left padding is the safe form and it is its own change with its own verification.

So the honest accounting is that the prize is about 91 ms of 444 -- **1.26x** -- and the price is a padding scheme that
touches answers. It is written down here rather than attempted, and the wiring gap is written down because it would
otherwise read as "recording does not help on batches", which is not what was measured.

### A shelf: documents that stay on the device

A `Batch` owns its cache, so asking twice about one document reads it twice and page reuse has nothing to reuse pages for.
`Prismyra.open_shelf()` is the other shape, and it is the server one: **the cache stays and the documents come and go.**

    shelf = engine.open_shelf()
    handle = shelf.put(document)          # one read
    shelf.ask({handle: questions})        # a branch pass, no read
    shelf.ask({handle: more_questions})   # another branch pass, still no read
    shelf.drop(handle)                    # the pages go back to the pool

What a shelf keeps per document is its pages and a snapshot of the recurrent state its read ended in. The pages are the
attention layers' share and live in the pool; the recurrent layers keep one state per **row** rather than per document, so
each document's state is kept aside and copied into its rows when it is answered. That snapshot is the same one a single
context has always taken, and the only new thing is that several are alive at once.

Five things had to be right and none of them would have raised on its own:

* **the counters, or the framework reads the next document as a continuation.** A layer picks its single-token path when
  the cache says it already holds tokens, so the second document put on a shelf went through the decode convolution and
  the boundary machinery recorded nothing;
* **the state has to come back to one row of zeros.** Setting the key to `None` reaches `torch.cat([None, ...])` and
  raises four frames down; removing the key raises `KeyError` in the same line; leaving the full-width buffer cannot be
  concatenated with a one-row read;
* **the framework prepends the convolution state it was holding**, convolves, and drops the prefix. On a fresh cache
  there is nothing to prepend; on a shelf there are `kernel - 1` tokens, so every document boundary after the first moves
  by that much. Without the shift the second document's convolution would begin three tokens inside the first;
* **the handles are not the positions.** The third document put on a shelf is handle 2 and position 0 of its own read, and
  taking the position overwrote handle 0 -- leaving the shelf holding one document under two names, which is the good
  case. The bad case is a free handle and two documents sharing a run;
* **a document being answered cannot be dropped**, because its run would be handed to the next document while a row's
  table still names it. The guard is cleared when a pass ends, in a `finally`, or a failed pass would leave a shelf unable
  to drop anything ever again.

### The read budget was a slope with no constant

Found by the shelf and worth more than the shelf. Reading a context transiently allocates, and admission modelled that as
**purely proportional to the tokens** -- which the two measurements behind it supported: 0.602 GiB at 3,040 tokens and
4.775 at 24,327, both 1.96e-4 GiB a token. A constant is invisible between those two points.

At thirty tokens it is not. Three reads of a short context left admission believing a read costs **32.6 MiB a token**,
which is 155 times the measured figure, and a shelf that would have held 65,536 tokens was refused above 1,024 with
"needs 23.1 GiB for reading it".

Two things were wrong and both are fixed:

* `cache_bytes` described the joined storage while the engine was paged. The pool's private region is sized for the worst
  remainder rather than a document's own, so it does not shrink with the context, and the difference was being attributed
  to the read's transient -- which is the thing divided by the tokens;
* the transient is now remembered as **a floor and a slope**. The floor comes from any read and the slope only from reads
  of at least one bucket, because below that the division measures the constant against an arbitrary number. The budget
  is the larger of the two rather than their sum: the floor already contains whatever proportional part the read it came
  from had.

`stats()["admission"]` reports both.

### A wider group does not buy throughput

The per-row cost of a branch pass is flat in the width -- 0.109 GiB at width 1 and at width 32 -- so a wider group looked
like free width. Sixteen callers, four questions each, one engine per group because the group is fixed when the cache is
allocated:

| group | requests / s | passes | mean documents per pass | held per context |
|---|---|---|---|---|
| 8 | 6.26 | 9 | 1.78 | 0.347 GiB |
| 16 | 7.32 | 5 | 3.20 | 0.674 GiB |
| **32** | **13.13** | 3 | 5.33 | 1.328 GiB |
| 64 | 9.07 | 2 | 8.00 | 2.637 GiB |

Group 64 forms wider passes and fewer of them and is still slower, which is the opposite of the premise. **The pass is not
where it loses.** The same eight documents, repeated five times each:

| group | cache | read | read and answer | answer |
|---|---|---|---|---|
| 32 | 4.1 ms | 155.1 ms | 444.5 ms | 289.3 ms |
| 64 | 4.6 ms | 149.2 ms | 433.3 ms | 284.1 ms |

Identical within run-to-run variation. So whatever the scheduler run measured is in the scheduling rather than in the
work, and **it is one run at each group against five repeats at the pass level, so the 13.13 against 9.07 is not a result**
-- it is a reason not to widen the group, which is a different and weaker claim. The default of 32 stays, the memory it
holds is half of what 64 holds, and the idea is closed rather than pursued: a wider group has to earn its memory with
throughput and it did not.

Three limits bound a pass, all read off the engine rather than configured: the questions, by the group; the documents, by
the group again, since every document needs a row; and the tokens, by what the pool holds. A request too wide for any pass
is refused when it is submitted rather than when it reaches the front, because a caller who will be refused should not
wait first.

**What this does not do.** Arrivals are closed-loop here, which measures the same thing the figures it is compared against
measured and not what an open arrival process would do. There is no priority and no deadline: the rule is first come,
first served within the limits. And a failure in one document fails every request in its pass, which is honest but coarse
-- the alternative is re-running the survivors, and that has not been built.

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

### The gigabyte the budget had never counted

The held figure was the key-value cache, and on this model the key-value cache is **ten of the forty-eight layers**. The
other thirty are gated delta net, and what they hold is not keys and values but a recurrent state and a convolution
window, one of each per row:

| | per row | at group 32 |
|---|---|---|
| recurrent state, 30 layers | 1.05 MiB | 33.6 MiB |
| convolution window, 30 layers | 0.01 MiB | 0.3 MiB |
| **total** | **1.06 MiB** | **33.9 MiB** |

That is a gigabyte per open context at the shipped group, and admission had been ignoring all of it since admission
existed. `cache.state_bytes` computes it from the model's own shapes -- value heads times value dim times key dim for the
recurrence, the kernel width for the window -- and `cache_bytes` includes it, so the number a refusal quotes is now the
number the pass allocates.

It went uncounted because nothing had ever asked where that state lives. The framework's layers **rebind** it on every
pass: each pass allocates a new tensor at the width it is running and drops the old one, so there was never an allocation
to attribute to the context, only a per-pass one that looked like part of the work. `fork.OWNED` changes that -- the
state is allocated once at the full group width when the context opens, and each pass takes a view of the first *rows* of
it. Two things follow. A recording can be taken against it, because its address no longer changes between passes. And
the gigabyte is now visibly held rather than invisibly transient, which is what made it possible to count.

Getting that accounting right took two wrong answers first, both of which refused honest work:

* the allocation happened inside the first branch pass, so it landed in the peak that pass was measured by. Divided by
  that pass's rows and multiplied by the group, a one-off gigabyte became a **24 GiB** answering estimate;
* then it was measured at the context's own length while being allocated at the bucket, so a context near the bottom of
  a bucket under-counted and one near the top over-counted.

Both are why the allocation is now at context-open time and the measurement is `room_for()`, not `len(tokens)`.

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

**1.9x behind, from 4.3x behind** -- at four questions per context, which is what a RACE article carries and which is
the wrong place to read that ratio. See the next section: the two arms cross at twelve.

Accuracy is not a like-for-like comparison and no claim is made from it: the arms score different tokens, since a
request returning one token can only score the letters while this package scores the option text. The token column is
what this package wins, by 2.7x -- it sends the context once per document where vLLM sends it once per question and
leans on the prefix cache not to recompute it.

### The ratio is a function, and the crossing is at twelve

Both arms cost something per context and something per question, and the constants and the slopes are different, so a
single question count reports one point on two lines. `--questions N` and `--sweep` fill each context up to N by
borrowing questions from the other contexts in the sample -- real questions with real suffix lengths, asked about the
wrong document, not scored, and counted in the throughput because answering one costs exactly what answering a real one
costs. Twenty RACE articles, one L40S, recording off in both arms:

| questions per context | fork q/s | vLLM q/s | ratio | fork ms/context | vLLM ms/context |
|---|---|---|---|---|---|
| 1 | 4.48 | 8.30 | 0.54 | 223.8 | 119.6 |
| 4 | 17.63 | 32.37 | 0.54 | 226.9 | 122.9 |
| 8 | 34.77 | 49.27 | 0.71 | 229.7 | 154.0 |
| 10 | 43.80 | 50.35 | 0.87 | 228.1 | 192.1 |
| **12** | **52.25** | **50.30** | **1.04** | 229.5 | 230.0 |
| 16 | 64.98 | 49.13 | 1.32 | 241.1 | 311.2 |
| 32 | 85.28 | 48.01 | 1.78 | 347.2 | 633.8 |
| 64 | 91.08 | 48.34 | 1.88 | 703.0 | 1293.8 |

Read the two `ms/context` columns rather than the ratio. **vLLM's cost is linear in the questions and this package's is
nearly flat**: from one question to sixteen, vLLM goes from 119.6 ms to 311.2 and the fork goes from 223.8 to 241.1 --
seventeen milliseconds for fifteen more questions, because they are fifteen more rows of one batch. vLLM's throughput is
the same 48 to 50 questions per second at every count, which is what a saturated engine looks like: it is doing the work
well and there is simply more of it.

So the two numbers to quote are the crossing and the asymptote. **Below twelve questions per context vLLM is the faster
way to ask, above twelve this package is, and the ratio tends to about 1.9x.** Neither of those is the 1.9x in the table
above, which was a coincidence of reading a limit at a point far below the crossing.

### And then the crossing went away

That whole table is one document per pass. With several documents in a pass -- see
[above](#reading-several-documents-in-one-pass) -- the crossing has nothing to cross, because the questions per pass no
longer come from one document. The same forty RACE articles, the same card, documents packed greedily up to the group:

| arm | questions asked | accuracy | questions / s | ms per context | tokens sent |
|---|---|---|---|---|---|
| this package, batched to 8 documents | 132 | 0.969 | **42.93** | 400.6 | **19,292** |
| vLLM, one request per question | 157 | 0.936 | 32.36 | 123.8 | 53,358 |

**1.33x, at four questions per document** -- the shape a RACE article has, and the shape where a single-document pass lost
by 0.54x. The tokens column is 2.8x, for the same reason as before.

Two things about that table. The arms drop their first unit as warm-up, and for the batched arm a unit is a whole batch,
so it discards eight documents' worth against vLLM's one -- which is why the asked counts differ, and the throughput is
per second so it is not distorted by it. And accuracy still is not a like-for-like comparison: the read-outs score
different tokens, which is said wherever this table appears and is not a claim.

One wrong row was published before this table was right: a point labelled 128 questions per context which was really
about eighty, because twenty articles cannot supply 128 questions for one context and `widen` truncated in silence. It
now refuses and says to raise `--limit`. The questions per second was correct; the axis it was plotted against was not,
which is the same failure as a check that compares something narrower than its claim.

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
  to six decimals. That table is at four rows and a short suffix, and **the 3.5x is a property of that shape rather than
  of the mechanism**, which is the next section and cost three attempts to establish.

  Off by default, and the reason is memory rather than doubt: a recording holds a private allocator pool, and this
  engine refuses a context by name from a budget it measures, so a feature that quietly takes device memory behind
  that budget would make the refusal wrong.

#### What a replay saves, and where it saves nothing

A recording removes the time the device spends waiting to be told what to do next. **Kernel launches are asynchronous**,
so once each kernel outlasts the call that launches it, the host is already ahead of the device and there is no waiting
left to remove. Measured on this model at thirty-two rows and a 3,000-token context, the same pass eagerly and replayed:

| suffix width | eager pass | replayed pass | ratio | recording cost | passes needed to pay |
|---|---|---|---|---|---|
| 16 | 109.4 ms | 74.8 ms | 0.684 | 493.6 ms | 17 |
| 32 | 110.7 ms | 94.7 ms | 0.855 | 462.1 ms | 40 |
| 64 | 144.2 ms | 142.1 ms | **0.986** | 562.4 ms | about 400 |
| 128 | 268.1 ms | 267.3 ms | **0.997** | 948.2 ms | about 2,000 |

The last column is deliberately not exact. It divides by one minus the ratio, so at a ratio of 0.997 a tenth of a
millisecond in either measurement moves it by two hundred: the run's own output printed 414 and 2,190 from its unrounded
timings where the rounded figures above give 411 and 2,009. **Near the point where a replay saves nothing, how many
passes would pay for a recording is not a quantity worth a precise figure** -- which is the same fact as the column
itself, seen from the arithmetic instead of from the card.

At a suffix of 128 tokens a recording is worth **0.8 ms of 268** and costs 948 ms to take. The 55.1 ms of kernel time
inside a 109.4 ms pass that motivated the whole mechanism was a short suffix, where the pass is host-bound; the same pass
at a longer suffix is device-bound, and there is nothing for a graph to do.

So the question "is this worth recording" cannot have a constant answer, and **two constants were shipped and measured
losing**:

| rule | what it predicted | what it measured, at 128 questions per context |
|---|---|---|
| record on a shape's second use | a saving | 47.21 questions/s against **102.07** with recording off |
| record when three more passes are coming | a saving | the same 2.16x loss |
| record when seven more are coming | a saving | 64.49 against **103.44**, at 256 questions per context |

The first rule recorded on the second sighting of a shape and there was usually no third, so every recording was taken,
proved and thrown away unused. The second and third came from a cost of "151.3 ms to record", a figure that cannot be
right and whose wrongness is only visible once the cost is counted in **passes**: three warm-up passes at 107.7 ms each
is 323 ms before the capture begins. Expressed in milliseconds the missing term was invisible. Expressed in passes it is
not expressible.

What ships instead measures. The engine times the eager pass it just ran and the proving replays it runs anyway, and
`graphs.keeping_pays` compares them: a recording is kept only if the passes still expected can pay for the four passes
and two proving replays it cost. A refusal names both figures --

    a replay of this shape costs 267.3 ms against 268.1 eagerly, so recording it pays from 2009 more passes and 7 are
    expected

-- and is remembered per shape, so finding out costs one recording per shape per engine rather than one per context.
`stats()["graphs_cost"]` reports the pair for every shape it measured.

Two figures supply "passes still expected", and neither is a guess: how many more groups of this shape the call in
progress will run, which a caller has already declared by handing over all its questions at once, and how many times the
shape has come back on this cache. **Neither can be the cost model.** That was the lesson: all three failures were a cost
model with a term missing or a term assumed constant, and none was a kernel behaving unexpectedly.

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
each, replaying from the third at 33.4 ms against 109 eagerly, answers identical. Three attempts have each found
something real and then failed:

| | what it found | state |
|---|---|---|
| first | `reset()` keeps the batch dimension, so a three-row cache met a thirty-two-row fork | fixed by `fork.OWNED`, shipped |
| second | the owned gigabyte landed in the first pass's peak and became a 24 GiB estimate | fixed, and the recurrent state is now counted at all |
| third | six device tests still refuse contexts on memory, with the accounting corrected | **not diagnosed** |

Each fix was worth shipping on its own -- the owned shape is what lets a recording exist at all, and the accounting was
wrong whether or not anything is ever pooled -- which is the only reason three failed attempts have left the package
better. A fourth should begin by measuring the held memory of a pooled engine rather than by writing more of it.

So today a recording pays for a session asking many groups about one document, and nothing else. It is refused by name
otherwise.

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
