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
this family's linear-attention layers carry 32 value heads, neither of which is the number in this formula. Taking one of
those instead overstates the cache by 8x or 16x.

`group` is therefore a memory dial as well as a cost dial: it divides the memory exactly and multiplies the traversals by
`ceil(questions / group)`. `Prismyra.cache_bytes(tokens)` reports the figure for a given configuration, and
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

## Concurrency

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
packed context pass alone is 90.1 ms against 4 x 90 ms separately; all 8 of 8 choices were unchanged against each user run
alone. `ask_many` in this package does none of that -- it answers requests one after another -- so the figures are the
size of the opportunity, not a claim about the code. The kernels already accept the sequence boundaries packing needs.

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
