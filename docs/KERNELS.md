# The kernels

Prismyra does not own the model's forward pass. The framework's implementation stays in place and specific modules are
replaced inside it. That is deliberate: it keeps the original available as the reference every replacement was verified
against, so a change does not remove its own oracle.

An adapter declares which architectures it handles, and two separate things are checked before it touches anything.
First, whether this checkpoint's *shape* is the one the adapter was measured against -- expert count, attention heads,
key-value heads, and nothing that depends on how many layers there are. A checkpoint cut down to fewer layers, or
grown to more, still has the same per-layer shape, so it still matches: depth is deliberately absent from this check.
Second, once a swap runs, how many modules it actually replaced -- and that total *is* layer-count-dependent (one
routed-expert block per layer, for instance), computed from the checkpoint's own layer count rather than assumed, so
it is exact at any depth. A different total than expected means the swap matched something it should not have, or
missed something it should have caught, and it fails rather than leaving the slow path silently in place -- a swap
that matches nothing looks exactly like a swap that worked.

## Verifying every kernel applies, end to end

`tests/test_gpu.py::test_require_kernels_starts_with_nothing_skipped` builds the engine with `require_kernels=True`
against a real checkpoint on a real device and asserts nothing was skipped -- the regression test for this, run on
the machine that measures. It does not install the package the way a user does, though, so the check that matters
for a release is the same thing from a clean install:

```bash
python3 -m venv .verify && source .verify/bin/activate
pip install "prismyra[server,fast] @ git+https://github.com/littlemex/Prismyra@<tag>"
prismyra-serve --model <a checkpoint this adapter recognises> --require-kernels &
curl -s localhost:8000/health   # {"ok":true,"depth":0} once the weights are loaded
curl -s localhost:8000/stats | python3 -c 'import json,sys; print(json.load(sys.stdin)["engine"]["kernels"])'
```

`kernels.complete` should be `true` and `kernels.skipped` empty. If it is not, the message `--require-kernels` refused
to start with says which kernel and why, and the two sections below -- particularly the note on head duplication --
are where that reason is explained.

## The same answer in every process: autotuned kernels are pinned

Several kernels on the read path are Triton kernels with `triton.autotune`: the first call in a process times each
candidate configuration and keeps the fastest. When candidates are close, the winner is decided by timing noise, and
some candidates do not compute the same floating-point sums. On an L40S, the flash-linear-attention kernel that inverts
the gated-delta layers' triangular blocks (`merge_16x16_to_64x64_inverse_kernel`) picks `num_warps=2` on about one cold
start in five and `num_warps=4` otherwise, and the two give probabilities up to 0.59 apart on single questions --
identical code, weights, driver and card, answering differently from one process to the next.

So at start the engine holds every autotuner in the process to one configuration and clears any choice already made
(`prismyra.kernels.autotune`). The configuration comes from a file per GPU generation shipped with the package,
`prismyra/kernels/pinned/sm_<major><minor>.json`, which names each kernel's configuration and says how it was chosen.
A generation with no file, or a kernel the file does not name, keeps its **first** declared candidate: still the same
in every process, possibly not the fastest, and listed under `engine.stats()["autotune"]["fallback"]`. Pinning is on
by default; `Prismyra(..., pin_autotune=False)` turns it off.

The right configuration depends on the generation, which is why it is data. On an RTX PRO 4500 (sm_120) the same
inverse kernel's timing picks `num_warps=2` every time, and there 2 and 4 give the same outputs; and the FP8 block
linear's `num_stages=4`, the fastest at the branch's shape, needs more shared memory than that card has at longer
shapes, so its table names `num_stages=2`. An entry must be valid for every shape the kernel sees -- the cold-start
test (`tests/test_gpu_cold_start.py`) runs the read path and fails on a configuration the card cannot launch.

On an L40S, pinned and unpinned answer at the same speed (median 94.8-95.0 ms against 96.0-96.6 ms over 400
questions, two cold starts each), and the pinned engine no longer spends the first call timing candidates.

To add a generation: run the read path once with pinning off and an empty `TRITON_CACHE_DIR`, take each kernel's
fastest configuration from the cache's `*.autotune.json` records (summed over the keys it recorded), check it is valid
at every shape, write the file, and run the cold-start test.

## What each replacement is worth

Measured on one context of about 5,000 tokens. Each row is its own paired run -- the same process with and without that
one change -- so the chain is not continuous and both ends are given.

| replacement | before | after | predicted |
|---|---|---|---|
| routed experts on a fused kernel | 288.1 ms | 208.3 ms | 81 |
| dense projections on a block-scaled fp8 kernel | 204.5 ms | 185.7 ms | 22.2 |
| normalisation on a faster kernel | 185.2 ms | 174.7 ms | 10.5 |
| head duplication deleted | 175.1 ms | 166.6 ms | 9 |
| convolution on a Triton kernel | 166.7 ms | 138.3 ms | 21.6, withdrawn -- see below |

Against vLLM doing the same work on the same card, three of the five now win:

| | Prismyra | vLLM |
|---|---|---|
| convolution, 30 layers | **3.0 ms** | 3.37 ms |
| recurrence, 30 layers | **18.9 ms** | 21.0 ms |
| attention, 10 layers | **6.8 ms** | 8.13 ms |
| routed experts, 40 layers | 29.5 ms | 29.97 ms |
| dense projections | 32.7 ms | **21.65 ms** |

## Notes on the ones with a catch

**Routed experts.** The framework groups tokens by expert by materialising a permuted copy and scattering the result
back -- about 166 MB each way per layer at this context length, 13 GB across forty layers. The fused kernel reads each row
through the routing index instead, so the copy never exists. This was the largest single change and it is not a faster
kernel doing the same work; it is less work.

**Head duplication.** The linear-attention layer duplicates query and key from 16 heads to 32 because the value side has
32. The framework's own chunked-matmul fallback cannot take the two head counts as given -- it needs the duplication --
but the borrowed recurrence kernel below can, and returns a **bit-identical** result once it is installed, so the
duplication is then work whose output is discarded. It is disabled by setting an attribute the layer reads only to make
that decision, which is a flag set through a name that no longer describes its value -- so the adapter installs the
recurrence kernel first and only then runs one layer both ways, against whichever implementation will actually be
called, and requires the outputs to match exactly before applying it anywhere. Probing before that kernel is installed
checks the wrong implementation and fails on every measured framework version, not only a newer one.

**Convolution.** Five routes were measured before the sixth worked. The framework falls back to a general
two-dimensional convolution; `F.conv1d` with per-channel groups is 0.80x that, a token-major variant 0.43x, four scaled
copies summed 0.34x. vLLM's variable-length kernel is **wrong** driven with the arguments its signature marks optional --
1.6e-01 to 4.4e-01 against a float32 reference, and infinite at 100 tokens -- because those arguments carry the paged
state its own model code maintains. So it is written in Triton: **8.1x** against what the layer paid, and **more accurate**
than the kernel it replaces (1.7e-03 against 2.4e-03 relative to float32). It reads a token-major tensor, and since the
layer's transpose is a view, consuming it directly removes a copy as well -- which is why the measured saving beat the
prediction.

**Attention.** The branch pass is where this matters. With a cache present the framework materialises an additive mask,
the fast kernel cannot take one, and attention drops to a memory-efficient kernel built for an older architecture: 43.3 ms
over ten layers against 10.9 ms. No mask is needed, because every branch's queries are the last positions of its own
sequence.

**Gated normalisation.** The gated delta net's output normalisation, `rms_norm(x) * weight * silu(gate)`, runs in thirty
layers of every pass and was eight elementwise kernels in the framework. Profiled at 64 questions about a 5,335-token document on an
L40S, those kernels were about 50 ms of kernel time. `FusedGatedRMSNorm` is one Triton kernel doing the same arithmetic in
the same order -- float32 accumulation, a round to bfloat16 before the weight, another after it, the gate in float32, one
final round -- and it agreed with the module it replaces to **0.0** on the verification input. The request went from
792 ms to 712 ms and one question from 435 ms to 380 ms (docs/PERFORMANCE.md): more than the kernel time, because the
launches went with them.

**Dense projections.** Still behind. vLLM reaches a CUTLASS path that wants a scale layout this checkpoint does not store.
Both kernels sit the same distance from a float32 reference (2.58e-02 against 2.64e-02), and that distance is dominated by
quantising the activations, not by either kernel -- so the swap is not a loss of accuracy, it is a different rounding.

## Routed experts in NVFP4 (Blackwell only, optional)

A second routed-expert path, `prismyra.kernels.nvfp4`, alongside the fused FP8 kernel above rather than replacing it:
selected per process with `PRISMYRA_EXPERTS=nvfp4`, and only usable on a GPU with native 4-bit floating-point tensor
cores (compute capability 12.0, "sm_120", e.g. the RTX PRO 4500) -- it is not a candidate for the L40S/H100 cards the
rest of this document measures on, and has not been measured on datacenter Blackwell ("sm_100", B200/GB200) either.
Where `PRISMYRA_EXPERTS=nvfp4` is set but the required vLLM kernels are not available, the module raises rather than
silently falling back to the FP8 path above; this is untested outside the one sm_120 card this document measures on.
The weights come from outside the checkpoint's own FP8 tensors: a side file (`PRISMYRA_NVFP4_EXPERTS`) holding every
layer's experts pre-converted to NVFP4 by `prepare_layer`, plus a calibration file (`PRISMYRA_NVFP4_CALIB`) giving
each layer's activation maxima, loaded with `tiny_experts` so the framework's `from_pretrained` accepts a checkpoint
whose index leaves the routed-expert keys out entirely rather than erroring on them. `engine.py` runs this conversion
before the adapter above runs, and the adapter recognises the already-converted `FusedExpertsFp4` modules instead of
re-wrapping them in the FP8 path -- the only place the two paths touch.

Why a 36-layer checkpoint needs this at all: the routed experts are most of a layer's weight, and 36 layers of them in
FP8 (about 29 GB) do not fit one 32 GB Blackwell card beside everything else a context pass needs -- the checkpoint
this project would otherwise serve on such a card is cut to 32 layers. Converting only the experts to NVFP4 takes
them to about 16 GB and lets the full 36 layers fit, at a measured accuracy cost indistinguishable from noise against
the un-quantised FP8 checkpoint (−0.07 points on a 1,400-question set, 95% interval [−0.86, +0.79]) and 1.86 points
ahead of the 32-layer checkpoint it would otherwise be compared against; that accuracy measurement predates this
checkpoint's `prismyra-serve` integration and used a different serving path than the speed figures below.

Speed, measured through `prismyra-serve` (median of 5 runs after 2 warm-up calls, one race-comprehension document of
about 5,300 tokens, "N questions" meaning N questions asked about that document in one call), same card, with this
project's own tuned FP8 kernel tables installed: one question in 279 ms against the 32-layer FP8 checkpoint's 268 ms,
sixteen in 411 ms against 393 ms -- about 4% behind at both widths measured. The 32-layer FP8 checkpoint could not
complete the sixty-four-question measurement at all on this card (it ran out of device memory; the NVFP4 checkpoint
answered in 743 ms). The gap's leading suspect is the NVFP4 MoE kernel's own per-shape autotuning (`autotune_tactics`):
its tuning buckets are powers of two, and a context pass's internal chunk sizes are not, so some shapes fall back to
an untuned tactic (`falling back to runner=MoERunner tactic=-1` in the kernel's own log) every time they occur. Not
yet closed.

## Measured and rejected

Recorded because they are cheap to re-propose:

- **Merging the per-layer projections.** Each linear-attention layer multiplies the same input by four matrices, so
  merging them is arithmetically identical. Bit-identical, and worth **+0.8 ms of 208**. The call count was never the
  problem.
- **`torch.compile`.** With a fixed shape it genuinely fused -- launches 10,204 to 5,872, copy calls 1,011 to 421, 56
  fused kernels generated -- and the clock did not move: 292.0 ms to 295.1. Fewer, larger copies move the same bytes.
- **fp4 weights.** 11% *slower* at every length. Weight traffic is 22 ms of a 138 ms pass, so halving it caps the gain at
  11 ms before subtracting the cost of reconstructing scales.
- **Dequantising to bfloat16.** Worth 21 ms on the dense projections and **-38 ms on the experts**, so a net loss. The
  part with enough arithmetic per byte wants narrow weights; the part without pays for the conversion.
- **CUDA graphs.** One recording works (83.9 ms to 51.7); a second in the same process faults on replay and the reason was
  never found. Not shipped.

## The convolution's figure is withdrawn

The 21.6% above describes a kernel that was not running. The replacement acts only on weights the adapter tagged, the tag
was a Python attribute, and the layer passes `weight.squeeze(1)` -- a view, with none of the original's attributes. So
every call fell through to the framework while `stats()` reported the kernel as applied.

Tagged by data pointer now. Re-measured on a 961-token context: **110.8 ms with the borrowed kernel against 112.2 ms with
the framework's own, so 1.0 ms rather than 28.4.** The original figure may have been taken on a framework version that
passed the weight itself; that is a guess and is labelled as one.

The kernel is kept for something the framework's own cannot do: it takes `seq_starts`, so it does not convolve across a
document boundary, and that is what lets several documents be read in one pass.
