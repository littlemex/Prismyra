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

**The `w8a8_block_dynamic_fp8_matmul_kernel` entry above is stale against the vLLM version this project pins today**
(checked by name against `vllm==0.27.1`'s source: no `triton.autotune`-decorated kernel by that name exists any more).
What the entry's own comment describes -- `num_stages=4` winning at the branch's shape, `num_stages=2` the only one
that fits every shape's shared memory -- is real, but it now happens one layer down, in `w8a8_triton_block_scaled_mm`
(`prismyra/kernels/fp8_tuning.py` has the detail). That function is a plain `@triton.jit` kernel, not a
`triton.autotune` one, so `autotune.py`'s pinning (above) never touches it; it looks up its tile size from a static,
per-shape, per-row-count table that vLLM reads from inside its own installed package. This project's own tuning
script writes that table straight into the pip-installed path, so it is real on whichever machine ran it and
**absent on every machine built since** -- a fresh install silently falls back to one untuned tile for every row
count (vLLM logs `Using default W8A8 Block FP8 kernel config` once per shape) until `fp8_tuning.install()` copies
this project's own shipped copy (`pinned/fp8_block_configs/`) into place. `kernels.apply()` calls it before any dense
FP8 matmul runs. Measured, RTX PRO 4500, this is worth 2.7-2.9% on top of the autotune pin above (276.1-277.2 ms
against 283.6-285.2 ms over three interleaved pairs, 1-question read); the per-shape, per-row-count tile choice itself
was checked `torch.equal` against the untuned fallback at all 70 (shape, row-count) cells this checkpoint uses before
any file in `pinned/fp8_block_configs/` was kept, so this is purely a speed change, not an accuracy one.

The dense-projection fusion below adds three shapes (N=9,216, 12,288 and 1,024, all K=2,048) that the table above did
not have cells for, so a fresh install used to fall back to the same untuned tile for all three until a dedicated
sweep (checked `torch.equal` against the untuned default before timing anything, not the other way around) found a
matching candidate for every cell and added it to `pinned/fp8_block_configs/`. Run first for the RTX PRO 4500
(sm_120), then repeated for the L40S (sm_89) within the same release, so both cards now carry tuned tables for all
three shapes. This project's own measurements still show the fusion gaining less on an L40S than on an RTX PRO 4500
at context width, same checkpoint, three rounds each alternating against `origin/main` (RTX PRO 4500: -1.5% to
-2.3% across the widths it was asked about; L40S: a smaller gain at some widths and one measurement within this
project's noise bar) -- with the per-shape tuning table now ruled out as the explanation on both cards, since both
have one.

## The same answer in every process, part two: swap self-checks are seeded too

The pinning above covers kernels whose *configuration* a timing race can pick differently. A second, independent
gap had the same symptom -- `tests/test_gpu_cold_start.py::test_cold_starts_pin_the_nvfp4_tactic_too` failing on
`nvfp4-36l`, on the RTX PRO 4500, with the NVFP4 tactic itself confirmed pinned in every failing run -- but a
different cause: whether a kernel *replacement runs at all*.

`_compare` (used by `_swap_and_verify` for the "dense_matmul" and "norm" swaps) and `_swap_gated_norm` each decide
whether to install a replacement by running both the framework's own implementation and the replacement on one
random probe and requiring they agree within a tolerance (5e-2 for "dense_matmul", `2 * BF16_ULP` for "norm" and
"gated_norm"). The probe used to be unseeded -- drawn fresh from the process-global RNG state on every construction.
Measured directly on `nvfp4-36l`'s own weights: four independent, unseeded draws of "dense_matmul"'s probe measured
2.912e-02, 3.893e-02, 3.968e-02, and 5.018e-02 against that 5e-2 tolerance -- the fourth is *over* it. A process whose
draw lands under the line installs `Fp8Linear` (and, downstream, the three `dense_fusion` groups that depend on it
being installed) for every dense FP8 projection in the model; a process whose draw lands over the line keeps the
framework's own implementation instead, which rounds at a different point (`_swap_and_verify`'s own docstring has
the detail) -- a difference compounded across every dense projection in all 36 layers, large enough to flip which
of two probabilities a question's answer reports.

The fix seeds both probes from a value derived from the module being checked (the same pattern `onepass.prove`
already uses for its own bucket proof, and `_delta_inputs` for the `gated_delta_rule` swap's), and widens
`_compare`'s probe from 4 rows to 64 (matching `_swap_gated_norm`'s own row count) to shrink the measurement's own
spread. Re-measured five independent times after the fix: all five gave the identical 2.820e-02 for "dense_matmul"
on `nvfp4-36l`. `test_cold_starts_pin_the_nvfp4_tactic_too` passed on seven independent cold starts after the fix
(it had failed within five before it, and within seven after fixing a narrower, unrelated cause two releases
earlier -- see this file's own autotune-pinning section and `kernels/nvfp4.py`'s tactic table for that one).

Checked and ruled out directly, not assumed, before landing on this: the NVFP4 GEMM tactic itself
(`engine.stats()["nvfp4_tactics"]` reported `pinned: True` in every failing run); its workspace buffer (sized once
for the largest profiled bucket and confirmed, by direct computation of `cutlass_fused_moe_workspace_size`, never to
need re-growing at the token counts this test uses); every Triton autotuner `pin_autotunes()` is supposed to cover
(`engine.stats()["autotune"]` showed every expected kernel name pinned-from-table or deterministically fallen-back
in three independent constructions); the GDN fused-norm kernel itself in isolation (called directly, no model, no
engine, fixed seeded input: identical across five independent processes); and `onepass.install_islands()`'s
module-forward monkey-patch (an early, false lead from a too-small sample -- ruled out once a larger sample showed
the same split with and without it).

Why `fp8-36l` never showed this: the same two checks measured 5.747e-03 ("dense_matmul") and 5.208e-03 ("norm")
against the same tolerances on that checkpoint's own weights -- comfortably clear of the "dense_matmul" line
(11% of the budget), closer than that on "norm" (67%). `engine.applied.notes` now says so whenever a passing
measurement uses more than half its tolerance (`MARGIN_WARN_FRACTION`, `kernels/qwen3_moe.py`), on either
checkpoint, so a future checkpoint drifting toward either line is visible before it crosses one rather than
after. The two checks that already demand bit-identical agreement rather than a tolerance (`_fuse_pair`'s
fused-vs-separate check, `_probe_head_duplication`'s before-vs-after check) are not at this risk: a probe's specific
values cannot move an exact-equality decision the way they can move one measured against 5e-2 or `2 * BF16_ULP`.

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

**Fusing the projections that share an input.** Three groups each call the same per-model dense-FP8 kernel more than
once on the same input: attention's `q_proj`/`k_proj`/`v_proj`, the gated delta net's `in_proj_qkv`/`in_proj_z`, and a
shared expert's `gate_proj`/`up_proj`. `_FusedDenseProjection` concatenates each group's weights and scales once at
construction, quantises and runs one matmul instead of several, and splits the result back out -- verified
bit-identical against the separate calls it replaces (`engine.applied.verified["dense_fusion"]`, checked on every
group this checkpoint has, on both checkpoints this verification ran against, `fp8-36l` and `nvfp4-36l`). The
branch-width pass this helps (16-32 rows) is 32-54% faster per group; the context-width pass (several thousand rows)
is a real gain for the shared expert's group (+17.0%), a wash for attention's (+0.5%, within noise), and a measured
loss for the gated delta net's group (-9.4%), because that group's fused call's wider output picks a worse matmul
tile than either separate call does on its own. Fusing and not fusing are bit-identical either way, so which one
runs is chosen on speed alone, for every group alike rather than only the one that loses: at or below
`FUSION_MAX_ROWS` rows (512, the largest width this project's own packing ever hands a branch pass) the fused call
runs; above it, each slot falls back to its own original separate call instead -- decided independently from each
slot's own input, with no coordination needed between the slots in a group. `PRISMYRA_WITHOUT=dense_fusion` disables
fusion outright, at both widths.

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
FP8 (about 29 GB) do not fit one 32 GB Blackwell card beside everything else a context pass needs -- the FP8
checkpoint this project would otherwise serve on such a card (the 36-layer or 40-layer one) is cut to 32 layers
instead. Converting only the experts to NVFP4 takes them to about 16 GB and lets the full 36 layers fit, at a
measured accuracy cost indistinguishable from noise against this checkpoint's own FP8 weights before that conversion
(−0.07 points on a 1,400-question set, 95% interval [−0.86, +0.79]) and 1.86 points ahead of the 32-layer FP8
checkpoint it would otherwise be compared against; that accuracy measurement predates this checkpoint's
`prismyra-serve` integration and used a different serving path than the speed figures below.

Why the 32-layer FP8 checkpoint (the one FP8 checkpoint that does fit this card) is not used as the speed comparison
here: it cannot complete a sixty-four-question request at all on this card (device memory runs out), where the
36-layer NVFP4 checkpoint can, so a width-for-width comparison between the two is not available at the width most
exposes a difference, and this project no longer treats a narrower comparison (one or sixteen questions only) as a
usable stand-in for it. Speed is instead tracked release to release: this checkpoint, this card, three rounds
alternating against `origin/main` (one document built by joining RACE articles until it reaches about 5,300 tokens,
median of 5 runs after 2 warm-up calls each round, this project's own tuned FP8 kernel tables installed) -- one
question 1.47% faster, sixteen 2.34% faster, sixty-four 3.18% faster, with the alternating rounds' own min-max ranges
not overlapping at any of the three widths. `interleaved_fork` was off (this release's old default) on both sides of
that comparison; the dense-projection fusion above is this project's leading explanation for the gain, since it is
the one change in this comparison not specific to NVFP4, but the comparison itself is release to release and so also
carries whatever else changed in between. `interleaved_fork`'s own effect is measured separately, on top of this,
with the fusion already in place on both sides: toggled on one already-built engine rather than against a separate
`origin/main` checkout, fifteen alternating rounds, min-max ranges not overlapping at either width reported, it adds
a further 5.9% at sixteen questions and 1.3% at sixty-four.

The NVFP4 MoE kernel's own per-shape autotuning (`autotune_tactics`) used to profile a different tactic per
power-of-two token-count bucket, which is exactly the row-count-dependent-algorithm shape of bug this project
guards against everywhere else: two passes carrying the identical real row at a different total row count could
cross a bucket boundary and get a different tactic, and so a different, non-bit-identical answer for that
unchanged row. Within this same release, this is now pinned to one bucket (at this checkpoint's largest expected
row count, `round_up=True`), so every real call -- a small questions-only pass or a full `open_batch` -- maps to
the same profiled tactic by construction, closing the gap rather than bounding it. The cost is that a small pass
now runs the tactic chosen for the largest one rather than its own dedicated choice; this project's own measurements
of that cost are recorded against the row-count-invariance fix itself, not assumed.

That closed the gap *within* one process. It did not close the one *between* processes: the one bucket's tactic
is still chosen by timing on first use, so two processes with no shared cache file could pick differently if
FlashInfer found two tactics near-equal there -- measured directly: the same release, as two separate processes
with no `PRISMYRA_NVFP4_TACTICS`, answered the same request bit-for-bit differently in 81 of 81 entries (max
move 0.334). This release ships one more table the same way `kernels/fp8_tuning.py` ships the dense-FP8 matmul
tiling: already-measured tactics for the card this checkpoint serves on (`kernels/pinned/nvfp4_tactics/`), read
automatically and read-only (a process whose FlashInfer/CUDA/cuDNN build does not match the table's falls back to
timing for itself, warns, and never overwrites the package's copy). `engine.stats()["nvfp4_tactics"]` reports
which of the two happened. A second, independent source of the same symptom was found verifying this: the
per-thread scratch buffer `FusedExpertsFp4._workspace()` reuses across calls was allocated with `torch.empty`,
so its first-allocation content was whatever this process's own CUDA allocator history happened to leave there,
and that leaked into the answer -- unrelated to which tactic ran. Confirmed by toggling `PRISMYRA_NVFP4_WORKSPACE`
(the kernel's own per-call scratch does not show the effect) and closed the same way padding is handled everywhere
else in this project: `torch.zeros` instead of `torch.empty`, paid at each buffer's first allocation (one buffer
per device, fusion mode, and thread, so normally once per process) rather than on every request.

See [docs/PERFORMANCE.md's settings table](PERFORMANCE.md#four-speed-settings-one-that-risks-the-answer) for
`interleaved_fork` and `wide_group`, the two flags that change how a multi-question request reaches these kernels.

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
