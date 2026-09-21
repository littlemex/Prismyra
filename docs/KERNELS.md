# The kernels

Prismyra does not own the model's forward pass. The framework's implementation stays in place and specific modules are
replaced inside it. That is deliberate: it keeps the original available as the reference every replacement was verified
against, so a change does not remove its own oracle.

An adapter declares which architectures it handles and how many modules of each kind it expects. A different number means
a different model, and it fails rather than leaving the slow path silently in place -- a swap that matches nothing looks
exactly like a swap that worked.

## What each replacement is worth

Measured on one context of about 5,000 tokens. Each row is its own paired run -- the same process with and without that
one change -- so the chain is not continuous and both ends are given.

| replacement | before | after | predicted |
|---|---|---|---|
| routed experts on a fused kernel | 288.1 ms | 208.3 ms | 81 |
| dense projections on a block-scaled fp8 kernel | 204.5 ms | 185.7 ms | 22.2 |
| normalisation on a faster kernel | 185.2 ms | 174.7 ms | 10.5 |
| head duplication deleted | 175.1 ms | 166.6 ms | 9 |
| convolution on a Triton kernel | 166.7 ms | 138.3 ms | 21.6 |

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
32. The recurrence handles the mismatch itself and returns a **bit-identical** result, so the duplication is work whose
output is discarded. It is disabled by setting an attribute the layer reads only to make that decision, which is a flag
set through a name that no longer describes its value -- so the adapter runs one layer both ways and requires the outputs
to match exactly before applying it anywhere.

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

**Dense projections.** Still behind. vLLM reaches a CUTLASS path that wants a scale layout this checkpoint does not store.
Both kernels sit the same distance from a float32 reference (2.58e-02 against 2.64e-02), and that distance is dominated by
quantising the activations, not by either kernel -- so the swap is not a loss of accuracy, it is a different rounding.

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
