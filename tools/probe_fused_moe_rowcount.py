"""inv2 round4: isolated probe of vLLM's FP8 Triton `fused_experts` row-count sensitivity, with
VLLM_BATCH_INVARIANT=1 already set (matching the real engine's `_enable_batch_invariance()`).

Candidate from the task brief: "fused_moe の Triton (VLLM_BATCH_INVARIANT が効いているか)". The env var
makes `fused_moe.py`'s own `get_default_config` pick one fixed tile config regardless of M (prismyra's own
comment on this, `engine.py::_enable_batch_invariance`). What that does *not* necessarily fix: the Triton
MoE kernel still gathers tokens into **per-expert groups** before the grouped matmul, and the size of each
expert's group is a function of the whole pass's routing, not of a single row's content. A document's own
row can therefore land at a different offset inside its expert's group -- possibly a full tile in one pass and
a partial (masked) tile in another -- purely because *other* rows' routing changed how many tokens that
expert's group holds overall. That is the row-count-dependent-tiling shape of bug this project chases
everywhere else, one level below the "pick one fixed tile size" fix.

Method: fix row 0's hidden state and its own top-k routing (ids/weights), with random FP8 weights of the
real model's shape (256 experts, 2048 hidden, 512 moe-intermediate, block (128,128) -- values don't matter for
an invariance probe, only shapes/dtype do, so no real checkpoint download is needed). Vary total M in {1, 2, 3}
by appending companion rows with their own random routing (which can land on the same experts as row 0, the
realistic case for a shared pool of 256 experts and top_k=8 out of a handful of rows). Compare row 0's own
output slice across M with torch.equal.
"""

import os

os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch

from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

torch.manual_seed(0)
DEVICE = "cuda"

HIDDEN = 2048
INTER = 512
EXPERTS = 256
TOP_K = 8
BLOCK = (128, 128)
FP8 = torch.float8_e4m3fn


def make_weights(rng):
    w1 = (torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=rng) * 0.05).to(FP8).to(DEVICE)
    w2 = (torch.randn(EXPERTS, HIDDEN, INTER, generator=rng) * 0.05).to(FP8).to(DEVICE)
    w1_scale = torch.rand(EXPERTS, (2 * INTER) // BLOCK[0], HIDDEN // BLOCK[1], generator=rng).to(DEVICE) * 0.1 + 0.01
    w2_scale = torch.rand(EXPERTS, HIDDEN // BLOCK[0], INTER // BLOCK[1], generator=rng).to(DEVICE) * 0.1 + 0.01
    return w1, w2, w1_scale.float(), w2_scale.float()


def main():
    rng = torch.Generator().manual_seed(1)
    w1, w2, w1_scale, w2_scale = make_weights(rng)
    quant = FusedMoEQuantConfig.make(
        quant_dtype=FP8, block_shape=list(BLOCK), w1_scale=w1_scale, w2_scale=w2_scale,
    )

    row0_rng = torch.Generator().manual_seed(99)
    x0 = torch.randn(1, HIDDEN, generator=row0_rng, device="cpu").to(DEVICE, torch.bfloat16)
    ids0 = torch.randperm(EXPERTS, generator=row0_rng)[:TOP_K].to(torch.int32).to(DEVICE).unsqueeze(0)
    w0 = torch.rand(1, TOP_K, generator=row0_rng).to(DEVICE)
    w0 = (w0 / w0.sum(-1, keepdim=True))

    companion_rng = torch.Generator().manual_seed(555)
    baseline = None
    for M in (1, 2, 3):
        xs, ids_list, ws = [x0], [ids0], [w0]
        for _ in range(M - 1):
            cx = torch.randn(1, HIDDEN, generator=companion_rng, device="cpu").to(DEVICE, torch.bfloat16)
            cids = torch.randperm(EXPERTS, generator=companion_rng)[:TOP_K].to(torch.int32).to(DEVICE).unsqueeze(0)
            cw = torch.rand(1, TOP_K, generator=companion_rng).to(DEVICE)
            cw = cw / cw.sum(-1, keepdim=True)
            xs.append(cx); ids_list.append(cids); ws.append(cw)
        x = torch.cat(xs, 0)
        ids = torch.cat(ids_list, 0)
        weights = torch.cat(ws, 0)

        out = fused_experts(x, w1, w2, weights, ids, quant_config=quant)
        out0 = out[0].clone()
        if baseline is None:
            baseline = out0
            print(f"M={M}: baseline captured")
            continue
        eq = torch.equal(baseline, out0)
        diff = (baseline.float() - out0.float()).abs().max().item()
        print(f"M={M}: row0 torch.equal={eq}  max|diff|={diff:.3e}")


if __name__ == "__main__":
    main()
