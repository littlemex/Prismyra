"""inv2 round4: isolated probe of the dense FP8 projection path (`Fp8Linear.forward` in
prismyra/kernels/qwen3_moe.py: `per_token_group_quant_fp8` -> `w8a8_triton_block_scaled_mm`) for row-count
sensitivity, with VLLM_BATCH_INVARIANT=1 set.

Candidate: every dense projection in every decoder layer (q/k/v/o_proj, shared-expert gate/up/down, and
anything else quantised to FP8 that isn't the router) goes through this path, not through
`torch.nn.functional.linear`/`aten::mm`/`addmm` -- `w8a8_triton_block_scaled_mm` is `vllm`'s own custom Triton
kernel, called directly, so `engine.py::_enable_batch_invariance()`'s dispatcher override (registered on
`aten::mm`/`addmm`/`matmul`/`linear`) never sees it and cannot make it row-count invariant. fp4spd's own
finding (RUN-fp4spd.md, S3) independently discovered that this exact kernel's tile config comes from vLLM's
`get_w8a8_block_fp8_configs()` -- a config table keyed **by the row count M itself** ("行数ごとの設定表").
If that lookup picks a different tile/split-k config for M=2 than for M=3, every dense projection in every
layer gets a different reduction order for the *same* row's own input, with no dispatcher-level fix touching
it -- exactly the row-count-dependent-algorithm shape of bug this project chases, and exactly the "RMSNorm や
スケールの dynamic per-token の計算" / implicitly the dense-FP8-GEMM-config category the task brief names.

Method: fix row 0's input vector and the dense layer's (random, fixed) FP8 weight+scale, call
`per_token_group_quant_fp8` + `w8a8_triton_block_scaled_mm` with total M in {1, 2, 4, 8, 16, 17, 32, 33} (a
spread crossing several of vLLM's own row-count config buckets), companion rows random. Compare row 0's own
output slice across M with torch.equal.
"""

import os

os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)

torch.manual_seed(0)
DEVICE = "cuda"
HIDDEN_IN = 2048
HIDDEN_OUT = 2048
BLOCK = (128, 128)
FP8 = torch.float8_e4m3fn


def main():
    wrng = torch.Generator().manual_seed(7)
    weight = (torch.randn(HIDDEN_OUT, HIDDEN_IN, generator=wrng) * 0.05).to(FP8).to(DEVICE)
    scale = (torch.rand(HIDDEN_OUT // BLOCK[0], HIDDEN_IN // BLOCK[1], generator=wrng) * 0.1 + 0.01).to(DEVICE).float()

    row0_rng = torch.Generator().manual_seed(13)
    x0 = torch.randn(1, HIDDEN_IN, generator=row0_rng, device="cpu").to(DEVICE, torch.bfloat16)

    companion_rng = torch.Generator().manual_seed(321)
    baseline = None
    for M in (1, 2, 4, 8, 16, 17, 32, 33):
        xs = [x0]
        for _ in range(M - 1):
            cx = torch.randn(1, HIDDEN_IN, generator=companion_rng, device="cpu").to(DEVICE, torch.bfloat16)
            xs.append(cx)
        x = torch.cat(xs, 0)
        xq, xs_scale = per_token_group_quant_fp8(x, BLOCK[1])
        out = w8a8_triton_block_scaled_mm(xq, weight, xs_scale, scale, list(BLOCK), output_dtype=x.dtype)
        out0 = out[0].clone()
        if baseline is None:
            baseline = out0
            print(f"M={M:3d}: baseline captured")
            continue
        eq = torch.equal(baseline, out0)
        diff = (baseline.float() - out0.float()).abs().max().item()
        print(f"M={M:3d}: row0 torch.equal={eq}  max|diff|={diff:.3e}")


if __name__ == "__main__":
    main()
