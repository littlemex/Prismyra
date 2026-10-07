"""inv5, round 7, chair's step (b): `diag_gated_rmsnorm_io_round7.py` found that `FusedGatedRMSNorm`'s own
INPUT (the GDN core's raw output) already differs between alone and group8 for document 0's branch row
(max|diff|=1.526e-05), ruling out the norm kernel and pointing at `chunk_gated_delta_rule`'s varlen branch call
itself. This calls that function directly (bypassing the whole model) with the real model's own shapes (16 key
heads / 32 value heads of 128, from config.json), a single document's 24 branch tokens held fixed, and a varying
number of appended companion tokens (0, 24, ..., 168, giving totals 24..192 -- the same range `diag_branch_attn_
io_round7.py` observed for `rows_in_call`), to see whether document 0's own first-column output changes.

Asserts the batch-invariant dispatcher is registered first (inv5, round 7, 13.5: the lesson from a false lead).

Usage: python3 diag_gdn_core_varlen_round7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import prismyra.engine as _engine_mod  # noqa: E402
from prismyra.kernels.qwen3_moe import _borrowed_delta  # noqa: E402

DEVICE = "cuda"
H, K, V = 32, 128, 128  # linear_num_value_heads, linear_key_head_dim, linear_value_head_dim (real config.json)
DOC_TOKENS = 24  # the real branch width this round measured (24 tokens for one rendered boolean question)


def main():
    _engine_mod._enable_batch_invariance()
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0

    kernel, why = _borrowed_delta("chunk", "chunk_gated_delta_rule")
    assert kernel is not None, why
    print(f"device={torch.cuda.get_device_name(0)}  kernel={kernel}", flush=True)

    gen = torch.Generator(device=DEVICE).manual_seed(42)

    def make(*shape, g=gen):
        return torch.randn(*shape, generator=g, device=DEVICE, dtype=torch.bfloat16)

    q_doc = make(1, DOC_TOKENS, H, K)
    k_doc = make(1, DOC_TOKENS, H, K)
    v_doc = make(1, DOC_TOKENS, H, V)
    g_doc = -make(1, DOC_TOKENS, H).abs().float()
    beta_doc = make(1, DOC_TOKENS, H).sigmoid().float()
    h0 = make(1, H, V, K).float()  # one initial recurrent state (identical "prefill result" in every scenario)

    def run(n_companions: int, companion_tokens: int):
        gen_c = torch.Generator(device=DEVICE).manual_seed(777)
        if n_companions == 0:
            q, k, v, g, beta = q_doc, k_doc, v_doc, g_doc, beta_doc
            cu = torch.tensor([0, DOC_TOKENS], device=DEVICE, dtype=torch.int32)
            h0_all = h0
        else:
            qs, ks, vs, gs, betas = [q_doc], [k_doc], [v_doc], [g_doc], [beta_doc]
            lengths = [DOC_TOKENS]
            for _ in range(n_companions):
                qs.append(torch.randn(1, companion_tokens, H, K, generator=gen_c, device=DEVICE, dtype=torch.bfloat16))
                ks.append(torch.randn(1, companion_tokens, H, K, generator=gen_c, device=DEVICE, dtype=torch.bfloat16))
                vs.append(torch.randn(1, companion_tokens, H, V, generator=gen_c, device=DEVICE, dtype=torch.bfloat16))
                gs.append(-torch.randn(1, companion_tokens, H, generator=gen_c, device=DEVICE).abs().float())
                betas.append(torch.randn(1, companion_tokens, H, generator=gen_c, device=DEVICE).sigmoid().float())
                lengths.append(companion_tokens)
            q = torch.cat(qs, dim=1)
            k = torch.cat(ks, dim=1)
            v = torch.cat(vs, dim=1)
            g = torch.cat(gs, dim=1)
            beta = torch.cat(betas, dim=1)
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            cu = torch.tensor(offsets, device=DEVICE, dtype=torch.int32)
            h0_all = torch.cat([h0] + [make(1, H, V, K, g=gen_c).float() for _ in range(n_companions)], dim=0)

        with torch.inference_mode():
            o, _ = kernel(
                q, k, v, g, beta, scale=K ** -0.5, initial_state=h0_all, output_final_state=False,
                cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
            )
        return o[:, :DOC_TOKENS].detach().to(torch.float32).clone()

    baseline = run(0, 0)
    print("baseline (document alone, no companion) computed.\n", flush=True)
    for n_companions, companion_tokens in ((1, 24), (3, 24), (7, 24), (1, 168), (7, 24)):
        out = run(n_companions, companion_tokens)
        eq = torch.equal(baseline, out)
        total = DOC_TOKENS + n_companions * companion_tokens
        tag = "IDENTICAL" if eq else f"DIFFERS max|diff|={(baseline - out).abs().max().item():.3e}"
        print(f"n_companions={n_companions} companion_tokens={companion_tokens} total={total:3d}: {tag}", flush=True)


if __name__ == "__main__":
    main()
