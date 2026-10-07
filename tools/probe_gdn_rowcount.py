"""inv2 round4: isolated probe of chunk_gated_delta_rule's batch-size sensitivity (branch pass, no cu_seqlens).

The branch pass (`prismyra/engine.py::_branch_across`) does NOT open `varlen.reading(...)` around its
`self.backbone(...)` call -- confirmed by reading the code: there is no `with varlen.reading(...)` between
building `ids`/`positions` and calling `run()`. So during the branch pass, `varlen.current()` is None and the
GDN wrapper (`prismyra/kernels/qwen3_moe.py::_delta_wrapper`) does not pass `cu_seqlens` -- the kernel takes
its plain batched shape `(B, T, H, K)` with B = the pass's padded row count, which is exactly what varies
between "target answered alone" (B=2) and "target answered with a companion" (B=3).

Hypothesis: `vllm.third_party.flash_linear_attention.ops.cumsum.chunk_local_cumsum_vector_kernel` (and its
sibling kernels inside `chunk_gated_delta_rule_fwd`) are wrapped in `triton.autotune(key=["B", "H", "S", "BT",
...])` -- B is literally part of the autotune cache key, so a different total batch size can select a
different kernel config (different `BS`/`num_warps`), which changes the floating-point reduction order inside
one row's own cumsum/matmul even though that row's own q/k/v/g/beta never changed. Each config choice is
deterministic once cached per-B within a process (consistent with inv's round-2 restart-hash check, which only
ever tested one fixed B), so the same B always reproduces, but two different B's can legitimately diverge.

Method: fix row 0 (q/k/v/g/beta, random but held constant across every B), call `chunk_gated_delta_rule` with
B in {1, 2, 3} (row 0 plus 0/1/2 companion rows with their own random content), no cu_seqlens (matching the
real branch pass's code path exactly), and compare row 0's own `o`/`final_state` slice with torch.equal.
"""

import torch

from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16

T = 8  # branch pass width (bucketed token count per row) -- small and realistic
HEADS = 32  # linear_num_value_heads, per prismyra/kernels/qwen3_moe.py::_delta_inputs
KEY_DIM = 128
VALUE_DIM = 128


def make_row(rng: torch.Generator):
    def r(*shape):
        return torch.randn(*shape, generator=rng, device="cpu").to(DEVICE, DTYPE)

    q = r(1, T, HEADS, KEY_DIM)
    k = r(1, T, HEADS, KEY_DIM)
    v = r(1, T, HEADS, VALUE_DIM)
    g = -r(1, T, HEADS).abs().float()
    beta = r(1, T, HEADS).sigmoid().float()
    return q, k, v, g, beta


def main():
    row0_rng = torch.Generator().manual_seed(42)
    q0, k0, v0, g0, beta0 = make_row(row0_rng)

    companion_rng = torch.Generator().manual_seed(777)
    baseline_o = baseline_state = None
    for B in (1, 2, 3):
        qs, ks, vs, gs, betas = [q0], [k0], [v0], [g0], [beta0]
        for _ in range(B - 1):
            cq, ck, cv, cg, cbeta = make_row(companion_rng)
            qs.append(cq); ks.append(ck); vs.append(cv); gs.append(cg); betas.append(cbeta)
        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)
        v = torch.cat(vs, 0)
        g = torch.cat(gs, 0)
        beta = torch.cat(betas, 0)

        o, final_state = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o0 = o[0].clone()
        state0 = final_state[0].clone()
        if baseline_o is None:
            baseline_o, baseline_state = o0, state0
            print(f"B={B}: baseline captured, o.shape={tuple(o0.shape)} state.shape={tuple(state0.shape)}")
            continue
        eq_o = torch.equal(baseline_o, o0)
        eq_s = torch.equal(baseline_state, state0)
        diff_o = (baseline_o.float() - o0.float()).abs().max().item()
        diff_s = (baseline_state.float() - state0.float()).abs().max().item()
        print(f"B={B}: o torch.equal={eq_o} max|diff|={diff_o:.3e}   final_state torch.equal={eq_s} max|diff|={diff_s:.3e}")


if __name__ == "__main__":
    main()
