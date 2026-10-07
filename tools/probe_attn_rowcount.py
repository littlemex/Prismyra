"""inv2 round4: isolated probe of flash_attn_varlen_func's row-count sensitivity.

Hypothesis: after per-document padding (round 3) and router/tactic pinning (rounds 1-2), the remaining
0.008346 residual in test_open_batch_matches_ask_bit_for_bit_whatever_the_companions_total_length comes from
the branch-pass attention call in prismyra/kernels/qwen3_moe.py's FlashAttention.forward (the `key is None`
/ paged branch): `cu_seqlens_q` has length `rows+1` where `rows` is the pass's *total* padded row count
(target rows + companion rows), even though `max_seqlen_q` (width bucket) and `max_seqlen_k` (pool capacity
bucket) are both already constants. `flash_attn_varlen_func` takes `num_splits: int = 0` ("auto": vLLM's
flash-attn chooses a KV-split count from a heuristic keyed on total grid occupancy -- batch*heads*q-blocks
relative to SM count -- which changes the softmax running-reduction's order, hence last-bit rounding, with no
change to any one row's own q/k/v/length). Forcing num_splits=1 should make the row's output invariant to how
many *other* rows share the call.

Method: build one page pool directly (bypassing the whole model), with a fixed target row (row 0: fixed q/k/v
content, fixed seqused) and 0, 1, or 2 companion rows (random content, random seqused, appended after row 0 --
the real pass's document order). cu_seqlens_q/block_table/seqused all have length `rows`, built fresh each
time; max_seqlen_q and max_seqlen_k are literal constants matching the engine's own bucketing contract. Compare
row 0's own output slice across R in {1, 2, 3} with torch.equal, once at num_splits=0 (current code) and once
at num_splits=1 (candidate fix).
"""

import os

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.bfloat16

HEADS_Q = 16
HEADS_KV = 2  # GQA: Qwen3.6's gated-delta attention layers use a small kv-head count; exact count doesn't matter
HEAD_DIM = 128
BLOCK = 16
PAGES_TOTAL = 4096
PAGES_PER_ROW_TABLE = 256  # table width -> capacity = PAGES_PER_ROW_TABLE * BLOCK, constant regardless of rows
MAX_BATCH = 8
Q_LEN = 8  # the branch pass's bucketed answer width -- constant regardless of how many rows share the call

CAPACITY = PAGES_PER_ROW_TABLE * BLOCK


def make_pool():
    k = torch.randn(PAGES_TOTAL, BLOCK, HEADS_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(PAGES_TOTAL, BLOCK, HEADS_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return k, v


def make_table(rows: int, seqused_fixed_first: int, rng: torch.Generator):
    table = torch.zeros(rows, PAGES_PER_ROW_TABLE, dtype=torch.int32, device=DEVICE)
    seqused = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    used_pages = set()

    def alloc_pages(n_pages):
        chosen = []
        while len(chosen) < n_pages:
            p = int(torch.randint(0, PAGES_TOTAL, (1,), generator=rng).item())
            if p not in used_pages:
                used_pages.add(p)
                chosen.append(p)
        return chosen

    for r in range(rows):
        length = seqused_fixed_first if r == 0 else int(torch.randint(16, CAPACITY - 16, (1,), generator=rng).item())
        n_pages = (length + BLOCK - 1) // BLOCK
        pages = alloc_pages(n_pages)
        table[r, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=DEVICE)
        seqused[r] = length
    return table, seqused


def run(rows: int, pool_k, pool_v, q_row0, num_splits: int, cpu_rng_state):
    rng = torch.Generator(device="cpu")
    rng.set_state(cpu_rng_state)
    table, seqused = make_table(rows, seqused_fixed_first=100, rng=rng)

    q_other = torch.randn(rows - 1, Q_LEN, HEADS_Q, HEAD_DIM, device=DEVICE, dtype=DTYPE) if rows > 1 else None
    q = q_row0.clone() if q_other is None else torch.cat([q_row0.clone(), q_other.reshape(-1, HEADS_Q, HEAD_DIM)], 0)
    cu_q = torch.arange(0, rows * Q_LEN + 1, Q_LEN, device=DEVICE, dtype=torch.int32)

    out = flash_attn_varlen_func(
        q,
        pool_k,
        pool_v,
        cu_seqlens_q=cu_q,
        max_seqlen_q=Q_LEN,
        max_seqlen_k=CAPACITY,
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        block_table=table,
        seqused_k=seqused,
        num_splits=num_splits,
    )
    out = out[0] if isinstance(out, tuple) else out
    return out[:Q_LEN].clone()


def main():
    pool_k, pool_v = make_pool()
    q_row0 = torch.randn(Q_LEN, HEADS_Q, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    # Same companion content/lengths offered to every R so the only thing that changes across R is *how many*
    # rows (and which ones) are present -- row 0's own q/k/v/table/seqused never change.
    base_state = torch.Generator(device="cpu").manual_seed(1234).get_state()

    for num_splits, label in ((0, "auto (current code)"), (1, "forced num_splits=1 (candidate fix)")):
        print(f"\n=== num_splits={num_splits} [{label}] ===")
        baseline = None
        for rows in (1, 2, 3):
            out0 = run(rows, pool_k, pool_v, q_row0, num_splits, base_state.clone())
            if baseline is None:
                baseline = out0
                print(f"  rows={rows}: baseline captured")
                continue
            eq = torch.equal(baseline, out0)
            maxdiff = (baseline.float() - out0.float()).abs().max().item()
            print(f"  rows={rows}: torch.equal(row0, baseline@rows=1)={eq}  max|diff|={maxdiff:.3e}")


if __name__ == "__main__":
    main()
