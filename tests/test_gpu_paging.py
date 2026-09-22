"""Whether the attention kernel tolerates several rows reading the same pages, which decides a design.

An open context currently holds `group` physical copies of its keys and values -- 2.17 GiB at 3,040 tokens, measured,
and the reason only three contexts fit on a 48 GiB card beside the weights. The copies are not a modelling necessity:
this model has two key-value heads, so one copy is about a thirtieth of that. What stands in the way is whether the
kernel will read one set of pages from many rows at once.

This is a kill criterion rather than a feature test. If the aliased read disagrees with the materialised one, the
paged design is not available and the copies stay. Marked `gpu` because it needs the kernel; it does not need the
model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

#: Tokens per page. vLLM's own kernels are built around sizes like this, and the shared region has to end on one of
#: these boundaries -- which is why the last partial page of a context has to be copied per row rather than shared.
BLOCK = 16


def test_many_rows_may_read_one_set_of_pages() -> None:
    """Rows whose block tables point at the same prefix pages must answer as rows with their own copies.

    Built as the real thing would be: one shared prefix written once, each row's own tail written into private pages,
    and a block table per row that names the shared pages first and the private ones after. The comparison is against
    the same keys and values materialised per row, which is what the package does today.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    flash = pytest.importorskip("vllm.vllm_flash_attn")

    rows, heads, dim = 8, 2, 64
    prefix_tokens, tail_tokens = 64, 16  # both multiples of the page size, so nothing has to be duplicated
    prefix_pages, tail_pages = prefix_tokens // BLOCK, tail_tokens // BLOCK
    torch.manual_seed(0)

    pool_pages = prefix_pages + rows * tail_pages
    key_pool = torch.randn(pool_pages, BLOCK, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    value_pool = torch.randn_like(key_pool)

    # Every row names the same prefix pages, then its own tail pages. That aliasing is the whole point.
    table = torch.tensor(
        [
            list(range(prefix_pages)) + [prefix_pages + row * tail_pages + i for i in range(tail_pages)]
            for row in range(rows)
        ],
        device="cuda",
        dtype=torch.int32,
    )

    queries = torch.randn(rows * tail_tokens, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    cu_q = torch.arange(0, rows * tail_tokens + 1, tail_tokens, device="cuda", dtype=torch.int32)
    seqused = torch.full((rows,), prefix_tokens + tail_tokens, device="cuda", dtype=torch.int32)

    paged = flash.flash_attn_varlen_func(
        queries,
        key_pool,
        value_pool,
        cu_seqlens_q=cu_q,
        max_seqlen_q=tail_tokens,
        max_seqlen_k=prefix_tokens + tail_tokens,
        softmax_scale=dim**-0.5,
        causal=True,
        block_table=table,
        seqused_k=seqused,
    )
    paged = paged[0] if isinstance(paged, tuple) else paged

    # The same attention with every row holding its own copy of the prefix, which is today's layout.
    keys, values = _materialise(key_pool, value_pool, table, prefix_tokens + tail_tokens)
    cu_k = torch.arange(
        0, rows * (prefix_tokens + tail_tokens) + 1, prefix_tokens + tail_tokens, device="cuda", dtype=torch.int32
    )
    copied = flash.flash_attn_varlen_func(
        queries,
        keys,
        values,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=tail_tokens,
        max_seqlen_k=prefix_tokens + tail_tokens,
        softmax_scale=dim**-0.5,
        causal=True,
    )
    copied = copied[0] if isinstance(copied, tuple) else copied

    moved = (paged.float() - copied.float()).abs().max().item()
    scale = copied.float().abs().max().item()
    # Relative to the output, and not bit-identity: a paged read reduces in a different order, so demanding equality
    # would reject a correct kernel. Two steps of bfloat16 is the same bar the kernel swaps are held to.
    assert moved / max(scale, 1e-6) < 2 * 2.0**-8, f"aliased pages disagreed by {moved:.3e} against {scale:.3e}"


def _materialise(key_pool, value_pool, table, length: int):
    """Gather each row's pages into a contiguous sequence, which is what the aliased read has to match."""
    rows = table.shape[0]
    keys = torch.empty(
        rows * length, key_pool.shape[2], key_pool.shape[3], device=key_pool.device, dtype=key_pool.dtype
    )
    values = torch.empty_like(keys)
    for row in range(rows):
        pages = table[row].tolist()
        gathered_k = torch.cat([key_pool[page] for page in pages])[:length]
        gathered_v = torch.cat([value_pool[page] for page in pages])[:length]
        keys[row * length : (row + 1) * length] = gathered_k
        values[row * length : (row + 1) * length] = gathered_v
    return keys, values


def test_a_partial_last_page_cannot_be_shared() -> None:
    """The arithmetic behind the one copy the design still has to make.

    Sharing ends on a page boundary, so a context whose length is not a multiple of the page size leaves a remainder
    that each branch has to own -- it is the page each branch then writes its own tokens into. This states the sizes
    rather than asserting the kernel, because getting it wrong is an out-of-bounds write rather than a wrong answer.
    """
    for context in (3040, 5000, 20000):
        shared = (context // BLOCK) * BLOCK
        remainder = context - shared
        assert shared % BLOCK == 0
        assert 0 <= remainder < BLOCK
        # What each branch must own privately: the remainder plus its own suffix, rounded up to whole pages.
        private_tokens = remainder + 512
        assert np.ceil(private_tokens / BLOCK) * BLOCK >= private_tokens
