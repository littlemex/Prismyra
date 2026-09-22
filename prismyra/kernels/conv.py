"""A causal depthwise convolution in Triton, reading a token-major tensor and respecting sequence boundaries.

Written because everything easier was measured and lost: the framework's path falls back to a general two-dimensional
convolution, `torch.nn.functional.conv1d` is slower still, and the borrowed variable-length kernel needs the paged
state its own model code maintains. Triton compiles at run time, so no toolchain is required.

Token-major on purpose: the layer holds its tensor that way before transposing for the old kernel, and since that
transpose is a view, consuming it directly removes the copy the old kernel's contiguity demand forced. Measured
figures are in docs/KERNELS.md.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


if triton is not None:

    @triton.jit
    def _kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        starts_ptr,
        n_tokens,
        n_channels,
        width: tl.constexpr,
        sx_t,
        sx_c,
        sw_c,
        sw_j,
        so_t,
        so_c,
        HAS_STARTS: tl.constexpr,
        SILU: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """One program per (token block, channel block); each output reads `width` earlier tokens of its own channel.

        The accumulator is float32 whatever the input dtype, so the sum rounds once at the end rather than per term.
        """
        offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_t = offs_t < n_tokens
        mask_c = offs_c < n_channels

        if HAS_STARTS:
            seq_start = tl.load(starts_ptr + offs_t, mask=mask_t, other=0)
        else:
            seq_start = tl.zeros([BLOCK_T], dtype=tl.int32)

        acc = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
        for j in tl.static_range(width):
            src_t = offs_t - (width - 1 - j)
            ok = mask_t & (src_t >= seq_start)
            x = tl.load(
                x_ptr + src_t[:, None] * sx_t + offs_c[None, :] * sx_c, mask=ok[:, None] & mask_c[None, :], other=0.0
            ).to(tl.float32)
            wj = tl.load(w_ptr + offs_c * sw_c + j * sw_j, mask=mask_c, other=0.0).to(tl.float32)
            acc += x * wj[None, :]

        if SILU:
            acc = acc * tl.sigmoid(acc)
        tl.store(
            out_ptr + offs_t[:, None] * so_t + offs_c[None, :] * so_c,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_t[:, None] & mask_c[None, :],
        )


def available() -> bool:
    return triton is not None


def causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    seq_starts: torch.Tensor | None = None,
    activation: str | None = "silu",
    block_t: int = 64,
    block_c: int = 128,
) -> torch.Tensor:
    """`x` is (tokens, channels), `weight` is (channels, width). Returns (tokens, channels).

    `seq_starts` gives each token the position its own sequence began at, so packed sequences do not read across their
    boundaries.
    """
    if triton is None:
        raise RuntimeError("triton is not available")
    if x.dim() != 2:
        raise ValueError(f"expected (tokens, channels), got {tuple(x.shape)}")
    n_tokens, n_channels = x.shape
    if weight.shape[0] != n_channels:
        raise ValueError(f"weight has {weight.shape[0]} channels and the input has {n_channels}")

    out = torch.empty_like(x)
    grid = (triton.cdiv(n_tokens, block_t), triton.cdiv(n_channels, block_c))
    _kernel[grid](
        x,
        weight,
        out,
        seq_starts if seq_starts is not None else x,
        n_tokens,
        n_channels,
        weight.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_STARTS=seq_starts is not None,
        SILU=activation in ("silu", "swish", True),
        BLOCK_T=block_t,
        BLOCK_C=block_c,
    )
    return out


def starts_from_boundaries(cu_seqlens: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Expand cumulative sequence boundaries into one start index per token.

    Per token rather than per sequence so a program covering a block of tokens can mask each one without searching.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.repeat_interleave(cu_seqlens[:-1], lengths)[:n_tokens].to(torch.int32)
