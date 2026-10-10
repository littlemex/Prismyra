"""The NVFP4 routed-expert path: quantisation math with no device needed, and the real thing on one.

`prismyra.kernels.nvfp4` was an experimental module that never shipped: this is the first time it has tests, and the
first time it is wired into `engine.py` and `kernels/qwen3_moe.py` rather than called by a one-off script. Two layers:

* the quantise/dequantise arithmetic is plain tensor math and needs no GPU -- a group's four-bit code must round-trip
  within the representable grid's own spacing, and the importance-weighted search (`quantize_search`) must not do
  worse than the plain scale it is offered as an improvement over;
* the engine actually answering through the borrowed kernel needs the real weights, a Blackwell card and vLLM's
  CUTLASS/FlashInfer NVFP4 build, so that half is `pytest.mark.gpu` like the rest of `test_gpu.py`, and skips rather
  than fails when any of those is absent.

Run with:

    pytest tests/test_nvfp4.py                                   # the CPU-only math
    PRISMYRA_EXPERTS=nvfp4 PRISMYRA_NVFP4_EXPERTS=... PRISMYRA_NVFP4_CALIB=... \\
        PRISMYRA_NVFP4_TEST_MODEL=<repo-or-dir> pytest -m gpu tests/test_nvfp4.py
"""

from __future__ import annotations

import os

import pytest
import torch

from prismyra.kernels.nvfp4 import _dequant_fp8, _e2m1, quantize, quantize_search

FP8 = torch.float8_e4m3fn


def _fp8_source(rows: int, cols: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A block-FP8 tensor like the one `prepare_layer` dequantises: 128x128 blocks, one scale per block."""
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(rows, cols, generator=g) * 0.3
    block_scale = full.view(rows // 128, 128, cols // 128, 128).abs().amax((1, 3)) / 448.0
    divisor = block_scale.view(rows // 128, 1, cols // 128, 1).clamp(min=1e-12)
    quantised = (full.view(rows // 128, 128, cols // 128, 128) / divisor).clamp(-448, 448).to(FP8)
    return quantised.view(rows, cols), block_scale.contiguous(), full


def test_dequant_fp8_restores_the_source_within_one_fp8_step():
    """`_dequant_fp8` inverts the block scaling `prepare_layer` is handed; the only loss between them is the
    original cast to FP8, which this checks against rather than assumes."""
    quantised, scale, full = _fp8_source(256, 256, seed=0)
    back = _dequant_fp8(quantised, scale)
    want = quantised.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(back, want)
    # And that is close to the pre-FP8 source -- not bit-identical, since FP8 already rounded it once.
    assert (back - full).abs().max() < 0.05 * full.abs().max()


def _decode(packed: torch.Tensor, scale: torch.Tensor, g: torch.Tensor, device) -> torch.Tensor:
    """The inverse of `quantize`/`quantize_search`'s packing, independent of either: a packed byte's low nibble is
    original column `2j` and its high nibble is `2j + 1` (`quantize` builds it as
    `code[:, 0::2] | (code[:, 1::2] << 4)`), so unpacking and interleaving them back -- rather than assuming they
    line up with the sixteen-wide scale groups, which they do not until the scale is itself repeated out to one
    value per original column -- is what a decoder has to do, not an extra step for a test to skip."""
    lo, hi = (packed & 0x0F).long(), ((packed >> 4) & 0x0F).long()
    table = _e2m1(device)

    def sign(c):
        return torch.where((c & 0x8).bool(), -1.0, 1.0)

    lo_val, hi_val = table[lo & 0x7] * sign(lo), table[hi & 0x7] * sign(hi)
    code_vals = torch.stack([lo_val, hi_val], dim=-1).flatten(-2)  # (rows, cols), columns back in original order
    eff = (scale.float() / g).repeat_interleave(16, dim=-1)  # one group's scale broadcast to its sixteen columns
    return code_vals * eff


def test_quantize_round_trips_within_the_e2m1_grid():
    """Sixteen-wide groups, packed two values per byte: decoding the packed code (`_decode`, the inverse this test
    does not get to assume is correct either) must land within the representable grid's own spacing of the source,
    the way `prepare_layer`'s own `quantize` -> served-weight round trip has to."""
    torch.manual_seed(1)
    w = (torch.rand(32, 64) - 0.5) * 4.0
    g = 448.0 * 6.0 / w.abs().amax()
    packed, scale = quantize(w, g)
    assert packed.dtype == torch.uint8
    assert packed.shape == (32, 32)  # two 4-bit codes per byte
    assert scale.shape == (32, 4)  # one scale per group of sixteen
    assert scale.dtype == FP8

    decoded = _decode(packed, scale, g, w.device)
    assert decoded.shape == w.shape
    # Per group, the largest step on the grid (4.0 to 6.0, the grid's own coarsest gap) scaled back into this
    # group's units -- no row's worst value may miss by more than that, or the code disagrees with its own scale.
    eff = (scale.float() / g).repeat_interleave(16, dim=-1)
    assert ((decoded - w).abs() <= 2.0 * eff + 1e-4).all()


def test_quantize_search_does_not_lose_to_the_scale_it_is_offered_as_an_improvement_over():
    """`quantize_search` tries a few fractions of the plain scale and keeps whichever minimises importance-weighted
    error -- one of the fractions it tries is 1.0, the plain scale's own choice, so it can only tie that baseline or
    beat it, never lose. A per-channel importance that is deliberately lopsided (one column dominates) is what should
    make the search actually move off 1.0."""
    torch.manual_seed(2)
    w = (torch.rand(16, 32) - 0.5) * 4.0
    g = 448.0 * 6.0 / w.abs().amax()
    importance = torch.ones(1, 32)
    importance[:, 0] = 1000.0  # one column that must be protected far more than the rest

    plain_packed, plain_scale = quantize(w, g)
    search_packed, search_scale = quantize_search(w, g, importance)

    def _error(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        decoded = _decode(packed, scale, g, w.device)
        return ((decoded - w) ** 2 * importance).sum()

    assert _error(search_packed, search_scale) <= _error(plain_packed, plain_scale) + 1e-6


pytestmark_gpu = pytest.mark.gpu


@pytestmark_gpu
def test_the_adapter_runs_the_routed_experts_in_nvfp4_and_still_answers():
    """The real thing: a checkpoint whose index leaves the routed experts out, loaded with PRISMYRA_EXPERTS=nvfp4,
    answers a boolean question and `kernels.apply`'s report says the routed experts were NVFP4 rather than silently
    falling back to a swap that found nothing (`routed_experts=0` would pass the FP8 count check just as emptily)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = os.environ.get("PRISMYRA_NVFP4_TEST_MODEL")
    if not model or os.environ.get("PRISMYRA_EXPERTS") != "nvfp4":
        pytest.skip(
            "set PRISMYRA_NVFP4_TEST_MODEL, PRISMYRA_EXPERTS=nvfp4, PRISMYRA_NVFP4_EXPERTS, PRISMYRA_NVFP4_CALIB"
        )
    try:
        import flashinfer  # noqa: F401
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import run_cutlass_moe_fp4  # noqa: F401
    except ImportError as e:
        pytest.skip(f"NVFP4 kernels are not importable on this build: {e}")

    from prismyra import Boolean, Prismyra

    engine = Prismyra(model, require_kernels=False)
    assert any("nvfp4" in note.lower() for note in engine.applied.notes), engine.applied.notes
    answer = engine.ask(
        "The sky is blue. Grass is green.",
        [Boolean(id="sky", prompt="Is the sky blue?")],
    )
    assert answer["sky"].value is True
