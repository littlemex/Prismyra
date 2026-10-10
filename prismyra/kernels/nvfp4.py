"""Routed experts in NVFP4 (4-bit weights and 4-bit activations), on vLLM's CUTLASS grouped kernel. Experimental.

Selected with `PRISMYRA_EXPERTS=nvfp4` and a calibration file in `PRISMYRA_NVFP4_CALIB` (per layer: the largest input
to the experts and the largest input to their down projection, measured on calibration rows). The recipe is the one
measured through vLLM as R1: each expert's FP8 blocks are restored to the values served today, then quantised in groups
of sixteen with an E4M3 scale and one FP32 scale per expert matrix (gate and up share it, because the kernel multiplies
them as one matrix). Activation scales are 448 * 6 / (1.25 * calibrated maximum), one per layer.

The weights are converted one layer at a time from the host, so the FP8 experts never have to fit on the device beside
the result -- the reason a 36-layer checkpoint fits a 32 GiB card at all.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from .autotune import arch_of

FP8 = torch.float8_e4m3fn
HEADROOM = 1.25

#: Per-card tables of already-measured NVFP4 GEMM tactics, shipped with the package the same way
#: `pinned/fp8_block_configs/` ships the dense-FP8 matmul tiling (`kernels/fp8_tuning.py`): one JSON file per GPU
#: generation, in the exact format `flashinfer.autotuner.autotune(cache=...)` reads and writes. See
#: `autotune_tactics()` for how a process without a matching table still answers deterministically within itself.
PINNED_NVFP4_DIR = Path(__file__).parent / "pinned" / "nvfp4_tactics"


def _e2m1(device):
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)


def quantize(w: torch.Tensor, global_scale: torch.Tensor):
    """(rows, cols) float32 and one global scale -> packed uint8 (rows, cols/2) and E4M3 scales (rows, cols/16)."""
    rows, cols = w.shape
    blocks = w.view(rows, cols // 16, 16)
    scale = (blocks.abs().amax(-1) / 6.0 * global_scale).clamp(max=448.0).to(FP8)
    eff = (scale.float() / global_scale).unsqueeze(-1)
    x = torch.where(eff > 0, blocks / eff, torch.zeros_like(blocks)).clamp(-6, 6)
    table = _e2m1(w.device)
    code = (x.abs().unsqueeze(-1) - table).abs().argmin(-1) | ((x < 0).to(torch.long) << 3)
    code = code.to(torch.uint8).view(rows, cols)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), scale.contiguous()


SEARCH = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


def quantize_search(w: torch.Tensor, global_scale: torch.Tensor, importance: torch.Tensor | None):
    """Like `quantize`, but each 16-group's scale is chosen from a few fractions of its maximum to minimise the
    importance-weighted squared error (AWQ/GPTQ-style protection of the input channels that carry the most signal).
    `importance` is per input column (e.g. the mean squared input); None means plain squared error."""
    rows, cols = w.shape
    blocks = w.view(rows, cols // 16, 16)
    imp = (
        importance.float().view(1, cols // 16, 16)
        if importance is not None
        else torch.ones(1, cols // 16, 16, device=w.device)
    )
    amax = blocks.abs().amax(-1)
    table = _e2m1(w.device)
    best_err = best_scale = None
    for c in SEARCH:
        scale = (amax / 6.0 * global_scale * c).clamp(max=448.0).to(FP8)
        eff = (scale.float() / global_scale).unsqueeze(-1)
        x = torch.where(eff > 0, blocks / eff, torch.zeros_like(blocks)).clamp(-6, 6)
        idx = (x.abs().unsqueeze(-1) - table).abs().argmin(-1)
        back = table[idx] * torch.where(x < 0, -1.0, 1.0) * eff
        err = (imp * (back - blocks) ** 2).sum(-1)
        if best_err is None:
            best_err, best_scale = err, scale.float()
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale.float(), best_scale)
    scale = best_scale.to(FP8)
    eff = (scale.float() / global_scale).unsqueeze(-1)
    x = torch.where(eff > 0, blocks / eff, torch.zeros_like(blocks)).clamp(-6, 6)
    code = (x.abs().unsqueeze(-1) - table).abs().argmin(-1) | ((x < 0).to(torch.long) << 3)
    code = code.to(torch.uint8).view(rows, cols)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), scale.contiguous()


def _importance() -> dict | None:
    path = os.environ.get("PRISMYRA_NVFP4_IMPORTANCE")
    return torch.load(path) if path else None


def _dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    o, i = w.shape
    return (w.float().view(o // 128, 128, i // 128, 128) * s.float().view(o // 128, 1, i // 128, 1)).view(o, i)


def calibration() -> dict:
    path = os.environ.get("PRISMYRA_NVFP4_CALIB")
    if not path:
        raise RuntimeError("PRISMYRA_EXPERTS=nvfp4 needs PRISMYRA_NVFP4_CALIB, the per-layer activation maxima")
    return json.load(open(path))["moe"]


WORKSPACE_CAP = int(float(os.environ.get("PRISMYRA_NVFP4_WORKSPACE_GIB", "2")) * 2**30)


class FusedExpertsFp4(nn.Module):
    """The routed path in NVFP4; router and shared expert untouched, as in `FusedExperts`."""

    def __init__(self, block: nn.Module, top_k: int, prepared: dict, act: dict, device):
        super().__init__()
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

        self.gate = block.gate
        self.shared_expert = block.shared_expert
        self.shared_expert_gate = block.shared_expert_gate
        self.top_k = top_k
        t = {k: v.to(device) for k, v in prepared.items()}
        e = t["w1"].shape[0]
        a13 = torch.full((e,), 448.0 * 6.0 / (HEADROOM * act["w13_in_amax"]), dtype=torch.float32, device=device)
        a2 = torch.full((e,), 448.0 * 6.0 / (HEADROOM * max(act["w2_in_amax"])), dtype=torch.float32, device=device)
        self.backend = os.environ.get("PRISMYRA_NVFP4_BACKEND", "flashinfer")
        # FlashInfer's default folds the top-k reduction into the second GEMM's epilogue with atomic adds, whose order
        # changes run to run: the same weights then answer differently on a re-run (measured: 7 of 114 Kev answers
        # flipped). The unfused reduction is deterministic. PRISMYRA_NVFP4_FUSED_FINALIZE=1 restores the fused one.
        self.fused_finalize = os.environ.get("PRISMYRA_NVFP4_FUSED_FINALIZE", "0") == "1"
        w1, s1 = t["w1"], t["s1"]
        if self.backend == "flashinfer":
            # FlashInfer's fused kernel wants the up rows before the gate rows.
            half = w1.shape[1] // 2
            w1 = torch.cat([w1[:, half:], w1[:, :half]], 1).contiguous()
            s1 = torch.cat([s1[:, half:], s1[:, :half]], 1).contiguous()
        self.w1, self.w2 = w1, t["w2"]
        self.w1_scale, self.w2_scale = swizzle_blockscale(s1), swizzle_blockscale(t["s2"])
        self.a1_gscale, self.a2_gscale = a13, a2
        self.g1_alphas = (1.0 / (t["g1"].float() * a13)).contiguous()
        self.g2_alphas = (1.0 / (t["g2"].float() * a2)).contiguous()
        self.n, self.k, self.e = t["w2"].shape[2] * 2, t["w1"].shape[2] * 2, e
        if hasattr(block, "experts"):
            del block.experts


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_topk

        from .qwen3_moe import _ROUTER_LINEAR

        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        # This used to be `logits, _, _ = self.gate(x)` -- calling the
        # router module's own `forward` whole, which is `F.linear(hidden_states, self.weight)` followed by a
        # softmax/top-k this function immediately recomputes with `fused_topk` and discards (the identical "called
        # it twice, threw one away" pattern `kernels.qwen3_moe.FusedExperts._route`, the FP8 path's router, already
        # comments on; measured: 36 calls, ~4.13ms on a 36-layer read, one per layer). An earlier
        # fix (a `_route` helper calling plain `torch.nn.functional.linear`) only removed the duplicate call; this
        # path separately never went through the row-count-invariant kernel the FP8 path's
        # `_route` already uses (`audit_sm120.py` measured a 5.9% non-bit-exact
        # residual the Blackwell cuBLAS-workspace fix alone did not close). The current version is a strict
        # superset of that earlier fix (same duplicate-call removal, plus row-count invariance), so it is the
        # one kept here; the standalone `_route` method the earlier fix added is dropped as dead code.
        linear = _ROUTER_LINEAR or torch.nn.functional.linear
        logits = linear(x, self.gate.weight)
        weights, ids = fused_topk(x, logits, self.top_k, renormalize=True)[:2]
        out = self.routed(x, weights, ids)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return (out + shared).reshape(shape)

    _ws: ClassVar[dict] = {}
    _ws_size: ClassVar[dict] = {}

    def _workspace(self, m: int, x_dtype) -> torch.Tensor:
        """One scratch buffer per calling thread for every MoE layer (they share shapes), grown to the largest
        token count that thread has seen, instead of the kernel allocating and freeing its scratch on each call.

        Keyed by `threading.get_ident()` as well as device and `fused_finalize`, not just the
        latter two. `Batcher(lanes=N)` runs `N` fully independent worker threads, each driving its own CUDA
        stream, and every one of the 36 MoE layers shared this one class-level buffer across all of them -- two
        lanes could genuinely call this at the same instant (`test_lanes_two_decisions_under_a_burst_do_not_
        move`'s burst does exactly that), and the check-then-create below is not atomic across threads: both
        could see a cache miss, both allocate their own buffer, and whichever one loses the race to the shared
        dict slot has its own buffer dropped out from under a CUDA kernel that is still writing into it on its
        own stream -- a used-after-dropped scratch buffer, which reads as the illegal-memory-access crash and
        the NaN probabilities this fix addresses (FlashInfer's own autotuner warning
        for an unseen shape appears immediately before the NaN, which is the same first-use race one layer up).
        Even with the race on the dict closed, one physical buffer still cannot be *used* by two lanes at once
        -- the CUTLASS kernel treats it as private scratch for the one call it was handed to -- so the fix is
        to give each lane its own, the same trade `fork._owned`'s own `lane` parameter already makes (one more
        full buffer per additional lane, not a smaller one shared unsafely). A thread id rather than an
        explicit `lane` parameter because nothing from `Batcher`'s own `lane` plumbing (`fork.py`, `engine.py`)
        reaches this deep into the model's forward pass; each lane is a dedicated, long-lived worker thread, so
        its identity is already a correct and available proxy for "which lane".
        """
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

        key = (self.w1.device, self.fused_finalize, threading.get_ident())
        size = FusedExpertsFp4._ws_size.get((key, m))
        if size is None:
            size = FusedExpertsFp4._ws_size[(key, m)] = cutlass_fused_moe_workspace_size(
                m, self.k, self.n, self.e, self.top_k, x_dtype=x_dtype, weight_dtype=torch.long,
                output_dtype=torch.bfloat16, use_fused_finalize=self.fused_finalize, device=self.w1.device)
        if size > WORKSPACE_CAP:
            return None                     # rare very large calls (the autotune sweep) keep the kernel's own scratch
        buf = FusedExpertsFp4._ws.get(key)
        if buf is None or buf.numel() < size:
            FusedExpertsFp4._ws[key] = None
            # Zero-initialized at allocation: this buffer is reused across calls (keyed by device, fusion mode,
            # and thread), so an uninitialized first allocation would leak this process's own CUDA allocator
            # history -- not the kernel's own output -- into the answer. Paid at each key's first allocation
            # (normally once per process), not on every request.
            buf = FusedExpertsFp4._ws[key] = torch.zeros(size, dtype=torch.uint8, device=self.w1.device)
        return buf

    def routed(self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import run_cutlass_moe_fp4

        m = x.shape[0]
        out = torch.empty_like(x)
        if self.backend == "flashinfer":
            # One fused kernel: the permutation, both grouped GEMMs, the activation and the weighted sum are inside it,
            # where the plain CUTLASS route runs them as separate kernels (row shuffles, a quantise, a reduction).
            from flashinfer.fused_moe.core import ActivationType
            from vllm import _custom_ops as ops
            from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

            xq, xsf = ops.scaled_fp4_quant(x.contiguous(), self.a1_gscale[:1])
            ws = self._workspace(m, xq.dtype) if os.environ.get("PRISMYRA_NVFP4_WORKSPACE", "1") == "1" else None
            flashinfer_cutlass_fused_moe(
                input=xq, token_selected_experts=ids.to(torch.int), token_final_scales=weights,
                fc1_expert_weights=self.w1.view(torch.long), fc2_expert_weights=self.w2.view(torch.long),
                output=out, output_dtype=x.dtype,
                quant_scales=[self.a1_gscale, self.w1_scale.view(torch.int32), self.g1_alphas,
                              self.a2_gscale, self.w2_scale.view(torch.int32), self.g2_alphas],
                input_sf=xsf, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, activation_type=ActivationType.Swiglu,
                use_fused_finalize=self.fused_finalize, workspace_buffer=ws,
            )
            return out
        ws13 = torch.empty(m * self.top_k * max(2 * self.n, self.k), dtype=x.dtype, device=x.device)
        ws2 = torch.empty(m * self.top_k * self.n, dtype=x.dtype, device=x.device)
        run_cutlass_moe_fp4(
            output=out, a=x.contiguous(), a1_gscale=self.a1_gscale, w1_fp4=self.w1, w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas, a2_gscale=self.a2_gscale, w2_fp4=self.w2, w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas, topk_weights=weights, topk_ids=ids.to(torch.int32), activation=MoEActivation.SILU,
            workspace13=ws13, workspace2=ws2, m=m, n=self.n, k=self.k, e=self.e, device=x.device,
        )
        return out


def prepare_layer(w13, s13, w2, s2, device, imp_in=None, imp_mid=None, tok=None) -> dict:
    """FP8 experts of one layer -> NVFP4 tensors (block scales unswizzled), one global scale per expert matrix."""
    out = {"w1": [], "s1": [], "g1": [], "w2": [], "s2": [], "g2": []}
    for i in range(w13.shape[0]):
        for w, s, k in ((w13, s13, "1"), (w2, s2, "2")):
            d = _dequant_fp8(w[i].to(device), s[i].to(device))
            g = 448.0 * 6.0 / d.abs().amax().clamp_min(1e-12)
            if imp_in is not None:
                imp = (imp_in if k == "1" else imp_mid)[i].to(device)
                imp = imp if (tok is None or tok[i] > 0) else None
                p, sc = quantize_search(d, g, imp)
            else:
                p, sc = quantize(d, g)
            out["w" + k].append(p)
            out["s" + k].append(sc)
            out["g" + k].append(g.float())
    return {k: torch.stack(v).cpu() for k, v in out.items()}


@torch.no_grad()
def convert(model: nn.Module, top_k: int, device) -> int:
    """Replace every sparse MoE block with the NVFP4 path, from the prepared file in `PRISMYRA_NVFP4_EXPERTS`."""
    from safetensors.torch import load_file

    calib = calibration()
    prepared = load_file(os.environ["PRISMYRA_NVFP4_EXPERTS"])
    names = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Qwen3_5MoeSparseMoeBlock"]
    for name, block in names:
        key = name[name.index("language_model.") :] if "language_model." in name else name
        layer = key.split(".")[2]
        tensors = {k: prepared[f"{layer}.{k}"] for k in ("w1", "s1", "g1", "w2", "s2", "g2")}
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], FusedExpertsFp4(block, top_k, tensors, calib[key], device))
    torch.cuda.empty_cache()
    if names and os.environ.get("PRISMYRA_NVFP4_AUTOTUNE", "1") == "1":
        first = model.get_submodule(names[0][0])
        if first.backend == "flashinfer":
            autotune_tactics(first)
    return len(names)


#: Kept alive for the engine's whole lifetime once `autotune_tactics` has run, never exited: FlashInfer's
#: `autotune()` context manager is what makes `map_to_tuning_buckets` round a runtime size to a profiled bucket
#: instead of falling back to an untuned tactic, and it only has that effect while the context is entered. The
#: profiling loop below closes its own `with` block before this module's caller ever serves a real request, so
#: without this second, permanently-open context, the profiling was real but every later call ran outside it --
#: measured as `falling back to runner=MoERunner tactic=-1` in FlashInfer's own log for every token count that
#: was not one of the exact profiled buckets, i.e. nearly every real request. One per process, since the tactics
#: are themselves process-global (see the first paragraph below).
_INFERENCE_AUTOTUNE_CTX = None

#: What `autotune_tactics()` used to pick this process's tactic, for `engine.stats()["nvfp4_tactics"]` and for
#: `tests/test_gpu_cold_start.py`: "env" (`PRISMYRA_NVFP4_TACTICS`), "bundled" (a `PINNED_NVFP4_DIR` table that
#: matched this card and this FlashInfer/CUDA/cuDNN build), or "profiled" (no table applied -- this process timed
#: the candidates itself, same as every process did before this existed; `None` before the first MoE layer
#: converts). `pinned` is `source in ("env", "bundled")`: whether *this* process's tactic is guaranteed to be the
#: same one another process picked, not merely consistent with itself.
_TACTIC_SOURCE: str | None = None


def tactics_status() -> dict:
    """`{"source": ..., "pinned": ..., "table": ...}`, the public read of `_TACTIC_SOURCE` for `engine.stats()`."""
    return {
        "source": _TACTIC_SOURCE,
        "pinned": None if _TACTIC_SOURCE is None else _TACTIC_SOURCE in ("env", "bundled"),
    }


def autotune_tactics(layer: FusedExpertsFp4, max_tokens: int = 16384) -> None:
    """Pick the fused MoE kernel's tactic (tile shape and schedule), the same way in every process.

    Without this FlashInfer runs one default tactic for every size. Every MoE layer has the same shapes, so one layer's
    choice serves all of them; FlashInfer keeps the choice for the rest of the process. Inputs are random: the timing
    depends on the shapes and the routing spread, not on the values.

    One bucket, not the powers-of-two ladder this used to profile. The previous
    version picked a *different* tactic per power-of-two bucket (`round_up=False`, so a size between two buckets ran
    the lower one's choice) -- which is exactly the row-count-dependent-algorithm shape of bug this project calls the
    companion effect everywhere else: two passes carrying the identical real row at a different *total* M could cross
    a bucket boundary (e.g. 32 companion questions -> 33) and get a different tactic, hence a different GEMM
    reduction order, hence a non-bit-identical answer for that unchanged row. This was never caught because no
    device test exercises an NVFP4 pass across a bucket boundary before `audit_sm120.py`.
    One bucket at `max_tokens` with `round_up=True` makes every real M (small questions-only pass or a full
    `open_batch`) map to the *same* profiled tactic, by construction -- determinism first, matching this project's
    standing rule; the speed cost of using the large-M tactic at small M is measured directly rather than assumed.

    That closed the *within-process* gap. It did not close the one between processes: the choice at this one
    bucket is still made by timing, so two processes can pick differently if FlashInfer finds two tactics
    near-equal here -- measured directly: the same release, as two separate processes
    with no cache file, answered the same 1/16/64-question request bit-for-bit differently in 81 of 81 entries.
    `PRISMYRA_NVFP4_TACTICS` names a JSON file the caller controls: loaded if it exists, written if not, so every
    process that shares it runs the same tactic -- but a caller who never sets it got the old, timing-picked
    behaviour by default. This project ships one more table, `PINNED_NVFP4_DIR`, exactly the way
    `kernels/fp8_tuning.py` ships the dense-FP8 matmul tiling: already-measured tactics for the card this
    checkpoint serves on, read automatically, with no setting required.

    Precedence: `PRISMYRA_NVFP4_TACTICS` first (a caller measuring its own tactics needs its file to
    win); otherwise `PINNED_NVFP4_DIR/sm_<arch>.json` if that file exists *and* FlashInfer accepts it
    (its own `_metadata` -- FlashInfer version, CUDA/cuBLAS/cuDNN versions, GPU name -- must match this process;
    a mismatch is a different environment than the one the table was measured on, not this one, so FlashInfer
    ignores it rather than silently handing out a tactic index that may not even exist in this build). Either way
    the table is loaded read-only (`AutoTuner.load_configs`, not passed as this call's own `cache=`), so a process
    that cannot use it never overwrites the package's copy with its own, environment-specific measurement -- the
    same failure mode `kernels/fp8_tuning.py`'s own docstring describes for a table that is not shipped read-only.
    When neither applies, this falls back to the old behaviour (profile fresh, pinned within this process only)
    and warns, the same way `kernels.autotune.pin()` warns when a Triton kernel has no table for this card.
    """
    from flashinfer.autotuner import AutoTuner, autotune

    global _INFERENCE_AUTOTUNE_CTX, _TACTIC_SOURCE  # noqa: PLW0603 - process-wide autotune state, by design
    buckets = (max_tokens,)
    env_cache = os.environ.get("PRISMYRA_NVFP4_TACTICS")
    cache = env_cache
    if env_cache:
        source = "env"
    else:
        source = "profiled"
        arch = arch_of(layer.w1.device)
        bundled = PINNED_NVFP4_DIR / f"{arch}.json" if arch else None
        if bundled and bundled.is_file():
            if AutoTuner.get().load_configs(str(bundled)):
                source = "bundled"
            else:
                warnings.warn(
                    f"the bundled NVFP4 tactic table {bundled} does not match this process's FlashInfer/CUDA/cuDNN "
                    "build or GPU; the tactic is timing-picked for this process and may differ from another "
                    "process's. engine.stats()['nvfp4_tactics'] reports this.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"no bundled NVFP4 tactic table for this card ({arch or 'unknown'}); the tactic is timing-picked "
                "for this process and may differ from another process's unless PRISMYRA_NVFP4_TACTICS is set. "
                "engine.stats()['nvfp4_tactics'] reports this.",
                stacklevel=2,
            )
        # Loaded above via `load_configs`, not handed to `autotune(cache=...)` below: that call only auto-saves
        # what it auto-loads, and this table must stay read-only (see the docstring's "Either way" paragraph).
        cache = None
    with torch.inference_mode(), autotune(True, cache=cache, tuning_buckets=buckets):
        m = max_tokens
        x = torch.randn(m, layer.k, device=layer.w1.device, dtype=torch.bfloat16)
        ids = torch.rand(m, layer.e, device=x.device).argsort(1)[:, : layer.top_k].int().contiguous()
        w = torch.full((m, layer.top_k), 1.0 / layer.top_k, device=x.device)
        layer.routed(x, w, ids)
    torch.cuda.synchronize()
    _TACTIC_SOURCE = source
    # Opened and never exited: every call to `routed()` for the rest of this process now runs inside it, with
    # tune_mode=False (look up the cached tactic rather than re-profile) and round_up=True so every real M -- below,
    # at, or (should it ever happen) above `max_tokens` -- maps to this one bucket's tactic, never a different one.
    ctx = autotune(False, cache=cache, tuning_buckets=buckets, round_up=True)
    ctx.__enter__()
    _INFERENCE_AUTOTUNE_CTX = ctx


class tiny_experts:
    """While loading, build the FP8 experts with one expert of one row, so the checkpoint loads without them (their
    keys are left out of the index) and the device never holds the FP8 experts."""

    def __enter__(self):
        import copy

        import transformers.integrations.finegrained_fp8 as fp8

        self._fp8, self._init = fp8, fp8.FP8Experts.__init__
        original = self._init

        def init(module, config, *args, **kwargs):
            small = copy.copy(config)
            for name in ("num_local_experts", "num_experts"):
                if hasattr(small, name):
                    setattr(small, name, 1)
            original(module, small, *args, **kwargs)

        fp8.FP8Experts.__init__ = init
        return self

    def __exit__(self, *exc):
        self._fp8.FP8Experts.__init__ = self._init


class Fp4Linear(nn.Module):
    """A dense FP8 projection re-quantised to NVFP4 (weights and activations), on vLLM's CUTLASS FP4 GEMM.
    Experimental: PRISMYRA_DENSE=nvfp4. Activation scale from the calibration file's `dense_in_amax`."""

    def __init__(self, fp8: nn.Module, in_amax: float, importance: torch.Tensor | None = None, search: bool = False):
        super().__init__()
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            pad_nvfp4_weight_for_cutlass,
            swizzle_blockscale,
        )

        w = _dequant_fp8(fp8.weight, fp8.scale)
        g = 448.0 * 6.0 / w.abs().amax().clamp_min(1e-12)
        if search:
            packed, sc = quantize_search(w, g, importance.to(w.device) if importance is not None else None)
        else:
            packed, sc = quantize(w, g)
        self.out_features, self.in_features = w.shape
        self.weight, self.pad = pad_nvfp4_weight_for_cutlass(packed)
        self.weight_scale = swizzle_blockscale(sc)
        ga = 448.0 * 6.0 / (HEADROOM * in_amax)
        self.ga = torch.tensor([ga], dtype=torch.float32, device=w.device)
        self.alpha = (1.0 / (self.ga * g.float())).reshape(1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import slice_nvfp4_output

        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, xs = scaled_fp4_quant(x2, self.ga, is_sf_swizzled_layout=True, backend="cutlass",
                                  padded_n=shape[-1] + self.pad * 2)
        out = cutlass_scaled_fp4_mm(xq, self.weight, xs, self.weight_scale, self.alpha, x.dtype)
        return slice_nvfp4_output(out, self.out_features).reshape(*shape[:-1], self.out_features)


@torch.no_grad()
def convert_dense(model: nn.Module) -> int:
    """Replace every Fp8Linear whose name does not match PRISMYRA_DENSE_KEEP (a regex) with Fp4Linear."""
    import re

    with open(os.environ["PRISMYRA_NVFP4_CALIB"]) as f:
        calib = json.load(f)["dense_in_amax"]
    keep = re.compile(os.environ.get("PRISMYRA_DENSE_KEEP", "^$"))
    search = os.environ.get("PRISMYRA_NVFP4_SEARCH") == "1"
    stats = _importance() if search else None
    done = 0
    for name, m in list(model.named_modules()):
        if type(m).__name__ != "Fp8Linear" or keep.search(name):
            continue
        key = name[name.index("language_model.") :] if "language_model." in name else name
        if key not in calib:
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        if stats and key in stats:
            imp = stats[key]["in"] / max(1.0, float(stats[key].get("tok", torch.tensor(1.0)).sum()))
        else:
            imp = None
        setattr(parent, name.rsplit(".", 1)[1], Fp4Linear(m, calib[key], imp, search))
        del m
        done += 1
        if os.environ.get("PRISMYRA_NVFP4_DEBUG") and done % 25 == 0:
            print("dense", done, round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)
    torch.cuda.empty_cache()
    return done
