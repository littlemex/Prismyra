"""inv5, round 8 (second variant): the first repro (cold-cache allocation race) did not trigger in 30/30
rounds even against the unfixed key, so this tests the other half of the hypothesis -- once the workspace
buffer is warm (already allocated, large enough), does *reusing the same physical buffer as scratch* from two
concurrently-running CUTLASS calls (two lanes, two streams) corrupt either call's result, independent of any
allocation-time race? Warms the layer once (single call, populates the cache), then launches both lanes'
calls concurrently at a size the warm buffer already covers, and compares each thread's output against a
sequential (one-at-a-time) reference computed with its own exact input.

Usage: python3 repro_nvfp4_workspace_race_round8b.py [n_rounds]
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import prismyra.engine as _engine_mod  # noqa: E402
from prismyra.kernels import nvfp4  # noqa: E402

E, TOPK, K, N = 256, 8, 2048, 512


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(K, E, bias=False)
        self.shared_expert_gate = torch.nn.Linear(K, 1, bias=False)
        self.shared_expert = torch.nn.Sequential(torch.nn.Linear(K, N), torch.nn.SiLU(), torch.nn.Linear(N, K))


def build_layer() -> nvfp4.FusedExpertsFp4:
    w13 = (torch.randn(E, 2 * N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s13 = torch.ones(E, (2 * N) // 128, K // 128, device="cuda")
    w2 = (torch.randn(E, K, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s2 = torch.ones(E, K // 128, N // 128, device="cuda")
    prepared = nvfp4.prepare_layer(w13, s13, w2, s2, "cuda")
    act = {"w13_in_amax": 1.0, "w2_in_amax": [1.0]}
    block = FakeBlock().to("cuda", torch.bfloat16)
    return nvfp4.FusedExpertsFp4(block, TOPK, prepared, act, "cuda")


def run_once(seed: int, layer: nvfp4.FusedExpertsFp4) -> str | None:
    m0, m1 = 11, 13  # fixed sizes, well inside the warm buffer from the very first call ever made
    gen0 = torch.Generator(device="cuda").manual_seed(seed * 100)
    x0 = torch.randn(m0, K, dtype=torch.bfloat16, device="cuda", generator=gen0) * 0.1
    gen1 = torch.Generator(device="cuda").manual_seed(seed * 100 + 1)
    x1 = torch.randn(m1, K, dtype=torch.bfloat16, device="cuda", generator=gen1) * 0.1

    # Sequential reference: one at a time, no concurrency, on the default stream.
    with torch.inference_mode():
        ref0 = layer.forward(x0.clone()).detach().clone()
        ref1 = layer.forward(x1.clone()).detach().clone()

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs: dict[int, torch.Tensor] = {}
    errors: list[str] = []

    def worker(idx: int, x: torch.Tensor) -> None:
        try:
            with torch.cuda.stream(streams[idx]):
                with torch.inference_mode():
                    out = layer.forward(x.clone())
                streams[idx].synchronize()
                outputs[idx] = out.detach().clone()
        except Exception as e:  # noqa: BLE001
            errors.append(f"thread {idx}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(0, x0)), threading.Thread(target=worker, args=(1, x1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    torch.cuda.synchronize()
    if any(t.is_alive() for t in threads):
        return f"round {seed}: a thread hung (deadlock?)"
    if errors:
        return f"round {seed}: " + "; ".join(errors)
    for idx, ref in ((0, ref0), (1, ref1)):
        if idx not in outputs:
            return f"round {seed}: thread {idx} produced no output"
        out = outputs[idx]
        if torch.isnan(out).any():
            return f"round {seed}: thread {idx} output has NaN"
        if not torch.equal(out, ref):
            d = (out.float() - ref.float()).abs().max().item()
            return f"round {seed}: thread {idx} output differs from sequential reference, max|diff|={d:.3e}"
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    _engine_mod._enable_batch_invariance()
    print(f"device={torch.cuda.get_device_name(0)} rounds={n}", flush=True)
    layer = build_layer()
    # Warm the cache once up front with a size at least as large as either m used below.
    with torch.inference_mode():
        layer.forward(torch.randn(16, K, dtype=torch.bfloat16, device="cuda") * 0.1)
    for i in range(n):
        failure = run_once(i, layer)
        if failure is not None:
            print(f"FAILURE: {failure}", flush=True)
            sys.exit(1)
        print(f"round {i}: OK", flush=True)
    print("all rounds passed", flush=True)


if __name__ == "__main__":
    main()
