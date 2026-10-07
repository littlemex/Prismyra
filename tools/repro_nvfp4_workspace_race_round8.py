"""inv5, round 8: minimal reproduction of the `lanes=2` NaN/illegal-memory-access failure
(`test_lanes_two_decisions_under_a_burst_do_not_move`, sm_120 only) traced to `FusedExpertsFp4._workspace`'s
class-level `_ws`/`_ws_size` cache (`prismyra/kernels/nvfp4.py`) -- shared by every layer *and* every lane,
with an unlocked check-then-create on a first-time (never-cached) shape. Two threads, each driving its own
CUDA stream (same pattern `Batcher(lanes=2)` uses), calling the SAME layer instance with a shape neither has
ever used before, racing to allocate the scratch workspace.

No checkpoint, no calibration file -- `prepare_layer` with random FP8 weights at the real model's shape, same
technique as `diag_nvfp4_moe_minimal_round7.py`.

Usage: python3 repro_nvfp4_workspace_race_round8.py [n_rounds]
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import prismyra.engine as _engine_mod  # noqa: E402
from prismyra.kernels import nvfp4  # noqa: E402

E, TOPK, K, N = 256, 8, 2048, 512  # real nvfp4-36l config.json shape


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


def run_once(seed: int) -> str | None:
    """One round: a *fresh* layer (so the workspace cache starts cold every time, matching the burst test's
    first-ever-shape situation) is called from two threads at once, each on its own stream, with a distinct,
    never-before-seen `m` (so both threads race on a genuine cache miss rather than one warming the other)."""
    _engine_mod._enable_batch_invariance()
    layer = build_layer()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    errors: list[str] = []
    outputs: dict[int, torch.Tensor] = {}

    def worker(idx: int, m: int) -> None:
        try:
            with torch.cuda.stream(streams[idx]):
                gen = torch.Generator(device="cuda").manual_seed(seed * 100 + idx)
                x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.1
                with torch.inference_mode():
                    out = layer.forward(x)
                streams[idx].synchronize()
                outputs[idx] = out.detach().clone()
                if torch.isnan(out).any():
                    errors.append(f"thread {idx} (m={m}): NaN in output")
        except Exception as e:  # noqa: BLE001
            errors.append(f"thread {idx} (m={m}): {type(e).__name__}: {e}")

    # distinct, odd, never-reused m values each round so the cache is genuinely cold for this exact shape
    m0, m1 = 7 + seed * 2, 9 + seed * 2
    threads = [threading.Thread(target=worker, args=(0, m0)), threading.Thread(target=worker, args=(1, m1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    torch.cuda.synchronize()
    if any(t.is_alive() for t in threads):
        return f"round {seed}: a thread hung (deadlock?)"
    if errors:
        return f"round {seed}: " + "; ".join(errors)
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"device={torch.cuda.get_device_name(0)} rounds={n}", flush=True)
    for i in range(n):
        failure = run_once(i)
        if failure is not None:
            print(f"FAILURE: {failure}", flush=True)
            sys.exit(1)
        print(f"round {i}: OK", flush=True)
    print("all rounds passed", flush=True)


if __name__ == "__main__":
    main()
