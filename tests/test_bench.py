"""The harness's judgement, without a device: fitting the cost model, merging a file, failing a regression."""

from __future__ import annotations

import json

from prismyra.bench import compare, fit_cost_model, merge_into


def a_staircase(context: float, per_group: float, group: int, counts) -> dict[str, float]:
    return {str(n): round(context + per_group * -(-n // group), 1) for n in counts}


def test_the_two_constants_are_recovered_from_the_curve():
    curve = a_staircase(138.0, 91.6, 32, (1, 8, 32, 64, 128))
    fit = fit_cost_model(curve, group=32)
    assert abs(fit["context_ms"] - 138.0) < 0.5
    assert abs(fit["per_group_ms"] - 91.6) < 0.5
    assert fit["worst_residual_ms"] < 0.5


def test_a_curve_that_is_no_longer_a_staircase_shows_up_in_the_residual():
    """The constants alone cannot say the shape broke, which is why the worst residual is reported beside them."""
    curve = a_staircase(138.0, 91.6, 32, (1, 8, 32, 64, 128))
    curve["8"] = curve["8"] + 60.0  # a per-question cost has appeared inside a group
    fit = fit_cost_model(curve, group=32)
    assert fit["worst_residual_ms"] > 50.0


def test_a_slower_point_fails_and_a_faster_one_does_not(tmp_path):
    reference = tmp_path / "ref.json"
    reference.write_text(json.dumps({"whole_job": {"prismyra_ms": {"1": 100.0, "32": 200.0}}}))

    slower = {"whole_job": {"prismyra_ms": {"1": 100.0, "32": 260.0}}}
    faster = {"whole_job": {"prismyra_ms": {"1": 100.0, "32": 140.0}}}
    within = {"whole_job": {"prismyra_ms": {"1": 100.0, "32": 209.0}}}

    assert compare(slower, reference, tolerance=0.10) == 1
    assert compare(faster, reference, tolerance=0.10) == 0  # one-sided: faster is reported, not failed
    assert compare(within, reference, tolerance=0.10) == 0


def test_a_refresh_cannot_invent_a_baseline(tmp_path):
    """The point of merging rather than writing: the numbers this machine cannot measure must survive untouched."""
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "model": "m",
                "whole_job": {"prismyra_ms": {"1": 999.0}, "vllm_ms": {"1": 134}, "crossover_questions": 4},
                "context_pass_steps": {"baseline_ms": 288.1, "final_ms": 138.3, "steps": []},
                "kernels_against_vllm": {"convolution_30_layers": {"prismyra_ms": 3.0, "vllm_ms": 3.37}},
            }
        )
    )
    measured = {
        "measured_on": "2026-09-21",
        "machine": {"gpu": "another card"},
        "context": {"tokens": 4924},
        "method": "…",
        "cost_model": {"context_ms": 140.0},
        "whole_job": {"prismyra_ms": {"1": 225.9}},
    }
    merge_into(path, measured)

    after = json.loads(path.read_text())
    assert after["whole_job"]["prismyra_ms"] == {"1": 225.9}  # measured, so replaced
    assert after["whole_job"]["vllm_ms"] == {"1": 134}  # hand-recorded, so kept
    assert after["whole_job"]["crossover_questions"] == 4
    assert after["context_pass_steps"]["baseline_ms"] == 288.1
    assert "kernels_against_vllm" in after
