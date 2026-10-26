"""`prismyra.telemetry.prometheus.render_prometheus`: a dependency-free `/stats` -> Prometheus text
converter. No `pytest.importorskip` anywhere in this file -- that is the whole point of this module existing
separately from `prismyra.telemetry.otel_backend`.
"""

from __future__ import annotations

from prismyra.telemetry.prometheus import render_prometheus


def test_numeric_leaves_become_gauges_with_dotted_paths_flattened_to_underscores():
    text = render_prometheus({"engine": {"model": "stub"}, "queue": {"depth": 3}})
    lines = text.strip().splitlines()
    assert "prismyra_queue_depth 3" in lines


def test_a_bool_is_reported_as_zero_or_one():
    text = render_prometheus({"batcher": {"started": True, "stopped": False}})
    assert "prismyra_batcher_started 1" in text
    assert "prismyra_batcher_stopped 0" in text


def test_strings_and_none_and_lists_are_skipped_not_coerced():
    text = render_prometheus({"engine": {"model": "stub", "note": None, "tags": ["a", "b"]}})
    assert text.strip() == ""


def test_output_ends_with_a_trailing_newline():
    assert render_prometheus({"a": 1}).endswith("\n")
