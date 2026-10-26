"""`prismyra.telemetry.otel_backend.OtelTelemetry`, against in-memory OTel exporters -- never a real
collector. Needs the ``otel`` extra; skips via `pytest.importorskip` where that is not installed, the same
pattern every other optional-extra test file in this project already uses.
"""

from __future__ import annotations

import time

import pytest

opentelemetry = pytest.importorskip("opentelemetry")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from prismyra.schema import Timing  # noqa: E402
from prismyra.telemetry.config import TelemetryConfig  # noqa: E402
from prismyra.telemetry.otel_backend import OtelTelemetry  # noqa: E402


@pytest.fixture
def telemetry():
    resource = Resource.create({"service.name": "test"})
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    t = OtelTelemetry(TelemetryConfig(), tracer_provider=tracer_provider, meter_provider=meter_provider)
    yield t, span_exporter, metric_reader


def _metric(reader, name):
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    return m
    return None


def test_record_request_emits_one_span_with_three_reconstructed_events(telemetry):
    t, span_exporter, _reader = telemetry
    t0 = time.monotonic()
    timing = Timing(queue_ms=5.0, context_ms=20.0, readout_ms=3.0)
    t.record_request(endpoint="/ask", start_time=t0, end_time=t0 + 0.05, served_by="prismyra", timing=timing)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ask"
    assert span.attributes["prismyra.endpoint"] == "/ask"
    assert span.attributes["prismyra.served_by"] == "prismyra"
    event_names = [e.name for e in span.events]
    assert event_names == ["queue", "context", "readout"]
    assert span.events[0].attributes["duration_ms"] == 5.0


def test_record_request_increments_the_request_counter_and_duration_histogram(telemetry):
    t, _span_exporter, reader = telemetry
    t0 = time.monotonic()
    t.record_request(endpoint="/ask", start_time=t0, end_time=t0 + 0.01, served_by="student:v1")

    counter = _metric(reader, "prismyra.requests")
    assert counter is not None
    assert sum(dp.value for dp in counter.data.data_points) == 1

    histogram = _metric(reader, "prismyra.request.duration")
    assert histogram is not None
    assert sum(dp.count for dp in histogram.data.data_points) == 1


def test_record_rejection_increments_the_rejection_counter(telemetry):
    t, _span_exporter, reader = telemetry
    t.record_rejection(endpoint="/ask", reason="503")
    counter = _metric(reader, "prismyra.requests.rejected")
    assert counter is not None
    assert sum(dp.value for dp in counter.data.data_points) == 1


def test_register_stats_source_is_read_by_the_observable_gauge_not_by_the_caller(telemetry):
    t, _span_exporter, reader = telemetry
    calls = {"n": 0}

    def fake_stats():
        calls["n"] += 1
        return {"depth": 3, "model": "stub", "ok": True}

    t.register_stats_source("queue", fake_stats)
    gauge = _metric(reader, "prismyra.stats")
    assert gauge is not None
    # Collected by the reader's own export pull, not by `register_stats_source` itself.
    assert calls["n"] >= 1
    paths = {dp.attributes["path"] for dp in gauge.data.data_points}
    assert "depth" in paths and "ok" in paths and "model" not in paths  # a string is not a number


def test_a_traceparent_header_sets_the_spans_parent_trace(telemetry):
    t, span_exporter, _reader = telemetry
    traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    t0 = time.monotonic()
    t.record_request(endpoint="/ask", start_time=t0, end_time=t0 + 0.001, traceparent=traceparent)
    spans = span_exporter.get_finished_spans()
    assert format(spans[0].context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
