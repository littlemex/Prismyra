"""The real `Telemetry` implementation, built on the OpenTelemetry SDK. Imported lazily -- only when
``--otel`` (or an equivalent constructor argument) is actually on -- so that nothing here is ever imported,
and nothing in ``opentelemetry`` ever runs, on the path this project's own contract calls "disabled"
(`prismyra.telemetry`'s own module docstring).

Three signals, one `Resource`, one OTLP/HTTP endpoint:

* **Traces.** One span per request (`record_request`), built *after* the request has already been answered
  (`start_time`/`end_time` passed in explicitly, not a live `start_as_current_span` wrapped around the
  request handler) -- this is a deliberate, stated boundary: it lets every request handler's own existing
  control flow (admission, batching, the several ``except`` branches `server.py` already has) stay exactly
  as it was, at the cost of the span not being "live" while the request is in flight. Three events
  (``queue``, ``context``, ``readout``) are reconstructed from `prismyra.schema.Timing`'s own already-measured
  durations, at their own offsets within the span -- the closest this can honestly get to "区切りは要求、
  prefill、枝、readout" without threading OTel context through `prismyra.engine`'s own internals.
* **Metrics.** A request counter and a duration histogram, from the same call; a rejection counter from
  `record_rejection`. `register_stats_source` turns any `/stats`-shaped dict into one generic observable
  gauge per numeric leaf, read by the SDK's own export timer -- never from the request path, so registering
  one costs nothing per request.
* **Logs.** Not produced by this class. `prismyra.learn.otel_sink.OtelExperienceSink` is where request
  *content* can be logged to OTel, and it is a wholly separate opt-in (see that module's own docstring for
  why the separation is structural, not a convention).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping

from .config import TelemetryConfig

_SANITIZE = re.compile(r"[^a-zA-Z0-9_.]")


def _flatten_numeric(prefix: str, obj: object, out: list[tuple[str, float]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numeric(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float)):  # bool is an int subclass, so this covers both
        out.append((prefix, float(obj)))


class OtelTelemetry:
    def __init__(self, cfg: TelemetryConfig, *, tracer_provider=None, meter_provider=None):
        """``tracer_provider``/``meter_provider``, when given, are used as-is instead of building a new OTLP
        exporter pointed at ``cfg.otlp_endpoint`` -- `tests/test_telemetry_otel_backend.py`'s own way to
        exercise every call this class makes against an in-memory exporter, without a real collector
        anywhere nearby. Every production call site (`prismyra.telemetry.build_telemetry`) leaves both
        `None`.
        """
        from opentelemetry.metrics import Observation
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        self._Observation = Observation
        self._propagator = TraceContextTextMapPropagator()
        self._owns_providers = tracer_provider is None and meter_provider is None

        resource = Resource.create({"service.name": cfg.service_name, **cfg.resource_attributes})

        if tracer_provider is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint.rstrip("/") + "/v1/traces"))
            )
        self._tracer_provider = tracer_provider
        self._tracer = self._tracer_provider.get_tracer("prismyra.telemetry")

        if meter_provider is None:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            metric_exporter = OTLPMetricExporter(endpoint=cfg.otlp_endpoint.rstrip("/") + "/v1/metrics")
            meter_provider = MeterProvider(
                resource=resource, metric_readers=[PeriodicExportingMetricReader(metric_exporter)]
            )
        self._meter_provider = meter_provider
        meter = self._meter_provider.get_meter("prismyra.telemetry")

        self._request_counter = meter.create_counter(
            "prismyra.requests", description="requests answered, by endpoint and served_by"
        )
        self._request_duration = meter.create_histogram(
            "prismyra.request.duration", unit="ms", description="total request duration"
        )
        self._rejection_counter = meter.create_counter(
            "prismyra.requests.rejected", description="requests refused before being answered"
        )
        self._stats_sources: dict[str, Callable[[], dict]] = {}
        meter.create_observable_gauge(
            "prismyra.stats", callbacks=[self._observe_stats], description="every numeric field of /stats, flattened"
        )

    def _observe_stats(self, _options=None):
        for source_name, fn in self._stats_sources.items():
            try:
                stats = fn()
            except Exception:  # noqa: BLE001 - a stats source failing must not break every other gauge
                continue
            flat: list[tuple[str, float]] = []
            _flatten_numeric("", stats, flat)
            for path, value in flat:
                yield self._Observation(value, {"source": source_name, "path": path})

    def register_stats_source(self, name: str, fn: Callable[[], dict]) -> None:
        """``fn`` is called by the SDK's own export timer, not from any request -- registering one, or ten,
        costs nothing on the request path at all."""
        self._stats_sources[name] = fn

    def extract_traceparent(self, headers: Mapping[str, str] | None):
        """A caller's own OTel trace context (e.g. a LangGraph agent calling this server), read from the
        standard ``traceparent`` header -- `None` when absent, which `record_request` reads as "no parent;
        this span starts its own trace", the ordinary case for a caller that is not itself inside a trace.
        """
        if not headers:
            return None
        return self._propagator.extract({"traceparent": headers["traceparent"]}) if "traceparent" in headers else None

    def record_request(
        self,
        *,
        endpoint: str,
        start_time: float,
        end_time: float,
        served_by: str = "prismyra",
        status: str = "ok",
        timing=None,
        traceparent: str | None = None,
    ) -> None:
        """One call, once per request, after the answer is already decided -- see the module docstring for
        why this is not a live, request-wrapping span. ``start_time``/``end_time`` are `time.monotonic()`
        values in seconds; only their *difference*, and their offsets from each other, are meaningful (OTel
        itself wants nanosecond epoch timestamps, which this converts to using the gap between
        `time.monotonic()` and `time.time()` measured once, at call time, so the reconstructed span lands at
        roughly the real wall-clock time without this class needing to carry that offset across calls).
        """
        context = self.extract_traceparent({"traceparent": traceparent}) if traceparent else None
        wall_offset = time.time() - time.monotonic()
        start_ns = int((start_time + wall_offset) * 1e9)
        end_ns = int((end_time + wall_offset) * 1e9)
        span = self._tracer.start_span(
            endpoint.lstrip("/") or "request",
            context=context,
            attributes={"prismyra.endpoint": endpoint, "prismyra.served_by": served_by, "prismyra.status": status},
            start_time=start_ns,
        )
        if timing is not None:
            offset = start_ns
            for name, ms in (
                ("queue", timing.queue_ms),
                ("context", timing.context_ms),
                ("readout", timing.readout_ms),
            ):
                span.add_event(name, {"duration_ms": ms}, timestamp=offset)
                offset += int(ms * 1e6)
        span.end(end_time=end_ns)

        attrs = {"endpoint": endpoint, "served_by": served_by, "status": status}
        self._request_counter.add(1, attrs)
        self._request_duration.record((end_time - start_time) * 1000.0, attrs)

    def record_rejection(self, *, endpoint: str, reason: str) -> None:
        self._rejection_counter.add(1, {"endpoint": endpoint, "reason": reason})

    def shutdown(self) -> None:
        if self._owns_providers:
            self._tracer_provider.shutdown()
            self._meter_provider.shutdown()
