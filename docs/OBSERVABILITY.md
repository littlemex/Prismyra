# Observability: metrics, traces and logs about the server itself

`prismyra.telemetry` is Prismyra's own operational observability -- how many requests, how long they took,
how deep the queue is, what the batcher is doing, whether the device is saturated, which requests were
rejected before ever reaching it. It answers "is the server healthy", never "what did a request ask".

This is a different concern from [`prismyra.learn`'s experience log](LEARN.md), on purpose, and the line
between the two is absolute, not a convention either side could accidentally cross:

| | `prismyra.telemetry` (this document) | `prismyra.learn`'s experience log ([LEARN.md](LEARN.md)) |
|---|---|---|
| answers | is the server healthy, where did the time go | what did a *registered tag's* request ask, and what did it answer |
| carries | counts, durations, shapes (token counts, question counts, `kind`), version strings | a tagged request's own context, question, options, probabilities |
| enabled by | `--otel` | `--learn-spec`, and only for a request matching a registered tag |
| off by default | yes -- nothing under `prismyra.telemetry.otel_backend` is imported, and behaviour and latency are unchanged | yes (unchanged from before this document existed) |
| can the two share code | SDK bootstrap only (an OTel `LoggerProvider`, if both happen to be configured) | never data -- separate OTel `Logger` names, separate flags, no shared code path |

Turning on `--otel` alone never emits a single `Experience` row. Turning on `--learn-spec` alone never
emits a span or a metric through this package. Both together keep producing exactly their own data, each
unaware of the other's existence past the SDK bootstrap they may share. `tests/test_telemetry_isolation.py`
checks all four combinations.

## What is instrumented

| signal | source | when |
|---|---|---|
| request count, by endpoint and `served_by` | `/ask`, `/v1/decide`, after each is answered | `--otel` |
| request duration (histogram) | same | `--otel` |
| rejected-request count, by endpoint and status code | a single `HTTPException` handler shared by every endpoint | `--otel` |
| one span per request, with `queue`/`context`/`readout` events | reconstructed from `prismyra.schema.Timing`'s own already-measured durations (see "Traces" below for why this is reconstructed, not live) | `--otel` |
| every numeric field of `/stats` (engine, queue, batcher) | one generic observable gauge, read by the SDK's own export timer, never from a request | `--otel` |
| the same numbers, Prometheus text format, at `/metrics` | a direct, dependency-free conversion of `/stats` | always, independent of `--otel` |

Nothing in this list is a request's own content. `prismyra.telemetry.otel_backend._flatten_numeric` only
ever reports numeric leaves of `/stats`'s own dict; a string, `None`, or a list is never coerced into a
metric value.

## Traces: why reconstructed, not live

A live span would mean wrapping `/ask`/`/v1/decide`'s own request handler in `tracer.start_as_current_span`
around every branch -- admission, the batcher, the several existing `except` clauses -- which risks
destabilising control flow this project has already measured and tested carefully
([`docs/PERFORMANCE.md`](PERFORMANCE.md)). Instead, `record_request` builds one span *after* the response
is already decided, with an explicit `start_time`/`end_time` and three events (`queue`, `context`,
`readout`) placed at offsets taken from `Timing`'s own numbers. The span is accurate about *when* each
phase happened and *how long* it took; it is not "live" while the request is in flight, which is the one
thing this approach gives up in exchange for touching none of the existing request-handling code.

A caller's own trace context (a LangGraph agent, say, already inside its own trace) is honoured through the
standard `traceparent` HTTP header -- send it on `/ask`/`/v1/decide`, and the request's own span becomes a
child of that trace rather than starting a new one.

## Exporting: one OTLP connection, your own collector's routing

Prismyra never knows about CloudWatch, Prometheus, LangSmith, or any other destination. It only ever speaks
OTLP/HTTP to one collector endpoint (`--otel-endpoint`); everything past that is the collector's own
configuration. A minimal collector config that fans metrics to a Prometheus-shaped receiver and traces/logs
to a second destination, in the collector's own `pipelines` shape (no real hostnames, accounts, or
namespaces here -- fill in your own collector's actual exporter blocks):

```yaml
receivers:
  otlp:
    protocols:
      http:
exporters:
  prometheus:
    endpoint: "0.0.0.0:9464"
  # one real exporter block per actual destination, e.g. the collector contrib
  # distribution's own `awscloudwatchlogs`/`awsemf` exporters for CloudWatch
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
    traces:
      receivers: [otlp]
      exporters: [] # your own trace destination's exporter name
    logs:
      receivers: [otlp]
      exporters: [] # your own log destination's exporter name
```

If a collector is more than this needs right now, `/metrics` (always on, no `--otel`, no extra) already
gives Prometheus something to scrape directly -- see the table above for exactly which numbers it carries.

## Configuration

| flag | default | meaning |
|---|---|---|
| `--otel` | off | turn the whole package on |
| `--otel-endpoint` | `http://localhost:4318` | the OTLP/HTTP collector base URL |
| `--otel-service-name` | `prismyra` | the OTel `Resource`'s `service.name` |

Requires `pip install "prismyra[otel]"`; with `--otel` unset, nothing under
`prismyra.telemetry.otel_backend` is imported and the extra does not need to be installed at all.

## Guarantee

With `--otel` unset (the default), `prismyra.telemetry` is never imported, `/ask` and `/v1/decide` answer
exactly as they always did -- byte-identical responses, the same latency, the same code path -- and
`/metrics` still works (it only ever reads `/stats`, which exists regardless). With `--otel` set, the added
cost on the request path is one `record_request`/`record_rejection` call after the answer is already
decided; the stats-derived gauges cost nothing on the request path at all, since they are read by the SDK's
own export timer. Measured on real hardware: p50 within +2% and p99 within +5% of the same traffic with
`--otel` off (see `docs/PERFORMANCE.md`'s own methodology for how a before/after pair like this is
measured).
