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

## What answered: perigraph's model half on every answer

Every answer from `/ask`, `/v1/decide` and `/v1/chat/completions` carries a `perigraph` object: the model half of a
[perigraph](https://github.com/littlemex/perigraph) version 3 record for whoever gave that answer, written by this
process because only it holds what it loaded. It is unconditional (no flag), and it is not telemetry: it travels in
the response, not over OTLP. `GET /perigraph` returns the same records for the backbone and each routed tag's loaded
student, for looking rather than joining, because a student can be reloaded between that request and an answer.

| part | backbone | student served by `prismyra.learn` |
|---|---|---|
| `checkpoint` | the files `prismyra.identity.backbone_files` names, hashed after load: `model.safetensors`, or the index and every shard its `weight_map` names; `config.json` and processor configs; known tokenizer and chat-template files; `adapter_config.json` and `adapter_model.safetensors`; the `*.py` files when the config has an `auto_map`; and `--heads` and `--calibrate-temperature`, entered as `+heads/<name>` and `+temperature/<name>` | the artifact's bytes as parsed, under the fixed name `student.json`, so a renamed copy is the same artifact |
| `precision` | element counts by dtype over parameters and buffers, the quantisation read off the loaded tensors and module types (`fp8`, `nvfp4`, `none`), and the device | absent (`extraction_failed`): a tree model exposes no parameter dtype |
| `runtime` | `prismyra`, `torch`, `transformers`, `kernels`, `triton`, `safetensors`, `tokenizers`, as installed | `prismyra`, the tree library, `numpy` |
| `answer_mode` | `option_scores` | `classifier_output` |

So a backbone answer has a `perigraph.model_identity` (all four parts) and a `perigraph.artifact_identity`
(`checkpoint` alone). The same files loaded in another dtype, or under another runtime, are another model over the same
artifact; a retrain, or other heads, is another artifact. A student answer has an artifact identity and no model
identity. A part that could not be read is absent with its reason and a detail
(`perigraph.model_absent.<part>.reason`), and an absent identity means unknown, never clean.

**The harness half.** A record names every part, so the eleven harness parts are on each record too, all absent: this
process stamps what answered and reads none of the request's harness fields. `tool_extension` is `not_observable`, as
it is for every collector; the other nine are `not_reachable` with a detail saying so. The manifests are static:
`perigraph.collector_reaches` is the empty string and `perigraph.model_reaches` names all four model parts. A record
from this collector therefore has no harness identity, and a perigraph receiver stores it and refuses a verdict on it
until a collector that reads the request supplies one; the model and artifact identities are unaffected. A model part
recorded `not_reachable` against that manifest is a contradiction, so a stand-in engine's record is refused a verdict.
The opaque version strings (`perigraph/2/model/<id>`, `perigraph/2/artifact/<id>`) keep the 2 that perigraph section 11
recommends, although records are version 3: the identities are derived as version 2 derived them, so a load keeps
its string, and every `arm_version` built from it, across the record version.
`tools/perigraph_conform.py` is the adapter for perigraph's `tools/conform.py` (stamper role).

**Why each part.** Each part is there because an earlier distillation record changed answers while every name stayed
the same: retraining from the same script and seed agreed with the previous model on 93.4 % of labels (R5 to R6, R12
to R13); dynamic int8 at load left the stored bytes unchanged and agreed on 33-40 %; a rebuilt image that lost
packages agreed on 97 of 100 answers (R14); and a student trained on option scores generated "1111" when asked to
generate. A version string that is not content (the phase 6 spec version, which changed when the spec was edited for
deployment) cannot tell these apart.

**Cost.** The checkpoint is hashed once per engine load, after the model is loaded, with eight threads. Each file's
size, mtime and inode are taken before loading and again around the hash, and a file that changed in between makes the
part `extraction_failed` rather than a digest of bytes that were not loaded; the file list is taken after the load,
because a loader can download files it then reads. On a g6e.12xlarge pod reading a shared file system,
`littlemex/prismyra-decision-35b-a3b-fp8` (47 files, 31.8 GiB) hashed in 17.4 s the first time and in 4.4 s with its
files already in the page cache.

**Comparability.** A model identity is comparable only within one collector and collector version
(`perigraph.collector`, `perigraph.collector_version`), because `precision` and `runtime` are this collector's
rendering: upgrading Prismyra or any listed package changes it. An artifact identity is what a trainer can compute when
it saves the files, using the same names.

**What it cannot see.** Files a loader reads that `backbone_files` does not name (a layout it cannot name, such as
pickled `.bin` weights or a `.tiktoken` tokenizer, is refused as `extraction_failed` rather than identified); packages
that matter and are not listed; and anything a student's feature builder computes with, such as an embedding model or
a feature module's code. A wrong file list gives a confident wrong identity.

**What it exposes.** The labels name dtype counts, the device and package versions. They do not name the model path.

## Guarantee

With `--otel` unset (the default), `prismyra.telemetry` is never imported, `/ask` and `/v1/decide` answer
exactly as they do without it -- byte-identical responses, the same latency, the same code path -- and
`/metrics` still works (it only ever reads `/stats`, which exists regardless). With `--otel` set, the added
cost on the request path is one `record_request`/`record_rejection` call after the answer is already
decided; the stats-derived gauges cost nothing on the request path at all, since they are read by the SDK's
own export timer. Measured on real hardware: p50 within +2% and p99 within +5% of the same traffic with
`--otel` off (see `docs/PERFORMANCE.md`'s own methodology for how a before/after pair like this is
measured).
