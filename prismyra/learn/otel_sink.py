"""The second `prismyra.learn.experience.ExperienceSink`: emits one tagged request's `Experience` as an
OpenTelemetry log record, exported over OTLP to whatever collector an operator points it at. Requires the
``otel`` extra (``pip install "prismyra[otel]"``); importing *this module* is the one place that extra is
required -- `prismyra.learn.hook.LearnHook` only imports it when a sink config actually names ``"otel"``.

**This is the content-carrying path, by design.** DISTILL-RL-DESIGN-v2.md's whole reason a tag has to be
registered before any of its traffic is logged (`prismyra.learn.spec`'s own docstring) is still exactly
enforced here -- this sink only ever receives an `Experience` `prismyra.learn.hook.LearnHook` already built
for a request that matched a registered tag, the same call site `LocalExperienceLog.put` has always had.
This is a different *stream* from `prismyra.telemetry`'s own OTel usage, never the same one: that package's
own docstring is explicit that operational telemetry carries no request content at all, and the separation
is structural (a different OTel `Logger` name, a different enable flag, no shared code path) rather than a
convention either module could accidentally violate from the other side.

Attribute names follow OpenTelemetry's own semantic conventions where one already exists (``gen_ai.*``);
everything this project's own schema adds that no convention names yet goes under the ``prismyra.*``
namespace, per that convention's own guidance for a project-specific attribute. The log record's ``body`` is
the same JSON `Experience.to_json()` produces -- one schema, whichever sink wrote it; `tests/
test_learn_otel_sink.py` checks this against an in-memory exporter, never a real collector.
"""

from __future__ import annotations

import json
import queue
import threading
import time

from .experience import Experience, LogStats


def attributes_for(exp: Experience) -> dict:
    attrs: dict[str, str | int] = {
        "gen_ai.system": "prismyra",
        "gen_ai.response.model": exp.versions.get("backbone", ""),
        "prismyra.tag": exp.tag,
        "prismyra.spec_version": exp.spec_version,
        "prismyra.answered_by": exp.answered_by,
        "prismyra.kind": exp.kind,
        "prismyra.package_version": exp.versions.get("package", ""),
    }
    if exp.request_id:
        attrs["prismyra.request_id"] = exp.request_id
    if exp.route_mode is not None:
        attrs["prismyra.route.mode"] = exp.route_mode
    if exp.route_bucket is not None:
        attrs["prismyra.route.bucket"] = exp.route_bucket
    if exp.route_params_version is not None:
        attrs["prismyra.route.params_version"] = exp.route_params_version
    return attrs


class OtelExperienceSink:
    """Bounded queue, one background thread -- `LocalExperienceLog`'s own shape, reused here so both sinks
    keep the same "drop rather than block" contract (section 8). ``logger_provider``, when given, is used
    as-is (this is how a caller shares one already-configured OTel SDK bootstrap, e.g. with `prismyra.
    telemetry`'s own provider, across more than one exporter without this class knowing or caring); left
    `None`, this builds and owns its own, pointed at ``endpoint``.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        service_name: str = "prismyra",
        headers: dict | None = None,
        max_queue: int = 10_000,
        logger_provider=None,
    ):
        if logger_provider is None:
            if endpoint is None:
                raise ValueError("OtelExperienceSink needs either 'endpoint' or an already-built 'logger_provider'")
            from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
            from opentelemetry.sdk._logs import LoggerProvider
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": service_name})
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, headers=headers or {}))
            )
            self._owns_provider = True
        else:
            self._owns_provider = False
        self._provider = logger_provider
        self._logger = logger_provider.get_logger("prismyra.learn.experience")

        self._stats = LogStats()
        self._q: queue.Queue[Experience] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="prismyra-learn-otel-sink", daemon=True)
        self._thread.start()

    def put(self, exp: Experience) -> bool:
        try:
            self._q.put_nowait(exp)
            return True
        except queue.Full:
            self._stats.dropped += 1
            return False

    def queue_depth(self) -> int:
        return self._q.qsize()

    def stats(self) -> dict:
        return self._stats.as_dict(self.queue_depth())

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._owns_provider:
            self._provider.shutdown()

    def _run(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                exp = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._emit(exp)
                self._stats.logged += 1
            except Exception:  # noqa: BLE001 - an export failure must not kill this thread, the same
                # discipline LocalExperienceLog._run already applies to a disk write failure.
                self._stats.dropped += 1

    def _emit(self, exp: Experience) -> None:
        from opentelemetry._logs import SeverityNumber

        now_ns = int(time.time() * 1e9)
        self._logger.emit(
            timestamp=int(exp.ts * 1e9),
            observed_timestamp=now_ns,
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            body=json.dumps(exp.to_json(), ensure_ascii=False),
            attributes=attributes_for(exp),
        )
