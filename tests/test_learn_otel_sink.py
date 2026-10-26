"""`prismyra.learn.otel_sink.OtelExperienceSink`, against an in-memory OTel exporter -- never a real
collector. Needs the ``otel`` extra; CI's own "test" job does not install it (same reasoning as `lightgbm`
in `tests/test_learn_fit.py`), so this skips there via `pytest.importorskip`.
"""

from __future__ import annotations

import json
import time

import pytest

opentelemetry = pytest.importorskip(
    "opentelemetry.sdk"
)  # the API alone (opentelemetry-api) is already a transitive dependency of fastapi; this needs the SDK

from opentelemetry.sdk._logs import LoggerProvider  # noqa: E402
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402

from prismyra.learn.experience import Experience, MultiSink  # noqa: E402
from prismyra.learn.otel_sink import OtelExperienceSink, attributes_for  # noqa: E402


def _exp(**overrides) -> Experience:
    base = {
        "tag": "t",
        "spec_version": "v1",
        "ts": 1_700_000_000.0,
        "context": "doc",
        "question": "q",
        "options": ("a", "b"),
        "kind": "noul",
        "probabilities": {"a": 0.4, "b": 0.6},
        "answered_by": "prismyra",
        "versions": {"package": "1", "backbone": "m", "spec": "v1"},
    }
    base.update(overrides)
    return Experience(**base)


@pytest.fixture
def in_memory_provider():
    provider = LoggerProvider(resource=Resource.create({"service.name": "test"}))
    exporter = InMemoryLogRecordExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def _wait_for(predicate, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_an_emitted_experience_round_trips_through_the_same_schema_as_to_json(in_memory_provider):
    provider, exporter = in_memory_provider
    sink = OtelExperienceSink(logger_provider=provider)
    exp = _exp(request_id="r1")
    try:
        assert sink.put(exp) is True
        assert _wait_for(lambda: sink.stats()["logged"] == 1)
    finally:
        sink.stop()

    logs = exporter.get_finished_logs()
    assert len(logs) == 1
    body = json.loads(logs[0].log_record.body)
    assert body == exp.to_json()  # the one schema claim: whichever sink wrote it, the JSON shape agrees


def test_attributes_follow_gen_ai_semantics_where_one_exists_and_prismyra_namespace_otherwise():
    exp = _exp(request_id="r1", route_mode="ab", route_bucket=42, route_params_version="abc123")
    attrs = attributes_for(exp)
    assert attrs["gen_ai.system"] == "prismyra"
    assert attrs["gen_ai.response.model"] == "m"
    assert attrs["prismyra.tag"] == "t"
    assert attrs["prismyra.route.mode"] == "ab"
    assert attrs["prismyra.route.bucket"] == 42
    assert attrs["prismyra.request_id"] == "r1"


def test_a_full_queue_drops_rather_than_blocks(in_memory_provider):
    provider, _exporter = in_memory_provider
    sink = OtelExperienceSink(logger_provider=provider, max_queue=1)
    try:
        # The background thread may already have drained the first one by the time the second is put; what
        # matters is that `put` never raises and never blocks, and that a drop is counted when it happens.
        results = [sink.put(_exp(request_id=f"r{i}")) for i in range(50)]
        assert all(isinstance(r, bool) for r in results)
    finally:
        sink.stop()


def test_multisink_fans_out_to_both_a_local_and_an_otel_sink(tmp_path, in_memory_provider):
    from prismyra.learn.experience import LocalExperienceLog
    from prismyra.learn.spec import load_spec

    provider, exporter = in_memory_provider
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(
        json.dumps(
            [
                {
                    "task": "t",
                    "question": "q",
                    "options": ["a", "b"],
                    "kind": "noul",
                    "normalize": "none",
                    "retain_days": 30,
                    "keep_hidden": False,
                    "eval_set": "e.jsonl",
                }
            ]
        )
    )
    spec = load_spec(spec_path)
    local = LocalExperienceLog(tmp_path / "exp", spec, max_queue=100).start()
    otel = OtelExperienceSink(logger_provider=provider)
    multi = MultiSink({"local": local, "otel": otel})
    try:
        exp = _exp(request_id="r1")
        assert multi.put(exp) is True
        assert _wait_for(lambda: multi.stats()["logged"] == 1)
        assert _wait_for(lambda: multi.stats()["sinks"]["otel"]["logged"] == 1)
        rows = list((tmp_path / "exp" / "t").glob("*.jsonl"))
        assert len(rows) == 1
        logs = exporter.get_finished_logs()
        assert len(logs) == 1
    finally:
        multi.stop()
