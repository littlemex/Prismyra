"""The isolation `prismyra.telemetry` and `prismyra.learn` have to keep from each other: enabling one emits
none of the other's data, ever, and disabling both is byte-for-byte what every test in `tests/
test_learn_server.py` already checks for `prismyra.learn` alone. `FakeTelemetry` stands in for
`prismyra.telemetry.otel_backend.OtelTelemetry` -- this file needs no OTel SDK and no `pytest.importorskip`,
because the property under test is *which data goes where*, not whether the real OTel wiring itself works
(that is `tests/test_telemetry_otel_backend.py`'s job). A real `OtelTelemetry` is deliberately not used here
for a second reason: its `OTLPSpanExporter`/`OTLPMetricExporter` retry against an unreachable collector with
real backoff delays (confirmed by hand against `http://127.0.0.1:1`), which would make this file slow and
non-deterministic for a property that has nothing to do with network retries.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

from prismyra.schema import Answer, Result, Timing


class StubEngine:
    model_name = "stub/model"
    group = 32
    heads = None

    class _Tok:
        def __call__(self, text, **_):
            return {"input_ids": list(range(len(text.split())))}

    tokenizer = _Tok()

    def cache_bytes(self, tokens):
        return tokens * 1024

    def stats(self):
        return {"model": "stub"}

    def ask(self, context, questions, images=None, videos=None):
        answers = {
            q.id: Answer(
                id=q.id,
                kind=q.kind,
                value=q.value_of(q.options[0]),
                option=q.options[0],
                probabilities=dict.fromkeys(q.options, 1.0 / len(q.options)),
            )
            for q in questions
        }
        return Result(answers=answers, timing=Timing(context_ms=1.0, readout_ms=2.0), model=self.model_name)


class FakeTelemetry:
    """A `prismyra.telemetry` stand-in with the same shape `prismyra.server.create_app` calls against --
    see this file's own module docstring for why a fake, not a real `OtelTelemetry`, is the right level for
    an isolation test."""

    def __init__(self):
        self.requests: list[dict] = []
        self.rejections: list[dict] = []
        self.stats_sources: dict[str, object] = {}

    def register_stats_source(self, name, fn):
        self.stats_sources[name] = fn

    def record_request(self, **kw):
        self.requests.append(kw)

    def record_rejection(self, **kw):
        self.rejections.append(kw)

    def shutdown(self):
        pass


PHISHING_SPEC = [
    {
        "task": "phishing-v1",
        "question": "Is there a limit?",
        "options": ["no", "yes"],
        "kind": "boolean",
        "normalize": "strip+lower",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "e.jsonl",
    }
]


@pytest.fixture
def client_factory(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import prismyra
    from prismyra.server import create_app

    monkeypatch.setattr(prismyra, "Prismyra", lambda *a, **k: StubEngine())

    def make(**kwargs):
        app = create_app("stub/model", **kwargs)
        return TestClient(app)

    return make, tmp_path


def write_spec(tmp_path, entries):
    path = tmp_path / "learn.json"
    path.write_text(json.dumps(entries))
    return path


def _wait_for(predicate, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _fake_build_telemetry(monkeypatch):
    fake = FakeTelemetry()
    import prismyra.telemetry as telemetry_pkg

    monkeypatch.setattr(telemetry_pkg, "build_telemetry", lambda cfg: fake)
    return fake


def test_both_disabled_is_unchanged(client_factory):
    make, _ = client_factory
    client = make()
    assert client.app.state.learn_hook is None
    assert client.app.state.telemetry is None
    response = client.post(
        "/ask", json={"context": "doc", "questions": [{"id": "a", "prompt": "q?", "kind": "boolean"}]}
    )
    assert response.status_code == 200


def test_otel_only_emits_zero_learn_experience_and_imports_no_learn_module(client_factory, monkeypatch):
    make, _ = client_factory
    fake = _fake_build_telemetry(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("prismyra.learn"):
            del sys.modules[name]

    client = make(otel=True)
    assert client.app.state.learn_hook is None
    assert client.app.state.telemetry is fake

    response = client.post(
        "/ask", json={"context": "doc", "questions": [{"id": "a", "prompt": "q?", "kind": "boolean"}]}
    )
    assert response.status_code == 200
    assert len(fake.requests) == 1
    # Shape only -- never the request's own content.
    recorded = fake.requests[0]
    assert set(recorded) <= {"endpoint", "start_time", "end_time", "served_by", "status", "timing", "traceparent"}
    assert "doc" not in str(recorded)
    assert not any(name.startswith("prismyra.learn") for name in sys.modules)


def test_learn_only_emits_zero_telemetry(client_factory):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, PHISHING_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    assert client.app.state.telemetry is None

    response = client.post(
        "/ask",
        json={"context": "buy now", "questions": [{"id": "a", "prompt": "Is there a limit?", "kind": "boolean"}]},
    )
    assert response.status_code == 200
    stats = _wait_for(lambda: client.get("/stats").json()["learn"]["logged"] == 1)
    assert stats
    assert "telemetry" not in client.get("/stats").json()  # this package never adds a /stats key at all


def test_both_enabled_keep_their_data_in_separate_streams(client_factory, monkeypatch):
    make, tmp_path = client_factory
    fake = _fake_build_telemetry(monkeypatch)
    spec_path = write_spec(tmp_path, PHISHING_SPEC)
    client = make(otel=True, learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))

    response = client.post(
        "/ask",
        json={"context": "buy now", "questions": [{"id": "a", "prompt": "Is there a limit?", "kind": "boolean"}]},
    )
    assert response.status_code == 200

    assert len(fake.requests) == 1
    assert "buy now" not in str(fake.requests[0])  # telemetry's own record carries no request content

    assert _wait_for(lambda: client.get("/stats").json()["learn"]["logged"] == 1)
    rows = list((tmp_path / "exp" / "phishing-v1").glob("*.jsonl"))
    row = json.loads(rows[0].read_text().strip())
    assert row["context"] == "buy now"  # learn's own record keeps exactly what it always did


def test_metrics_endpoint_is_available_regardless_of_otel_or_learn(client_factory):
    make, _ = client_factory
    client = make()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prismyra_queue_depth" in response.text or "prismyra_" in response.text


def test_a_rejected_request_is_counted_by_telemetry_when_on(client_factory, monkeypatch):
    make, _ = client_factory
    fake = _fake_build_telemetry(monkeypatch)
    client = make(otel=True)
    # An explicit `raise HTTPException(422, ...)` inside the handler body (empty context, no media) -- not
    # a pydantic `RequestValidationError`, which FastAPI's own default handler answers before this
    # project's `HTTPException` handler would ever see the request.
    response = client.post("/ask", json={"context": "", "questions": [{"id": "a", "prompt": "q?", "kind": "boolean"}]})
    assert response.status_code == 422
    assert len(fake.rejections) == 1
    assert fake.rejections[0]["endpoint"] == "/ask"
