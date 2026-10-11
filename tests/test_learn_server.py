"""`prismyra.server` with `--learn-spec`: wired through a stub engine, no device needed.

The one claim this file exists to check is DISTILL-RL-DESIGN-v2.md section 8's contract: without `learn_spec`,
`prismyra.learn` is never imported and `/ask`/`/v1/decide` answer exactly as they always did; with it, only a
registered tag's traffic is ever logged, off the request path.
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


PHISHING_SPEC = [
    {
        "task": "phishing-v1",
        "question": "Is there a limit?",
        "options": ["no", "yes"],
        "kind": "boolean",
        "normalize": "strip+lower",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "evals/x.jsonl",
    },
    {
        # `/v1/decide`'s own "noul" vocabulary (`decide.NOUL_OPTIONS`), registered separately: a `/ask`
        # `Boolean` question and a `/v1/decide` `noul` item never share a `kind` string or an option set
        # (`["no", "yes"]` versus `["false", "true"]`), so one tag cannot answer for both endpoints.
        "task": "phishing-decide-v1",
        "question": "any wording, matched by task name only",
        "options": ["false", "true"],
        "kind": "noul",
        "normalize": "none",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "evals/x.jsonl",
    },
]


def test_without_learn_spec_prismyra_learn_is_never_imported(client_factory):
    make, _ = client_factory
    for name in list(sys.modules):
        if name.startswith("prismyra.learn"):
            del sys.modules[name]
    client = make()
    response = client.post(
        "/ask",
        json={"context": "doc", "questions": [{"id": "a", "prompt": "Is there a limit?", "kind": "boolean"}]},
    )
    assert response.status_code == 200
    assert "learn" not in client.get("/stats").json()
    assert client.app.state.learn_hook is None
    assert not any(name.startswith("prismyra.learn") for name in sys.modules)


def test_an_untagged_request_with_learn_spec_on_is_answered_unchanged_and_not_logged(client_factory):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, PHISHING_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post(
        "/ask",
        json={"context": "doc", "questions": [{"id": "a", "prompt": "totally different question", "kind": "boolean"}]},
    )
    assert response.status_code == 200
    time.sleep(0.2)
    stats = client.get("/stats").json()
    assert stats["learn"]["logged"] == 0


def test_a_tagged_ask_request_is_logged_after_the_response(client_factory):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, PHISHING_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post(
        "/ask",
        json={"context": "buy now", "questions": [{"id": "a", "prompt": "Is there a limit?", "kind": "boolean"}]},
    )
    assert response.status_code == 200

    deadline = time.time() + 2.0
    stats = {}
    while time.time() < deadline:
        stats = client.get("/stats").json()
        if stats["learn"]["logged"] >= 1:
            break
        time.sleep(0.02)
    assert stats["learn"]["logged"] == 1
    assert stats["learn"]["dropped"] == 0

    rows = list((tmp_path / "exp" / "phishing-v1").glob("*.jsonl"))
    assert len(rows) == 1
    row = json.loads(rows[0].read_text().strip())
    assert row["context"] == "buy now"
    assert row["versions"]["backbone"] == "stub/model"


def test_a_tagged_decide_request_is_logged_including_by_task_name(client_factory):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, PHISHING_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post(
        "/v1/decide",
        json={"kind": "noul", "state": "buy now", "question": "unrelated wording", "task": "phishing-decide-v1"},
    )
    assert response.status_code == 200

    deadline = time.time() + 2.0
    stats = {}
    while time.time() < deadline:
        stats = client.get("/stats").json()
        if stats["learn"]["logged"] >= 1:
            break
        time.sleep(0.02)
    assert stats["learn"]["logged"] == 1
    rows = list((tmp_path / "exp" / "phishing-decide-v1").glob("*.jsonl"))
    row = json.loads(rows[0].read_text().strip())
    assert row["context"] == "buy now"
    assert row["kind"] == "noul"


def test_stats_omits_learn_when_disabled(client_factory):
    make, _ = client_factory
    client = make()
    assert "learn" not in client.get("/stats").json()
