"""`/v1/chat/completions`: a learn-spec tag asked in the OpenAI chat shape, answered through `/v1/decide`."""

from __future__ import annotations

import json
import math

import pytest

from prismyra.identity import stand_in
from prismyra.schema import Answer, Result, Timing

TASK = "sentiment-v1"
SPEC = [{"task": TASK, "question": "What is the sentiment?", "options": ["bearish", "bullish", "neutral"],
         "kind": "choice", "normalize": "strip+lower", "retain_days": 30, "keep_hidden": False,
         "eval_set": "e.jsonl"}]


class StubEngine:
    model_name = "stub/model"
    model_stamp = stand_in("a test stub loads no model")
    group = 32
    heads = None
    seen: list = []

    class _Tok:
        """Every distinct string is one token (as in tests/test_decide.py), so choice labels can be planned."""

        def __call__(self, text, **_):
            return {"input_ids": [hash(text.strip()) % 10_000_000]}

    tokenizer = _Tok()

    def cache_bytes(self, tokens):
        return tokens * 1024

    def stats(self):
        return {"model": "stub"}

    def ask(self, context, questions, images=None, videos=None):
        StubEngine.seen.append((context, [q.prompt for q in questions]))
        answers = {}
        for q in questions:
            p = dict(zip(q.options, [0.7, 0.2, 0.1][:len(q.options)]))
            answers[q.id] = Answer(id=q.id, kind=q.kind, value=q.value_of(q.options[0]), option=q.options[0],
                                   probabilities=p)
        return Result(answers=answers, timing=Timing(context_ms=1.0, readout_ms=2.0), model=self.model_name)


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import prismyra
    from prismyra.server import create_app

    monkeypatch.setattr(prismyra, "Prismyra", lambda *a, **k: StubEngine())
    spec = tmp_path / "learn.json"
    spec.write_text(json.dumps(SPEC))
    StubEngine.seen = []
    return TestClient(create_app("stub/model", learn_spec=str(spec), learn_log_dir=str(tmp_path / "exp")))


def chat(client, **over):
    body = {"model": TASK, "messages": [{"role": "system", "content": "ignored"},
                                        {"role": "user", "content": "Shares fell sharply"}]}
    body.update(over)
    return client.post("/v1/chat/completions", json=body)


def test_answers_as_one_assistant_message_through_decide(client):
    r = chat(client)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == TASK
    (c,) = body["choices"]
    assert c["message"] == {"role": "assistant", "content": "bearish"} and c["finish_reason"] == "stop"
    assert "logprobs" not in c
    assert body["prismyra"]["probabilities"]["bearish"] == pytest.approx(0.7)
    assert StubEngine.seen and "Shares fell sharply" in StubEngine.seen[0][0]


def test_logprobs_are_the_natural_log_of_prismyras_probabilities(client):
    c = chat(client, logprobs=True, top_logprobs=3).json()["choices"][0]
    (tok,) = c["logprobs"]["content"]
    assert tok["token"] == "bearish" and tok["logprob"] == pytest.approx(math.log(0.7))
    assert [t["token"] for t in tok["top_logprobs"]] == ["bearish", "bullish", "neutral"]
    assert tok["top_logprobs"][2]["logprob"] == pytest.approx(math.log(0.1))


@pytest.mark.parametrize("over, status", [
    ({"model": "nope"}, 404),
    ({"stream": True}, 400),
    ({"messages": [{"role": "system", "content": "x"}]}, 422),
    ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, 422),
])
def test_refusals(client, over, status):
    assert chat(client, **over).status_code == status


def test_text_content_parts_are_joined(client):
    r = chat(client, messages=[{"role": "user", "content": [{"type": "text", "text": "Shares"},
                                                            {"type": "text", "text": "fell"}]}])
    assert r.status_code == 200 and "Shares\nfell" in StubEngine.seen[0][0]


def test_without_a_learn_spec_the_endpoint_is_absent(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import prismyra
    from prismyra.server import create_app

    monkeypatch.setattr(prismyra, "Prismyra", lambda *a, **k: StubEngine())
    client = TestClient(create_app("stub/model"))
    assert chat(client).status_code == 404
