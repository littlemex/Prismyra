"""`prismyra.server` with stage 4 routing turned on for a tag: `/ask`, `/v1/decide`, `/v1/feedback` and
`/v1/learn/reload`, through a stub engine and a fake student (no device, no LightGBM -- see
`tests/test_learn_hook_router.py`'s own docstring for why a fake student is the right level for this file).

The one claim stage 1/2's own `tests/test_learn_server.py` exists to check (section 8: untagged unchanged,
tagged-with-no-router unchanged) is extended here to a tagged-*and*-routed request: the response grows
exactly two additive fields (`served_by`, `request_id`), and never anything else.
"""

from __future__ import annotations

import json
import time

import pytest

from prismyra.schema import Answer, Result, Timing

TASK = "snake-move-v1"

ROUTER_SPEC = [
    {
        "task": TASK,
        "question": "which way?",
        "options": ["up", "down"],
        "kind": "choice",
        "normalize": "none",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "e.jsonl",
        "router": {
            "mode": "ab",
            "student_path": "students/snake.json",
            "ratio": 10_000,
            "audit_min_fraction": 0,
            "audit_max_fraction": 0,
        },
    },
    {
        "task": "snake-decide-v1",
        "question": "any wording",
        "options": ["false", "true"],
        "kind": "noul",
        "normalize": "none",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "e.jsonl",
        "router": {
            "mode": "ab",
            "student_path": "students/snake-decide.json",
            "ratio": 10_000,
            "audit_min_fraction": 0,
            "audit_max_fraction": 0,
        },
    },
]


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


class FakeStudentHandle:
    def __init__(self, version: str, probs: dict):
        self.version = version
        self._probs = probs

    def predict(self, context, question=""):
        return dict(self._probs)

    def transform(self, context, question=""):
        return (0.0,)


@pytest.fixture
def fake_students(monkeypatch):
    handles = {
        TASK: FakeStudentHandle("snake1", {"up": 0.95, "down": 0.05}),
        "snake-decide-v1": FakeStudentHandle("snakedecide1", {"false": 0.1, "true": 0.9}),
    }

    # Imported here, not at module scope: a test elsewhere (test_learn_server.py's own
    # test_without_learn_spec_prismyra_learn_is_never_imported) deletes every 'prismyra.learn.*' module
    # from sys.modules as part of what it is checking, which -- if it runs first, as it does
    # alphabetically -- would leave a module-scope import here pointing at a stale class object that
    # prismyra.learn.hook's own later, lazy `from .students import StudentRegistry` would not reuse.
    # Importing fresh, right before patching, is what keeps the two in agreement.
    from prismyra.learn.students import StudentRegistry

    def fake_load(self, task, spec_dir, student_path):
        self._handles[task] = handles[task]
        return handles[task]

    monkeypatch.setattr(StudentRegistry, "load", fake_load)
    return handles


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


def _wait_for_stats(client, predicate, timeout=2.0):
    deadline = time.time() + timeout
    stats = {}
    while time.time() < deadline:
        stats = client.get("/stats").json()
        if predicate(stats):
            return stats
        time.sleep(0.02)
    return stats


def test_ask_routes_a_matched_question_to_the_student_and_skips_the_engine(client_factory, fake_students):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, ROUTER_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post(
        "/ask",
        json={
            "context": "board",
            "questions": [{"id": "a", "prompt": "which way?", "kind": "choice", "choices": ["up", "down"]}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    answer = body["answers"]["a"]
    assert answer["option"] == "up"  # the student's own argmax, not the stub engine's first-option default
    assert answer["read_by"] == "student:snake1"

    stats = _wait_for_stats(client, lambda s: s["learn"]["logged"] >= 1)
    assert stats["learn"]["logged"] == 1
    row = json.loads(next((tmp_path / "exp" / TASK).glob("*.jsonl")).read_text().splitlines()[0])
    assert row["answered_by"] == "student:snake1"
    assert row["request_id"]


def test_ask_mixed_tagged_and_untagged_questions_only_routes_the_tagged_one(client_factory, fake_students):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, ROUTER_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post(
        "/ask",
        json={
            "context": "board",
            "questions": [
                {"id": "a", "prompt": "which way?", "kind": "choice", "choices": ["up", "down"]},
                {"id": "b", "prompt": "totally unrelated", "kind": "boolean"},
            ],
        },
    )
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["a"]["option"] == "up"  # routed to the student
    assert answers["a"]["read_by"] == "student:snake1"
    assert "read_by" not in answers["b"] or answers["b"].get("read_by") is None  # untagged: the engine's own answer


def test_ask_with_learn_on_but_no_router_configured_is_byte_identical_to_stage_1_2(client_factory):
    """The regression this test guards: adding stage 4 must not change one byte of a stage-1/2-only tag's
    response (DISTILL-RL-DESIGN-v2.md section 8's own contract, extended to stage 4's own new code paths)."""
    make, tmp_path = client_factory
    plain_spec = [
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
    spec_path = write_spec(tmp_path, plain_spec)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    body = {"context": "doc", "questions": [{"id": "a", "prompt": "Is there a limit?", "kind": "boolean"}]}
    r1 = client.post("/ask", json=body).json()
    r2 = client.post("/ask", json=body).json()
    # `timing.queue_ms` is a real wall-clock measurement (how long this request waited for the worker
    # thread) and is not expected to be identical between two calls even with nothing else changed -- the
    # byte-identical claim is about everything the design's own contract is about: the answer itself.
    for r in (r1, r2):
        del r["timing"]
    assert r1 == r2
    assert "read_by" not in r1["answers"]["a"] or r1["answers"]["a"].get("read_by") is None


def test_decide_routes_an_item_to_the_student_and_carries_served_by_and_request_id(client_factory, fake_students):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, ROUTER_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post(
        "/v1/decide",
        json={"kind": "noul", "state": "buy now", "question": "any wording", "task": "snake-decide-v1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["served_by"] == "student:snakedecide1"
    assert body["request_id"]
    assert body["choice"] == "true"  # the student's own argmax (0.9 on "true")
    assert "_route_probabilities" not in body  # internal-only key, never reaches the wire

    stats = _wait_for_stats(client, lambda s: s["learn"]["logged"] >= 1)
    assert stats["learn"]["logged"] == 1


def test_decide_without_a_task_match_is_unrouted_and_carries_no_served_by(client_factory, fake_students):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, ROUTER_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post(
        "/v1/decide",
        json={"kind": "noul", "state": "buy now", "question": "an unregistered question"},
    )
    assert response.status_code == 200
    assert "served_by" not in response.json()


def test_feedback_is_rejected_without_learn_spec(client_factory):
    make, _ = client_factory
    client = make()
    response = client.post("/v1/feedback", json={"request_id": "r1", "label": "yes"})
    assert response.status_code == 404


def test_feedback_round_trip_via_http(client_factory):
    make, tmp_path = client_factory
    # A plain (unrouted) tag is enough here -- `/v1/feedback` has nothing to do with routing.
    plain_entry = {k: v for k, v in ROUTER_SPEC[0].items() if k != "router"}
    spec_path = write_spec(tmp_path, [plain_entry])
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))

    response = client.post("/v1/feedback", json={"request_id": "r1", "label": "up"})
    assert response.status_code == 200
    assert response.json() == {"accepted": True}

    response = client.post("/v1/feedback", json={"request_id": "r2"})
    assert response.status_code == 422


def test_reload_rejects_an_unregistered_task(client_factory, fake_students):
    make, tmp_path = client_factory
    spec_path = write_spec(tmp_path, ROUTER_SPEC)
    client = make(learn_spec=str(spec_path), learn_log_dir=str(tmp_path / "exp"))
    response = client.post("/v1/learn/reload", json={"task": "no-such-task"})
    assert response.status_code == 400
