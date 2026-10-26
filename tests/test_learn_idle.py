"""`prismyra.learn.idle`: the idle-time retraining loop, exercised against a tiny stdlib HTTP server
standing in for `prismyra.server` (its `/stats` and `/v1/learn/reload`) and a monkeypatched `refit_one_tag`
standing in for a real LightGBM fit -- this file is about the *polling and scheduling* logic (is the server
busy, is a tag due, did exactly one tag get picked per idle window), not about distillation itself, which
`tests/test_learn_students.py`/`tests/test_learn_fit.py` already cover.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from prismyra.learn import idle as idle_mod
from prismyra.learn.idle import IdleConfig, IdleError, RefitReport, TagIdleState, is_busy, load_state, save_state


def test_is_busy_reads_the_queue_depth():
    assert is_busy({"queue": {"depth": 1}}) is True
    assert is_busy({"queue": {"depth": 0}}) is False


def test_is_busy_also_reads_the_batcher_when_present():
    assert is_busy({"queue": {"depth": 0}, "batcher": {"queue_depth": 2}}) is True
    assert is_busy({"queue": {"depth": 0}, "batcher": {"queue_depth": 0}}) is False


def test_is_busy_tolerates_a_configurable_floor():
    assert is_busy({"queue": {"depth": 3}}, max_queue_depth=5) is False
    assert is_busy({"queue": {"depth": 6}}, max_queue_depth=5) is True


def test_tagidlestate_round_trips_through_json(tmp_path):
    state = {"t1": TagIdleState(last_fit_ts=123.0, last_fit_row_count=50, last_admitted=True)}
    save_state(tmp_path / "state.json", state)
    reloaded = load_state(tmp_path / "state.json")
    assert reloaded["t1"] == state["t1"]


def test_load_state_on_a_missing_file_is_empty(tmp_path):
    assert load_state(tmp_path / "nope.json") == {}


class _FakeServer:
    """A tiny HTTP server with mutable `/stats` and a recording `/v1/learn/reload`."""

    def __init__(self):
        self.stats = {"queue": {"depth": 0}}
        self.reload_calls: list[dict] = []
        handler = self._make_handler()
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/stats":
                    body = json.dumps(outer.stats).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                outer.reload_calls.append(json.loads(raw.decode("utf-8")))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        return Handler

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def stop(self):
        self.httpd.shutdown()
        self.thread.join(timeout=2.0)


@pytest.fixture
def fake_server():
    server = _FakeServer()
    yield server
    server.stop()


ROUTER_SPEC = [
    {
        "task": "snake-move-v1",
        "question": "which way?",
        "options": ["up", "down"],
        "kind": "choice",
        "normalize": "none",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "e.jsonl",
        "router": {"mode": "ab", "student_path": "students/snake.json", "ratio": 5000},
    }
]


def _make_spec(tmp_path):
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps(ROUTER_SPEC))
    return spec_path


def test_run_forever_refuses_a_spec_with_no_routed_tag(tmp_path, fake_server):
    plain = [{k: v for k, v in ROUTER_SPEC[0].items() if k != "router"}]
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps(plain))
    cfg = IdleConfig(
        server_url=fake_server.url,
        learn_spec=spec_path,
        log_dir=tmp_path / "exp",
        backbone="stub",
        package_version="0",
        state_path=tmp_path / "state.json",
    )
    with pytest.raises(IdleError, match="no tag has a router"):
        idle_mod.run_forever(cfg, stop_after=1, sleep=lambda _s: None)


def test_run_forever_never_refits_while_the_server_stays_busy(tmp_path, fake_server, monkeypatch):
    fake_server.stats = {"queue": {"depth": 5}}
    calls = []
    monkeypatch.setattr(idle_mod, "refit_one_tag", lambda *a, **k: calls.append(1))

    spec_path = _make_spec(tmp_path)
    cfg = IdleConfig(
        server_url=fake_server.url,
        learn_spec=spec_path,
        log_dir=tmp_path / "exp",
        backbone="stub",
        package_version="0",
        state_path=tmp_path / "state.json",
        idle_polls_required=2,
    )
    idle_mod.run_forever(cfg, stop_after=10, sleep=lambda _s: None)
    assert calls == []


def test_run_forever_refits_exactly_once_per_idle_window(tmp_path, fake_server, monkeypatch):
    fake_server.stats = {"queue": {"depth": 0}}
    calls: list[str] = []

    def fake_refit(spec_path, tag, cfg, state, now):
        calls.append(tag.task)
        row_count = idle_mod._count_experience_rows(tmp_path / "exp", tag.task)
        state[tag.task] = TagIdleState(last_fit_ts=now, last_fit_row_count=row_count, last_admitted=True)
        return RefitReport(task=tag.task, attempted=True, admitted=True, reloaded=False)

    monkeypatch.setattr(idle_mod, "refit_one_tag", fake_refit)

    spec_path = _make_spec(tmp_path)
    (tmp_path / "exp" / "snake-move-v1").mkdir(parents=True)
    with (tmp_path / "exp" / "snake-move-v1" / "2026-01-01.jsonl").open("w") as fh:
        for i in range(100):
            fh.write(json.dumps({"ts": i}) + "\n")

    cfg = IdleConfig(
        server_url=fake_server.url,
        learn_spec=spec_path,
        log_dir=tmp_path / "exp",
        backbone="stub",
        package_version="0",
        state_path=tmp_path / "state.json",
        idle_polls_required=2,
        min_new_rows=10,
        retrain_interval_s=0.0,
    )
    idle_mod.run_forever(cfg, stop_after=20, sleep=lambda _s: None)
    # Exactly once, not once per idle window reached: `fake_refit` advances `last_fit_row_count` to the row
    # count it was just given, and the experience file never grows past that in this test, so every later
    # idle window finds the tag no longer due (`min_new_rows` unmet) -- which is the property under test:
    # an idle tag whose experience has not grown is not refit again and again for no reason.
    assert calls == ["snake-move-v1"]
