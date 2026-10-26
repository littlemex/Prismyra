"""`prismyra.learn.http_source.HttpPullExperienceSource`: the generic paginated HTTP/JSON pull contract, and
`prismyra.learn.fit.run`'s own `source=` override point -- both against a tiny stdlib HTTP server, never a
real destination.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from prismyra.learn.experience import Experience
from prismyra.learn.http_source import HttpPullExperienceSource, HttpSourceError

ROWS = [
    Experience(
        tag="t",
        spec_version="v",
        ts=float(i),
        context=f"doc{i}",
        question="q",
        options=("a", "b"),
        kind="noul",
        probabilities={"a": 0.1, "b": 0.9},
        answered_by="prismyra",
        versions={"package": "1", "backbone": "m", "spec": "v"},
        request_id=f"r{i}",
    )
    for i in range(5)
]


class _PagingServer:
    """Serves `ROWS` two at a time, honouring ``since`` and an opaque ``page`` cursor."""

    def __init__(self, page_size=2):
        self.page_size = page_size
        self.requests: list[dict] = []
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
                parsed = urlparse(self.path)
                qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                outer.requests.append(qs)
                since = float(qs["since"]) if "since" in qs else 0.0
                rows = [r for r in ROWS if r.ts >= since]
                start = int(qs.get("page", "0"))
                page_rows = rows[start : start + outer.page_size]
                next_page = str(start + outer.page_size) if start + outer.page_size < len(rows) else None
                body = json.dumps({"rows": [r.to_json() for r in page_rows], "next_page": next_page}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        return Handler

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/experiences"

    def stop(self):
        self.httpd.shutdown()
        self.thread.join(timeout=2.0)


@pytest.fixture
def server():
    s = _PagingServer()
    yield s
    s.stop()


def test_iter_experiences_follows_pagination_to_the_end(server):
    source = HttpPullExperienceSource(server.url)
    rows = list(source.iter_experiences("t"))
    assert [r.request_id for r in rows] == [f"r{i}" for i in range(5)]
    assert [r.context for r in rows] == [f"doc{i}" for i in range(5)]


def test_since_is_passed_through_as_a_query_parameter(server):
    source = HttpPullExperienceSource(server.url)
    rows = list(source.iter_experiences("t", since=2.0))
    assert [r.request_id for r in rows] == ["r2", "r3", "r4"]
    assert server.requests[0]["since"] == "2.0"


def test_an_unreachable_server_raises_a_named_error():
    source = HttpPullExperienceSource("http://127.0.0.1:1/experiences", timeout=1.0)
    with pytest.raises(HttpSourceError):
        list(source.iter_experiences("t"))


def test_a_response_missing_rows_raises(server):
    class _BadHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"not_rows": []}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    bad = HTTPServer(("127.0.0.1", 0), _BadHandler)
    thread = threading.Thread(target=bad.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = bad.server_address
        source = HttpPullExperienceSource(f"http://{host}:{port}/experiences")
        with pytest.raises(HttpSourceError, match="'rows'"):
            list(source.iter_experiences("t"))
    finally:
        bad.shutdown()
        thread.join(timeout=2.0)


def test_fit_run_accepts_an_explicit_source_override(server, tmp_path):
    """`prismyra.learn.fit.run`'s own `source=` parameter -- a tag's experience can come from anywhere an
    `ExperienceSource` reads from, not only `LocalExperienceSource`'s on-disk layout. Needs the `learn`
    extra (this is the one test in this file that does): `run()` always reaches `fit_student`, which always
    imports LightGBM, whichever source fed it the rows.
    """
    pytest.importorskip("lightgbm")
    import json as _json

    from prismyra.learn.fit import run
    from prismyra.learn.spec import load_spec

    spec_path = tmp_path / "learn.json"
    spec_path.write_text(
        _json.dumps(
            [
                {
                    "task": "t",
                    "question": "q",
                    "options": ["a", "b"],
                    "kind": "noul",
                    "normalize": "none",
                    "retain_days": 30,
                    "keep_hidden": False,
                    "eval_set": "does-not-exist.jsonl",
                }
            ]
        )
    )
    load_spec(spec_path)  # sanity: the spec itself is valid
    source = HttpPullExperienceSource(server.url)
    artifact = run(
        spec_path=spec_path,
        task="t",
        log_dir=None,
        package_version="0",
        backbone="stub",
        source=source,
        min_data_in_leaf=1,
    )
    # Not admitted (the eval_set does not exist), but that is exactly `fit.py`'s own, already-tested
    # admission logic -- what this test is checking is that it got to run at all, over every row the pull
    # server served, not over `log_dir`/the spec's own `experience/` directory, which does not exist here.
    assert not artifact.admitted
    assert artifact.metrics["deduped_n"] == 5
    assert len(server.requests) >= 3  # 5 rows at page_size=2: pages at 0, 2, 4
