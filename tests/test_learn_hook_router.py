"""`prismyra.learn.hook.LearnHook`'s stage 4 surface (`route`, `record_route`, `submit_feedback`,
`reload_student`), exercised end to end with a *fake* student registry -- real `StudentHandle`/
`StudentRegistry.load` need the `learn` extra (LightGBM) and are covered separately, in
`tests/test_learn_students.py`, behind `pytest.importorskip("lightgbm")`. `LearnHook` itself never imports
LightGBM directly; it only constructs a `prismyra.learn.students.StudentRegistry`, which is exactly what this
file replaces.
"""

from __future__ import annotations

import json
import time

import pytest

from prismyra.learn.hook import LearnHook

TASK = "snake-move-v1"
ENTRY_AB = {
    "task": TASK,
    "question": "which way?",
    "options": ["up", "down"],
    "kind": "choice",
    "normalize": "none",
    "retain_days": 30,
    "keep_hidden": False,
    "eval_set": "evals/x.jsonl",
    # `audit_*_fraction` pinned to 0: without it, a cold tag's audit lane defaults to its own safe maximum
    # (`Controller._audit_fraction_locked`'s "no volume signal yet" branch), which would make whether any one
    # test input happens to land in that lane depend on its hash -- fine for production (favour auditing when
    # unsure), but exactly the kind of input-dependent flakiness a deterministic test must not have.
    "router": {
        "mode": "ab",
        "student_path": "students/snake.json",
        "ratio": 10_000,
        "audit_min_fraction": 0,
        "audit_max_fraction": 0,
    },
}


class FakeStudentHandle:
    """`prismyra.learn.students.StudentHandle`'s own interface, without LightGBM. ``probs`` is returned by
    every `predict` call, and ``transform`` returns a fixed-length stand-in feature vector -- both plenty for
    exercising `LearnHook`'s own logic, which never looks past the shape of either."""

    def __init__(self, version: str, probs: dict[str, float]):
        self.version = version
        self._probs = probs
        self.calls = 0

    def predict(self, context: str, question: str = "") -> dict[str, float]:
        self.calls += 1
        return dict(self._probs)

    def transform(self, context: str, question: str = "") -> tuple[float, ...]:
        return (1.0, 2.0, 3.0)


@pytest.fixture
def fake_student(monkeypatch):
    """Installs a `FakeStudentHandle` wherever `StudentRegistry.load` would otherwise build a real one --
    `make_hook` below constructs a `LearnHook` for a spec with a router block, which eagerly calls `load` once
    per routed tag in `__init__`.
    """
    handle = FakeStudentHandle(version="fake1", probs={"up": 0.9, "down": 0.1})

    # Imported here, not at module scope: a test elsewhere (test_learn_server.py's own
    # test_without_learn_spec_prismyra_learn_is_never_imported) deletes every 'prismyra.learn.*' module
    # from sys.modules as part of what it is checking, which -- if it runs first, as it does
    # alphabetically -- would leave a module-scope import here pointing at a stale class object that
    # prismyra.learn.hook's own later, lazy `from .students import StudentRegistry` would not reuse.
    # Importing fresh, right before patching, is what keeps the two in agreement.
    from prismyra.learn.students import StudentRegistry

    def fake_load(self, task, spec_dir, student_path):
        self._handles[task] = handle
        return handle

    monkeypatch.setattr(StudentRegistry, "load", fake_load)
    return handle


def make_hook(tmp_path, entries, **kwargs):
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps(entries))
    return LearnHook(spec_path, package_version="9.9.9", backbone="stub/model", **kwargs)


def _wait_for(predicate, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_tag_with_no_router_has_route_return_none(tmp_path):
    entry = {k: v for k, v in ENTRY_AB.items() if k != "router"}
    hook = make_hook(tmp_path, [entry], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        assert tag is not None
        assert hook.route(tag, question="which way?", options=["up", "down"], kind="choice", context="board") is None
    finally:
        hook.stop()


def test_a_routed_tag_with_ratio_10000_always_serves_the_student(tmp_path, fake_student):
    hook = make_hook(tmp_path, [ENTRY_AB], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        outcome = hook.route(tag, question="which way?", options=["up", "down"], kind="choice", context="board")
        assert outcome is not None
        assert outcome.served_by == "student:fake1"
        assert outcome.needs_engine is False
    finally:
        hook.stop()


def test_record_route_for_a_student_served_unaudited_request_logs_without_an_engine_call(tmp_path, fake_student):
    hook = make_hook(tmp_path, [ENTRY_AB], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        outcome = hook.route(tag, question="which way?", options=["up", "down"], kind="choice", context="board")
        hook.record_route(
            tag,
            outcome,
            context="board",
            question="which way?",
            options=["up", "down"],
            kind="choice",
            engine_probabilities=None,
        )
        assert _wait_for(lambda: hook.stats()["logged"] == 1)
        rows = list((tmp_path / "exp" / TASK).glob("*.jsonl"))
        row = json.loads(rows[0].read_text().splitlines()[0])
        assert row["answered_by"] == "student:fake1"
        assert row["request_id"] == outcome.request_id
        assert row["route_mode"] == "ab"
        assert row["comparison_probabilities"] is None  # not audited -- nothing to compare against
        assert row["probabilities"] == {"up": 0.9, "down": 0.1}
    finally:
        hook.stop()


def test_record_route_for_prismyra_served_request_stores_the_engine_answer_as_probabilities(tmp_path, fake_student):
    entry = {**ENTRY_AB, "router": {**ENTRY_AB["router"], "ratio": 0}}  # never routes to the student
    hook = make_hook(tmp_path, [entry], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        outcome = hook.route(tag, question="which way?", options=["up", "down"], kind="choice", context="board")
        assert outcome.served_by == "prismyra"
        hook.record_route(
            tag,
            outcome,
            context="board",
            question="which way?",
            options=["up", "down"],
            kind="choice",
            engine_probabilities={"up": 0.2, "down": 0.8},
        )
        assert _wait_for(lambda: hook.stats()["logged"] == 1)
        rows = list((tmp_path / "exp" / TASK).glob("*.jsonl"))
        row = json.loads(rows[0].read_text().splitlines()[0])
        assert row["answered_by"] == "prismyra"
        assert row["probabilities"] == {"up": 0.2, "down": 0.8}
    finally:
        hook.stop()


def test_shadow_mode_logs_prismyras_answer_plus_the_students_as_comparison(tmp_path, fake_student):
    entry = {**ENTRY_AB, "router": {"mode": "shadow", "student_path": "students/snake.json"}}
    hook = make_hook(tmp_path, [entry], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        outcome = hook.route(tag, question="which way?", options=["up", "down"], kind="choice", context="board")
        assert outcome.served_by == "prismyra"
        assert outcome.student_probabilities == {"up": 0.9, "down": 0.1}
        hook.record_route(
            tag,
            outcome,
            context="board",
            question="which way?",
            options=["up", "down"],
            kind="choice",
            engine_probabilities={"up": 0.3, "down": 0.7},
        )
        assert _wait_for(lambda: hook.stats()["logged"] == 1)
        rows = list((tmp_path / "exp" / TASK).glob("*.jsonl"))
        row = json.loads(rows[0].read_text().splitlines()[0])
        assert row["answered_by"] == "prismyra"
        assert row["probabilities"] == {"up": 0.3, "down": 0.7}  # the shadow student never changes the answer
        assert row["comparison_probabilities"] == {"up": 0.9, "down": 0.1}
    finally:
        hook.stop()


def test_retain_context_none_drops_the_context_but_keeps_the_drift_summary(tmp_path, fake_student):
    entry = {**ENTRY_AB, "retain_context": "none"}
    hook = make_hook(tmp_path, [entry], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        outcome = hook.route(
            tag, question="which way?", options=["up", "down"], kind="choice", context="a secret board"
        )
        hook.record_route(
            tag,
            outcome,
            context="a secret board",
            question="which way?",
            options=["up", "down"],
            kind="choice",
            engine_probabilities=None,
        )
        assert _wait_for(lambda: hook.stats()["logged"] == 1)
        rows = list((tmp_path / "exp" / TASK).glob("*.jsonl"))
        row = json.loads(rows[0].read_text().splitlines()[0])
        assert row["context"] == ""
        assert row["feature_summary"] is not None
    finally:
        hook.stop()


def test_retain_context_features_only_stores_precomputed_features_instead_of_context(tmp_path, fake_student):
    entry = {**ENTRY_AB, "retain_context": "features_only"}
    hook = make_hook(tmp_path, [entry], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        outcome = hook.route(
            tag, question="which way?", options=["up", "down"], kind="choice", context="a secret board"
        )
        hook.record_route(
            tag,
            outcome,
            context="a secret board",
            question="which way?",
            options=["up", "down"],
            kind="choice",
            engine_probabilities=None,
        )
        assert _wait_for(lambda: hook.stats()["logged"] == 1)
        rows = list((tmp_path / "exp" / TASK).glob("*.jsonl"))
        row = json.loads(rows[0].read_text().splitlines()[0])
        assert row["context"] == ""
        assert row["precomputed_features"] == [1.0, 2.0, 3.0]
    finally:
        hook.stop()


def test_submit_feedback_round_trip(tmp_path):
    plain_entry = {
        "task": "phishing-v1",
        "question": "Is this spam?",
        "options": ["no", "yes"],
        "kind": "noul",
        "normalize": "none",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "e.jsonl",
    }
    hook = make_hook(tmp_path, [plain_entry], log_dir=tmp_path / "exp")
    try:
        with pytest.raises(ValueError, match="exactly one"):
            hook.submit_feedback("r1")
        with pytest.raises(ValueError, match="exactly one"):
            hook.submit_feedback("r1", label="yes", reward=1.0)
        assert hook.submit_feedback("r1", label="yes") is True
        assert _wait_for(lambda: hook.stats()["feedback"]["logged"] == 1)

        from prismyra.learn.experience import load_feedback

        joined = load_feedback(tmp_path / "exp" / "_feedback")
        assert joined["r1"].label == "yes"
    finally:
        hook.stop()


def test_reload_student_swaps_in_a_newly_loaded_handle(tmp_path, monkeypatch):
    from prismyra.learn.students import StudentRegistry  # see `fake_student`'s own note on why this is local

    first = FakeStudentHandle(version="v1", probs={"up": 0.6, "down": 0.4})
    second = FakeStudentHandle(version="v2", probs={"up": 0.1, "down": 0.9})
    state = {"current": first}

    def fake_load(self, task, spec_dir, student_path):
        self._handles[task] = state["current"]
        return state["current"]

    monkeypatch.setattr(StudentRegistry, "load", fake_load)
    hook = make_hook(tmp_path, [ENTRY_AB], log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="which way?", options=["up", "down"], kind="choice")
        outcome = hook.route(tag, question="which way?", options=["up", "down"], kind="choice", context="board")
        assert outcome.answer_probabilities == {"up": 0.6, "down": 0.4}

        state["current"] = second
        handle = hook.reload_student(TASK)
        assert handle.version == "v2"

        outcome = hook.route(tag, question="which way?", options=["up", "down"], kind="choice", context="board")
        assert outcome.answer_probabilities == {"up": 0.1, "down": 0.9}
    finally:
        hook.stop()


def test_reload_student_on_an_unrouted_tag_raises(tmp_path):
    plain_entry = {
        "task": "phishing-v1",
        "question": "Is this spam?",
        "options": ["no", "yes"],
        "kind": "noul",
        "normalize": "none",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "e.jsonl",
    }
    hook = make_hook(tmp_path, [plain_entry], log_dir=tmp_path / "exp")
    try:
        with pytest.raises(ValueError, match="no router"):
            hook.reload_student("phishing-v1")
    finally:
        hook.stop()
