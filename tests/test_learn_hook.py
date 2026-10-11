"""`prismyra.learn.hook.LearnHook`: the one object `prismyra.server` touches, exercised end to end without a
device or a server."""

from __future__ import annotations

import json
import time

from prismyra.learn.hook import LearnHook

ENTRY = {
    "task": "phishing-v1",
    "question": "Is this email a phishing attempt?",
    "options": ["no", "yes"],
    "kind": "noul",
    "normalize": "strip+lower",
    "retain_days": 30,
    "keep_hidden": True,
    "eval_set": "evals/x.jsonl",
}


def make_hook(tmp_path, **kwargs):
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps([ENTRY]))
    return LearnHook(spec_path, package_version="9.9.9", backbone="stub/model", **kwargs)


def _wait_for(predicate, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_tag_matches_and_versions_travel_with_the_experience(tmp_path):
    hook = make_hook(tmp_path, log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="Is this email a phishing attempt?", options=["no", "yes"], kind="noul")
        assert tag is not None
        hook.submit(
            tag,
            context="buy now!!!",
            question="Is this email a phishing attempt?",
            options=["no", "yes"],
            kind="noul",
            probabilities={"no": 0.2, "yes": 0.8},
            answered_by="prismyra",
        )
        assert _wait_for(lambda: hook.stats()["logged"] == 1)
    finally:
        hook.stop()

    rows = list((tmp_path / "exp" / "phishing-v1").glob("*.jsonl"))
    assert len(rows) == 1
    row = json.loads(rows[0].read_text().strip())
    assert row["versions"] == {"package": "9.9.9", "backbone": "stub/model", "spec": hook.spec.version}
    assert row["h"] is None  # not submitted with h, even though keep_hidden is true for this tag


def test_an_untagged_request_is_never_submitted(tmp_path):
    hook = make_hook(tmp_path, log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question="totally different question", options=["no", "yes"], kind="noul")
        assert tag is None
        assert not (tmp_path / "exp").exists() or not list((tmp_path / "exp").glob("**/*.jsonl"))
    finally:
        hook.stop()


def test_h_is_dropped_unless_the_tag_asks_to_keep_it(tmp_path):
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps([{**ENTRY, "keep_hidden": False}]))
    hook = LearnHook(spec_path, package_version="1", backbone="m", log_dir=tmp_path / "exp")
    try:
        tag = hook.tag(task=None, question=ENTRY["question"], options=["no", "yes"], kind="noul")
        hook.submit(
            tag,
            context="x",
            question=ENTRY["question"],
            options=["no", "yes"],
            kind="noul",
            probabilities={"no": 0.5, "yes": 0.5},
            answered_by="prismyra",
            h=(1.0, 2.0, 3.0),
        )
        assert _wait_for(lambda: hook.stats()["logged"] == 1)
    finally:
        hook.stop()
    rows = list((tmp_path / "exp" / "phishing-v1").glob("*.jsonl"))
    row = json.loads(rows[0].read_text().strip())
    assert row["h"] is None


def test_stats_reports_queue_depth_and_dropped(tmp_path):
    hook = make_hook(tmp_path, log_dir=tmp_path / "exp", max_queue=10_000)
    try:
        stats = hook.stats()
        assert set(stats) == {"logged", "dropped", "queue_depth"}
    finally:
        hook.stop()


def test_max_queue_is_actually_honored_by_the_underlying_queue(tmp_path):
    """Regression: `LearnHook.__init__` used to accept `max_queue` and never pass it to
    `LocalExperienceLog`, so every hook silently queued at the default (10,000) no matter what
    `--learn-max-queue` said. Caught on real hardware (Tokyo L40S) when a deliberately-stalled
    drain thread queued 500 experiences with `max_queue=1` and dropped none."""
    hook = make_hook(tmp_path, log_dir=tmp_path / "exp", max_queue=1)
    hook.log.stop()  # stop the drain thread so the queue cannot empty between submits
    tag = hook.tag(task=None, question=ENTRY["question"], options=["no", "yes"], kind="noul")
    for _ in range(5):
        hook.submit(
            tag,
            context="x",
            question=ENTRY["question"],
            options=["no", "yes"],
            kind="noul",
            probabilities={"no": 0.5, "yes": 0.5},
            answered_by="prismyra",
        )
    stats = hook.stats()
    assert stats["queue_depth"] == 1
    assert stats["dropped"] == 4
