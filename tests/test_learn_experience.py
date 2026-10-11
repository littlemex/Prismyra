"""`prismyra.learn.experience`: the bounded queue drops rather than blocks, and retention actually deletes."""

from __future__ import annotations

import json
import time

from prismyra.learn.experience import Experience, LocalExperienceLog
from prismyra.learn.spec import load_spec

ENTRY = {
    "task": "phishing-v1",
    "question": "Is this email a phishing attempt?",
    "options": ["no", "yes"],
    "kind": "noul",
    "normalize": "strip+lower",
    "retain_days": 1,
    "keep_hidden": False,
    "eval_set": "evals/x.jsonl",
}


def make_spec(tmp_path, retain_days=1):
    path = tmp_path / "learn.json"
    path.write_text(json.dumps([{**ENTRY, "retain_days": retain_days}]))
    return load_spec(path)


def make_exp(spec, ts=None, context="an email"):
    return Experience(
        tag="phishing-v1",
        spec_version=spec.version,
        ts=ts if ts is not None else time.time(),
        context=context,
        question=ENTRY["question"],
        options=("no", "yes"),
        kind="noul",
        probabilities={"no": 0.1, "yes": 0.9},
        answered_by="prismyra",
        versions={"package": "0.0.0", "backbone": "stub", "spec": spec.version},
    )


def test_round_trip_json():
    e = Experience(
        tag="t",
        spec_version="v1",
        ts=1.0,
        context="c",
        question="q",
        options=("a", "b"),
        kind="boolean",
        probabilities={"a": 0.2, "b": 0.8},
        answered_by="prismyra",
        versions={"package": "1", "backbone": "m", "spec": "v1"},
        h=(0.1, 0.2),
    )
    assert Experience.from_json(e.to_json()) == e


def test_a_put_experience_is_written_and_counted(tmp_path):
    spec = make_spec(tmp_path)
    log = LocalExperienceLog(tmp_path / "exp", spec, max_queue=10).start()
    try:
        assert log.put(make_exp(spec)) is True
        for _ in range(50):
            if log.stats.logged == 1:
                break
            time.sleep(0.05)
        assert log.stats.logged == 1
        assert log.stats.dropped == 0
    finally:
        log.stop()

    files = list((tmp_path / "exp" / "phishing-v1").glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["tag"] == "phishing-v1"
    assert row["probabilities"] == {"no": 0.1, "yes": 0.9}


def test_a_full_queue_drops_and_counts_rather_than_blocking(tmp_path):
    spec = make_spec(tmp_path)
    log = LocalExperienceLog(tmp_path / "exp", spec, max_queue=1)
    # Never started: nothing drains the queue, so the second `put` must find it full.
    try:
        assert log.put(make_exp(spec)) is True
        assert log.put(make_exp(spec)) is False
        assert log.stats.dropped == 1
    finally:
        log._stop.set()  # no thread was started; nothing to join


def test_retention_deletes_files_older_than_retain_days(tmp_path):
    spec = make_spec(tmp_path, retain_days=1)
    log = LocalExperienceLog(tmp_path / "exp", spec, max_queue=10)
    tag_dir = tmp_path / "exp" / "phishing-v1"
    tag_dir.mkdir(parents=True)
    # Written directly to disk, not through `_write`: a file `_write` opened stays in `_open_files` and
    # `_maybe_purge` deliberately never deletes a file it might still be appending to (see that method's own
    # reasoning) -- an old, already-closed file, which is what retention is actually meant to clear out, is
    # exactly what going around `_write` reproduces here.
    old_date = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 5 * 86400))
    (tag_dir / f"{old_date}.jsonl").write_text(json.dumps(make_exp(spec).to_json()) + "\n")
    recent = make_exp(spec, ts=time.time())
    log._write(recent)

    assert len(list(tag_dir.glob("*.jsonl"))) == 2

    log._maybe_purge("phishing-v1", min_interval_s=0.0)

    remaining = list(tag_dir.glob("*.jsonl"))
    assert len(remaining) == 1
    assert time.strftime("%Y-%m-%d", time.gmtime(recent.ts)) == remaining[0].stem


def test_an_unknown_tag_is_not_purged(tmp_path):
    """A tag that is not (or no longer) in the spec is left alone -- there is no retention policy to apply, and
    guessing one would be worse than doing nothing."""
    spec = make_spec(tmp_path)
    log = LocalExperienceLog(tmp_path / "exp", spec, max_queue=10)
    (tmp_path / "exp" / "ghost-tag").mkdir(parents=True)
    old_file = tmp_path / "exp" / "ghost-tag" / "2000-01-01.jsonl"
    old_file.write_text("{}\n")
    log._maybe_purge("ghost-tag", min_interval_s=0.0)
    assert old_file.exists()
