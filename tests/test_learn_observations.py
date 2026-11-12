"""`prismyra.learn.observations`: graded attempts in the open observation record format.

Written against the module's contract: rows carry the assigned arm, the components that answered in `path`, and an
`arm_version` naming what was serving; a row is graded only against a label the operator supplied; a scalar reward
becomes a verdict only through an injected function; a row from another student artifact is refused; item ids are
content digests unless the items file names them.
"""

from __future__ import annotations

import dataclasses
import json
import types

import pytest

from prismyra.learn import observations as obsmod
from prismyra.learn.experience import Experience, FeedbackRecord

OPTIONS = ("negative", "positive")


def _exp(context, p_pos, *, answered_by="prismyra", request_id="", route_mode=None, comparison=None,
         params="p1", ts=1_700_000_000.0):
    return Experience(tag="t", spec_version="s", ts=ts, context=context, question="q", options=OPTIONS, kind="noul",
                      probabilities={"negative": 1 - p_pos, "positive": p_pos}, answered_by=answered_by,
                      versions={"package": "0.4", "backbone": "qwen", "spec": "s"}, request_id=request_id,
                      route_mode=route_mode, route_params_version=params, comparison_probabilities=comparison)


class FakeArtifact:
    """Just what `artifact_version` and `score_items` read."""

    features = None
    spec_version, backend, model_text = "s", "lightgbm", "m"

    def predict_proba(self, context, question=""):
        p = 0.9 if "good" in context else 0.2
        return {"negative": 1 - p, "positive": p}


def test_item_id_is_the_content_digest():
    import hashlib

    assert obsmod.item_id("héllo") == "sha256:" + hashlib.sha256("héllo".encode()).hexdigest()


def test_score_grades_against_the_items_label_and_names_the_version():
    art = FakeArtifact()
    rows = obsmod.score_items(art, [{"item": "a", "context": "good day", "label": "positive"},
                                    {"item": "b", "context": "bad day", "label": "positive"},
                                    {"item": "c", "context": "good"}],
                              task_type="sent", arm="student", label_source="dataset")
    assert [r["outcome"] for r in rows] == ["accepted", "rejected", "unobserved"]
    assert rows[2]["reason"] == "no label"
    assert all(r["format"] == 3 and r["arm_version"] == obsmod.artifact_version(art) for r in rows)
    assert rows[0]["score"] == pytest.approx(0.9)


def test_export_attributes_rows_to_the_assigned_arm_with_the_components_in_path():
    rows = obsmod.export_experience(
        [_exp("x", 0.9, answered_by="student:abc"),
         _exp("y", 0.4, route_mode="confidence", comparison={"negative": 0.5, "positive": 0.5}),
         _exp("z", 0.1)],
        task_type="sent", arm="router", labels={obsmod.item_id("x"): "positive", obsmod.item_id("y"): "positive",
                                                 obsmod.item_id("z"): "negative"})
    assert [r["arm"] for r in rows] == ["router"] * 3
    assert [r["path"] for r in rows] == [["student"], ["student", "prismyra"], ["prismyra"]]
    assert [r["outcome"] for r in rows] == ["accepted", "rejected", "accepted"]
    assert len({r["arm_version"] for r in rows}) == 1     # one serving snapshot, one version


def test_a_different_routing_snapshot_is_a_different_version():
    rows = obsmod.export_experience([_exp("x", 0.9, params="p1"), _exp("y", 0.9, params="p2")],
                                    task_type="sent", arm="router")
    assert rows[0]["arm_version"] != rows[1]["arm_version"]


def test_a_row_from_another_student_artifact_is_refused():
    from prismyra.learn.students import student_version

    art = FakeArtifact()
    mine = f"student:{student_version(art)}"
    obsmod.export_experience([_exp("x", 0.9, answered_by=mine)], task_type="sent", arm="r", student=art)
    with pytest.raises(obsmod.ObservationError, match="not the declared student"):
        obsmod.export_experience([_exp("x", 0.9, answered_by="student:other")], task_type="sent", arm="r",
                                 student=art)


def test_feedback_label_wins_and_a_reward_needs_a_verdict_function():
    fb = {"r1": FeedbackRecord(request_id="r1", ts=0.0, label="negative"),
          "r2": FeedbackRecord(request_id="r2", ts=0.0, reward=0.7)}
    exps = [_exp("x", 0.9, request_id="r1"), _exp("y", 0.9, request_id="r2")]
    rows = obsmod.export_experience(exps, task_type="sent", arm="r", feedback=fb)
    assert rows[0]["outcome"] == "rejected" and rows[0]["label_source"] == "feedback"
    assert rows[1]["outcome"] == "unobserved" and rows[1]["reason"] == "reward without a verdict module"
    graded = obsmod.export_experience(exps, task_type="sent", arm="r", feedback=fb,
                                      verdict=lambda reward, exp: "accepted" if reward > 0.5 else "rejected",
                                      verdict_name="mymod:fn")
    assert graded[1]["outcome"] == "accepted" and graded[1]["grader"] == "mymod:fn"


def test_a_row_without_retained_context_is_refused():
    with pytest.raises(obsmod.ObservationError, match="retain_context"):
        obsmod.export_experience([_exp("", 0.9)], task_type="sent", arm="r")


def test_rows_refuse_what_a_reader_would_refuse():
    with pytest.raises(obsmod.ObservationError):
        obsmod.row(item="a", task_type="t", arm="r", outcome="unobserved")
    with pytest.raises(obsmod.ObservationError):
        obsmod.row(item="", task_type="t", arm="r", outcome="accepted")


def test_export_cli_writes_a_manifest_and_says_when_drops_are_unknown(tmp_path, capsys):
    root = tmp_path / "log" / "t"
    root.mkdir(parents=True)
    (root / "2026-11-12.jsonl").write_text(json.dumps(_exp("x", 0.9).to_json()) + "\n")
    out = tmp_path / "obs.jsonl"
    assert obsmod.main(["export", "--log-root", str(tmp_path / "log"), "--task", "t", "--task-type", "sent",
                        "--arm", "router", "--out", str(out)]) == 0
    assert "logged/dropped unknown" in capsys.readouterr().out
    manifest = json.loads((tmp_path / "obs.jsonl.manifest.json").read_text())
    assert manifest["rows"] == 1 and manifest["dropped"] is None
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"logged": 1, "dropped": 0}))
    assert obsmod.main(["export", "--log-root", str(tmp_path / "log"), "--task", "t", "--task-type", "sent",
                        "--arm", "router", "--stats", str(stats), "--out", str(out)]) == 0
    assert json.loads((tmp_path / "obs.jsonl.manifest.json").read_text())["dropped"] == 0
