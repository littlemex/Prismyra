"""`prismyra.learn.observations`: graded attempts in the open observation record format.

Written against the module's contract: item ids and label joins go by the content of ``[context, question]``; a row is
graded only against an operator-supplied label that is one of the options; ties and non-finite probabilities are not
answers; a scalar reward needs an injected, named verdict; export names the router arm, its components in `path`, and a
version that refuses student rows from an undeclared artifact; each logged request is its own `run`; bad rows are
counted, not fatal; the manifest says when drops are unknown.
"""

from __future__ import annotations

import json

import pytest

from prismyra.learn import observations as obsmod
from prismyra.learn.experience import Experience, FeedbackRecord

OPTIONS = ("negative", "positive")
KW = dict(task_type="sent", arm="router", student_arm="student", fallback_arm="prismyra")


def _exp(context, p_pos, *, answered_by="prismyra", request_id="r", route_mode=None, comparison=None,
         params="p1", ts=1_700_000_000.0, question="q", versions=None):
    return Experience(tag="t", spec_version="s", ts=ts, context=context, question=question, options=OPTIONS,
                      kind="noul", probabilities={"negative": 1 - p_pos, "positive": p_pos}, answered_by=answered_by,
                      versions=versions or {"package": "0.4", "backbone": "qwen", "spec": "s"},
                      request_id=request_id, route_mode=route_mode, route_params_version=params,
                      comparison_probabilities=comparison)


class FakeArtifact:
    features = None
    spec_version, backend, model_text = "s", "lightgbm", "m"

    def to_json(self):
        return {"tag": "t", "spec_version": self.spec_version, "options": list(OPTIONS), "kind": "noul", "dims": 1,
                "idf": [], "model_text": self.model_text, "features": None, "feature_names": None,
                "backend": self.backend, "backend_version": None, "text_tokenizer": "words"}

    def predict_proba(self, context, question=""):
        p = 0.9 if "good" in context else 0.5 if "tie" in context else 0.2
        return {"negative": 1 - p, "positive": p}


def labels(tmp_path, rows):
    p = tmp_path / "items.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return obsmod.load_items(p)


def test_item_ids_cover_context_and_question():
    assert obsmod.item_id("ctx", "Food good?") != obsmod.item_id("ctx", "Service good?")
    assert obsmod.item_id("ctx").startswith("sha256:") and len(obsmod.item_id("ctx")) == 71


def test_score_grades_against_labels_and_refuses_a_label_outside_the_options(tmp_path):
    items = labels(tmp_path, [{"item": "a", "context": "good day", "label": "positive"},
                              {"item": "b", "context": "bad day", "label": "positive"},
                              {"item": "c", "context": "tie"}, {"item": "d", "context": "tie", "question": "x",
                                                                "label": "negative"}])
    rows = obsmod.score_items(FakeArtifact(), items, task_type="sent", arm="student", label_source="dataset")
    assert [r["outcome"] for r in rows] == ["accepted", "rejected", "unobserved", "unobserved"]
    assert rows[2]["reason"] == "no label" and rows[3]["reason"] == "tied top probabilities"
    from prismyra.learn.students import artifact_version
    assert all(r["arm_version"] == artifact_version(FakeArtifact()) for r in rows)
    bad = labels(tmp_path, [{"item": "a", "context": "good", "label": "Positive"}])
    with pytest.raises(obsmod.ObservationError, match="not one of the options"):
        obsmod.score_items(FakeArtifact(), bad, task_type="sent", arm="student", label_source="dataset")


def test_items_file_refusals(tmp_path):
    with pytest.raises(obsmod.ObservationError, match="twice"):
        labels(tmp_path, [{"item": "a", "context": "x"}, {"item": "a", "context": "y"}])
    with pytest.raises(obsmod.ObservationError, match="different label"):
        labels(tmp_path, [{"context": "x", "label": "positive"}, {"item": "z", "context": "x", "label": "negative"}])


def test_export_joins_named_labels_by_content_and_names_components(tmp_path):
    lab = labels(tmp_path, [{"item": "q1", "context": "x", "question": "q", "label": "positive"},
                            {"item": "q2", "context": "y", "question": "q", "label": "positive"},
                            {"item": "q3", "context": "z", "question": "q", "label": "negative"}])
    from prismyra.learn.students import artifact_version, student_version
    mine = f"student:{student_version(FakeArtifact())}"
    rows, counts = obsmod.export_experience(
        [_exp("x", 0.9, answered_by=mine, request_id="r1"),
         _exp("y", 0.4, route_mode="confidence", comparison={"negative": 0.5, "positive": 0.5}, request_id="r2"),
         _exp("z", 0.1, request_id="r3")],
        student=FakeArtifact(), labels=lab, label_source="dataset", **KW)
    assert [r["item"] for r in rows] == ["q1", "q2", "q3"]
    assert [r["path"] for r in rows] == [["student"], ["student", "prismyra"], ["prismyra"]]
    assert [r["outcome"] for r in rows] == ["accepted", "rejected", "accepted"]
    assert [r["run"] for r in rows] == ["r1", "r2", "r3"]
    assert len({r["arm_version"] for r in rows}) == 1 and artifact_version(FakeArtifact()) in rows[0]["arm_version"]


def test_per_component_rows_carry_each_components_own_version(tmp_path):
    """The versions an offline score of the same parts carries, so a cascade's certificate meets served traffic."""
    from prismyra.learn.students import artifact_version, student_version
    lab = labels(tmp_path, [{"item": "q1", "context": "x", "question": "q", "label": "positive"},
                            {"item": "q2", "context": "y", "question": "q", "label": "negative"},
                            {"item": "q3", "context": "z", "question": "q", "label": "negative"}])
    mine = f"student:{student_version(FakeArtifact())}"
    rows, _ = obsmod.export_experience(
        [_exp("x", 0.9, answered_by=mine, request_id="r1"),
         _exp("y", 0.3, route_mode="confidence", comparison={"negative": 0.4, "positive": 0.6}, request_id="r2"),
         _exp("z", 0.1, request_id="r3")],
        student=FakeArtifact(), labels=lab, label_source="dataset", per_component=True, **KW)
    got = [(r["item"], r["arm"], r["arm_version"], r["outcome"]) for r in rows]
    fallback = "backbone:qwen|package:0.4|spec:s"
    assert got == [("q1", "student", artifact_version(FakeArtifact()), "accepted"),
                   ("q2", "student", artifact_version(FakeArtifact()), "rejected"),
                   ("q2", "prismyra", fallback, "accepted"), ("q3", "prismyra", fallback, "accepted")]
    assert all("path" not in r and "router" != r["arm"] for r in rows)
    with pytest.raises(obsmod.ObservationError, match="no --student"):
        obsmod.export_experience([_exp("y", 0.3, route_mode="confidence", comparison={"negative": 0.4,
                                                                                      "positive": 0.6})],
                                 per_component=True, **KW)


def test_student_rows_need_the_declared_artifact():
    with pytest.raises(obsmod.ObservationError, match="no --student"):
        obsmod.export_experience([_exp("x", 0.9, answered_by="student:abc")], **KW)
    with pytest.raises(obsmod.ObservationError, match="not the declared student"):
        obsmod.export_experience([_exp("x", 0.9, answered_by="student:abc")], student=FakeArtifact(), **KW)


def test_a_snapshot_window_separates_two_routing_tables():
    exps = [_exp("x", 0.9, params="p1", ts=10.0), _exp("y", 0.9, params="p2", ts=20.0)]
    rows, counts = obsmod.export_experience(exps, since=15.0, **KW)
    assert len(rows) == 1 and counts["outside_window"] == 1 and "route:p2" in rows[0]["arm_version"]


def test_bad_rows_are_counted_not_fatal():
    rows, counts = obsmod.export_experience(
        [_exp("", 0.9), _exp("x", 0.9, versions={"package": "0.4", "backbone": "", "spec": "s"}), _exp("y", 0.9)],
        **KW)
    assert len(rows) == 1
    assert counts["skipped: context not retained"] == 1
    assert counts["skipped: versions missing ['backbone']"] == 1


def test_feedback_and_labels_that_disagree_are_unobserved_and_counted(tmp_path):
    lab = labels(tmp_path, [{"context": "x", "question": "q", "label": "positive"}])
    fb = {"r1": FeedbackRecord(request_id="r1", ts=0.0, label="negative"),
          "zz": FeedbackRecord(request_id="zz", ts=0.0, label="negative")}
    rows, counts = obsmod.export_experience([_exp("x", 0.9, request_id="r1")], feedback=fb, labels=lab,
                                            label_source="dataset", **KW)
    assert rows[0]["outcome"] == "unobserved" and counts["unobserved: feedback and labels disagree"] == 1
    assert counts["feedback not joined to any exported row"] == 1


def test_a_reward_needs_a_named_verdict_function():
    fb = {"r2": FeedbackRecord(request_id="r2", ts=0.0, reward=0.7)}
    exps = [_exp("y", 0.9, request_id="r2")]
    rows, _ = obsmod.export_experience(exps, feedback=fb, **KW)
    assert rows[0]["reason"] == "reward without a verdict module"
    with pytest.raises(obsmod.ObservationError, match="needs a name"):
        obsmod.export_experience(exps, feedback=fb, verdict=lambda r, e: "accepted", **KW)
    rows, _ = obsmod.export_experience(exps, feedback=fb, verdict=lambda r, e: "accepted" if r > 0.5 else "rejected",
                                       verdict_name="mymod:fn", **KW)
    assert rows[0]["outcome"] == "accepted" and rows[0]["grader"] == "mymod:fn"
    assert rows[0]["label_source"] == "feedback-reward"


def test_rows_refuse_what_a_reader_would_refuse():
    base = dict(item="a", task_type="t", arm="r", run="1", arm_version="v")
    with pytest.raises(obsmod.ObservationError):
        obsmod.row(outcome="unobserved", **base)
    with pytest.raises(obsmod.ObservationError):
        obsmod.row(outcome="accepted", score=float("nan"), **base)
    with pytest.raises(obsmod.ObservationError):
        obsmod.row(outcome="accepted", **{**base, "arm_version": ""})


def test_export_cli_reads_stats_first_and_says_when_drops_are_unknown(tmp_path, capsys):
    root = tmp_path / "log" / "t"
    root.mkdir(parents=True)
    (root / "2026-11-12.jsonl").write_text(json.dumps(_exp("x", 0.9).to_json()) + "\n")
    out = tmp_path / "obs.jsonl"
    argv = ["export", "--log-root", str(tmp_path / "log"), "--task", "t", "--task-type", "sent", "--arm", "router",
            "--student-arm", "student", "--fallback-arm", "prismyra", "--out", str(out)]
    assert obsmod.main(argv) == 0
    assert "logged/dropped unknown" in capsys.readouterr().out
    assert json.loads((tmp_path / "obs.jsonl.manifest.json").read_text())["dropped"] == "unknown"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tags": {}}))
    before = out.read_text()
    assert obsmod.main(argv + ["--stats", str(bad)]) == 2
    assert out.read_text() == before
    good = tmp_path / "stats.json"
    good.write_text(json.dumps({"logged": 1, "dropped": 2}))
    assert obsmod.main(argv + ["--stats", str(good)]) == 0
    assert "dropped 2 rows" in capsys.readouterr().out


def test_one_run_name_lets_served_rows_pair_with_an_offline_score():
    rows, _ = obsmod.export_experience([_exp("x", 0.9, request_id="r1"), _exp("y", 0.9, request_id="r2")],
                                       per_component=True, run="score", **KW)
    assert [r["run"] for r in rows] == ["score", "score"]
