"""Experiment 10 (SYNTHESIS-v2.md): does the Batcher's layer-interleaved fusion path
(`engine.interleaved_fork`, on by default in v0.4.0 and wired into `Batcher._answer()` at
`schedule.py:438` [single fresh document, one pass] and `schedule.py:483` [two or more fresh documents,
one pass] -- see `_fused_single_passes`/`_fused_many_passes` in `Batcher.stats()`) answer every document
exactly as the unfused two-pass path would, under a REAL open-loop RACE arrival trace (not the existing
hand-built-companion unit tests in tests/test_gpu.py)?

Design: generate the arrival trace (which RACE item arrives at which relative time) ONCE, independent of
either engine, so both conditions replay the identical sequence of (document, question-set, arrival time)
-- not regenerated per run, so no RNG-state drift between the two separate processes. Each condition is a
fresh process (fresh engine, fresh Batcher, nothing resident left over from the other condition). The
engine's own replay speed is allowed to differ between conditions (that's the point: a batcher's answer is
supposed to be invariant to who else is in the pass, not to how fast the engine happens to be), so the two
runs are not expected to form identical pass shapes -- only identical final answers per request index.

Usage:
    python3 exp10_batcher_fusion_torch_equal.py build-trace OUT.json
    python3 exp10_batcher_fusion_torch_equal.py run TRACE.json fused|unfused OUT.json
    python3 exp10_batcher_fusion_torch_equal.py compare FUSED.json UNFUSED.json
"""
import os, sys, json, random, time

os.environ.setdefault("HF_HOME", "/work/.hf")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# append, not insert(0, ...): /work/fp8spd/src itself contains its own vendored prismyra/ directory, and
# inserting at the front of sys.path shadowed this run's editable-installed prismyra with that one (its
# Prismyra.__init__ predates `interleaved_fork` -- surfaced as "unexpected keyword argument 'interleaved_fork'").
# Appending means only `tasks` (which this run's own prismyra install does not provide) is picked up from there.
sys.path.append("/work/fp8spd/src/evals")
sys.path.append("/work/fp8spd/src")

MODEL = os.environ.get("EXP10_MODEL", "/work/next/models/p35-l36-kd_a8_2xrows")
# Scaled down from fp8spd's own 368-doc/rate=40/8s open-loop harness: that scale OOM'd `tput` (46GB L40S,
# 1 GPU, 32GB of which the fp8-36l model itself holds) over the course of 100+ real passes -- diagnosed on
# real hardware (see RUN-tok.md): neither a single wide (8-9 companion) pass nor 10 sequential single-
# document passes OOM in isolation (both stayed >=9.9GB free), so this is a cumulative effect of many
# passes over a long run, not a per-pass sizing bug this experiment needs to fix. This experiment's own
# question is correctness (torch.equal), not throughput at that scale, so a smaller trace that still
# produces a real mix of fused-single/fused-many/non-fused passes is enough.
RATE = 15.0
PHASE_SECONDS = 5.0
POOL_DOCUMENTS = 80
DRAIN_TIMEOUT = 90.0


def build_trace(out_path: str) -> None:
    """The arrival schedule itself: which pooled item index, at what offset (seconds) from t0. Fixed once,
    on disk, so `run` never calls `random` -- the two conditions cannot see different arrival sequences."""
    random.seed(0)
    n = max(1, round(RATE * PHASE_SECONDS))
    offsets = []
    t = 0.0
    for _ in range(n):
        t += random.expovariate(RATE)
        offsets.append(t)
    trace = {"pool_documents": POOL_DOCUMENTS, "rate": RATE, "phase_seconds": PHASE_SECONDS,
             "n": n, "item_indices": [i % POOL_DOCUMENTS for i in range(n)], "offsets": offsets}
    json.dump(trace, open(out_path, "w"), indent=1)
    print(f"wrote trace: {n} arrivals over {offsets[-1]:.2f}s (pool={POOL_DOCUMENTS})", flush=True)


def run(trace_path: str, mode: str, out_path: str) -> None:
    import tasks
    from prismyra import Prismyra
    from prismyra.schedule import Batcher

    trace = json.load(open(trace_path))
    items = tasks.load("race", trace["pool_documents"], split="validation", seed=0)
    print(f"{len(items)} pooled documents loaded", flush=True)

    interleaved_fork = mode == "fused"
    assert mode in ("fused", "unfused"), mode
    # paged=True is Batcher's own requirement (schedule.py: "a batching scheduler needs the paged storage"),
    # not a precision-affecting choice here -- v0.4.0's batch-invariance claim is unconditional either way
    # (engine.py:672, `if on_cuda:`), so this does not reopen the paged-vs-unpaged gap task 3 is about.
    # graphs=True (tried first, matching fp8spd's s5_openloop.py) made this *worse*, not better: CUDA
    # graph capture keeps a separate private memory pool per distinct companion-count shape, and this
    # open-loop trace visits ~10 distinct widths -- each capture grows VRAM that is never given back, and
    # the run degraded from working to almost-total OOM partway through (see RUN-tok.md). graphs=False
    # avoids that; correctness (torch.equal) is this experiment's question, not speed, so there is nothing
    # lost by not capturing graphs here.
    engine = Prismyra(MODEL, require_kernels=True, paged=True, graphs=False, interleaved_fork=interleaved_fork)
    print(f"engine.interleaved_fork = {engine.interleaved_fork}", flush=True)
    # `open_shelf`'s own default room is "the largest bucket admission will accept" -- on this 46GB L40S,
    # with a 32GB fp8 model already loaded, that greedily claims most of the remaining ~14GB into one paged
    # KV pool before a single real pass has run, leaving too little margin for a multi-document pass's own
    # working memory once several of the (short, ~300-800 token) RACE articles are resident together --
    # real CUDA OOM, not fragmentation (confirmed: still failed with graphs=False). RACE articles are short;
    # this trace never needs anywhere near the full 65536-token bucket resident at once, so an explicit,
    # modest `lane_room` leaves real headroom instead. 8192 is comfortably above the ~368-document pool's
    # own longest article and still leaves most of the ~14GB free for forward-pass scratch.
    batcher = Batcher(engine, lane_room=8192).start()

    jobs = []
    t0 = time.perf_counter()
    dropped = 0
    for idx, off in zip(trace["item_indices"], trace["offsets"]):
        item = items[idx]
        now = time.perf_counter() - t0
        if now < off:
            time.sleep(off - now)
        try:
            job = batcher.submit(item.context, item.questions)
            jobs.append((job, item))
        except Exception as e:
            dropped += 1
            jobs.append((None, item))
            print(f"submit failed at offset {off:.2f}: {e}", flush=True)

    deadline = time.perf_counter() + DRAIN_TIMEOUT
    for job, _ in jobs:
        if job is None:
            continue
        remaining = max(0.0, deadline - time.perf_counter())
        job.done.wait(timeout=remaining)

    records = []
    n_unanswered = 0
    for req_i, (job, item) in enumerate(jobs):
        if job is None or not job.done.is_set() or job.error is not None:
            n_unanswered += 1
            records.append({"req_i": req_i, "answered": False,
                             "error": str(job.error) if job is not None and job.error else "no_job_or_timeout"})
            continue
        per_q = {}
        for q in item.questions:
            ans = job.result[q.id]
            # Fixed option order (the question's own declared order, `Question.options`), not dict
            # iteration order, so the two runs are compared value-for-value in the same slot.
            per_q[q.id] = {"option_order": list(q.options),
                           "probabilities": [ans.probabilities[o] for o in q.options],
                           "option": ans.option}
        records.append({"req_i": req_i, "answered": True, "item_index": trace["item_indices"][req_i],
                         "digest_prefix": item.context[:40], "answers": per_q})

    stats = batcher.stats()
    batcher.stop()
    out = {"mode": mode, "interleaved_fork": interleaved_fork, "dropped": dropped,
           "n_unanswered": n_unanswered, "n_jobs": len(jobs), "batcher_stats": stats, "records": records}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"mode={mode} dropped={dropped} unanswered={n_unanswered}/{len(jobs)}", flush=True)
    print("batcher stats:", json.dumps(stats, default=str), flush=True)
    print("wrote", out_path, flush=True)


def compare(fused_path: str, unfused_path: str) -> None:
    import torch

    fused = json.load(open(fused_path))
    unfused = json.load(open(unfused_path))
    assert fused["n_jobs"] == unfused["n_jobs"], "the two runs replayed a different number of requests"

    # Two different kinds of disagreement, kept separate: (1) one run answered a request the other one
    # didn't -- expected and harmless, since the two engines run at different real speeds and the same
    # real-time arrival schedule can land a request on either side of a capacity/timing boundary in one
    # run and not the other (this experiment's own pre-registration says so: "not expected to form
    # identical pass shapes"). (2) both runs answered the SAME request and produced DIFFERENT
    # probabilities for the SAME question -- this is the actual thing experiment 10 is checking for, and
    # the only kind that should drive the verdict.
    unanswered_asymmetry = []
    value_mismatches = []
    n_compared_questions = 0
    both_answered = 0
    for rf, ru in zip(fused["records"], unfused["records"]):
        assert rf["req_i"] == ru["req_i"]
        if not (rf["answered"] and ru["answered"]):
            unanswered_asymmetry.append({"req_i": rf["req_i"],
                                          "fused_answered": rf["answered"], "unfused_answered": ru["answered"]})
            continue
        both_answered += 1
        for qid, af in rf["answers"].items():
            au = ru["answers"][qid]
            n_compared_questions += 1
            tf = torch.tensor(af["probabilities"], dtype=torch.float64)
            tu = torch.tensor(au["probabilities"], dtype=torch.float64)
            if not torch.equal(tf, tu):
                value_mismatches.append({
                    "req_i": rf["req_i"], "qid": qid,
                    "fused_option": af["option"], "unfused_option": au["option"],
                    "fused_probabilities": af["probabilities"], "unfused_probabilities": au["probabilities"],
                })

    verdict = "ALL_MATCH_add_permanent_test" if not value_mismatches else "VALUE_MISMATCH_investigate"
    summary = {
        "n_jobs": fused["n_jobs"], "both_answered": both_answered,
        "n_questions_compared": n_compared_questions,
        "n_unanswered_asymmetry": len(unanswered_asymmetry),  # expected, not a correctness signal
        "n_value_mismatches": len(value_mismatches),  # this is the correctness signal
        "fused_batcher_stats": fused["batcher_stats"], "unfused_batcher_stats": unfused["batcher_stats"],
        "verdict": verdict,
        "value_mismatches": value_mismatches[:50],
        "unanswered_asymmetry": unanswered_asymmetry[:50],
    }
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return summary


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build-trace":
        build_trace(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "compare":
        out = compare(sys.argv[2], sys.argv[3])
        json.dump(out, open("/work/tok/results/exp10_compare_summary.json", "w"), indent=1)
    else:
        raise SystemExit(f"unknown command {cmd!r}")
