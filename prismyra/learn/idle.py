"""Stage 5: fitting and validating a student during the server's own idle time, automatically, in a process
that is not the one holding the GPU. DISTILL-RL-DESIGN-v2.md section 5:

    作るときは、独立した段階として作る。空き時間の仕事が、録画の数え方や、文書の置き場所の並び (LRU) を
    乱さないことを、先に試験で確かめる。

This module is a standalone console script (``prismyra-learn-idle``), not a thread inside
``prismyra.server``, for exactly the reason section 5 asks for: the serving process's own queue, batcher and
paged cache are a closed system this package has measured carefully (`docs/PERFORMANCE.md`), and the
cheapest way to guarantee a background job cannot perturb any of that accounting is for the background job
to not share a process, a thread, or a lock with it at all. The only channel between the two is HTTP --
``GET /stats`` to read load, ``POST /v1/learn/reload`` to hand over a freshly fitted, already-admitted
student -- exactly the two endpoints an operator could drive by hand. Nothing here imports
``prismyra.server``, ``prismyra.engine`` or ``prismyra.schedule``.

**No new dependency.** Every import below is the standard library (``urllib``, ``json``, ``os``, ``time``)
plus ``prismyra.learn.fit``/``.students``/``.spec``, which already need the ``learn`` extra
(LightGBM) for a reason that has nothing to do with this module. Lowering this process's own scheduling
priority (``os.nice``) and bounding how many CPU threads one fit is allowed (``fit.run``'s own
``num_threads``) are both already-standard-library or already-plumbed knobs, not new machinery.

**What "idle" means here.** `is_busy` reads `prismyra.server`'s own `/stats` -- the admission queue's depth
(`Worker.stats()`'s `"depth"`) and, when `--batcher` is on, the batcher's own queue depth -- and calls the
server busy if either is above a configurable floor (default: any waiting request at all). This is
deliberately conservative: a GPU that is merely *not currently saturated* is not the same claim as "safe to
also run CPU-only LightGBM training on this box without a caller noticing", and the second claim is also
not what this checks -- CPU contention from a background fit is bounded by `niceness`/`num_threads`, not by
polling `/stats` more cleverly, which is why both knobs exist together rather than either alone.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .spec import LearnSpec, TagSpec, load_spec

#: The BLAS backends `fit.run`'s own numpy usage can be built against. Not LightGBM's own thread pool
#: (`--num-threads`, read directly by `fit.run` as its own `num_threads` parameter) -- these are a *second*,
#: independent source of threads, picked by numpy's BLAS backend (OpenBLAS/MKL) the first time it
#: initialises, before `fit.run` is ever called and regardless of what it is told. Measured on real
#: hardware: with only `--num-threads 1` set and these left at their OS default (every core on the box),
#: one fit call drove roughly half of a 48-core machine's cores above 20% utilisation at once -- exactly
#: the CPU contention `cap_blas_threads` exists to prevent.
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cap_blas_threads(num_threads: int) -> None:
    """Must run before `numpy` (and so before `lightgbm`, and before `fit.run`) is imported anywhere in this
    process -- a BLAS backend reads these once, at its own import time, and setting them afterwards has no
    effect. `main` calls this immediately after parsing arguments, before any of its own `import lightgbm`/
    `import prismyra` lines. `os.environ.setdefault`, not a flat assignment: an operator who has already set
    one of these for their own reasons keeps that choice.
    """
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ.setdefault(var, str(num_threads))


class IdleError(Exception):
    """A configuration problem found before any polling started -- a bad spec, a tag with no router, an
    unreachable server at startup. Not raised once the poll loop is running: a server that goes briefly
    unreachable mid-loop is logged and retried, never a reason to exit (`run_forever`'s own docstring)."""


def http_get_json(url: str, timeout: float = 5.0) -> dict | None:
    """``GET`` ``url``, parsed as JSON, or `None` on any network or parse failure -- a transient failure to
    reach the server is "assume busy, try again later", never an exception that would stop the loop.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def http_post_json(url: str, payload: dict, timeout: float = 5.0) -> dict | None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def is_busy(stats: dict, *, max_queue_depth: int = 0) -> bool:
    """`max_queue_depth` is a count of *waiting* requests this call tolerates before calling the server
    busy, not a latency target -- `/stats`'s own `"queue"."depth"` and (when present) `"batcher"."queue_depth"`
    are the two places a request waits before a kernel ever runs for it (`docs/PERFORMANCE.md`'s own
    distinction between queue time and device time). The default, 0, means "idle" is "nothing is waiting at
    all", the most conservative reading available without guessing at a latency number this module has no
    way to calibrate against a caller's own traffic.
    """
    depth = stats.get("queue", {}).get("depth", 0)
    batcher = stats.get("batcher")
    if batcher is not None:
        depth = max(depth, batcher.get("queue_depth", 0))
    return depth > max_queue_depth


@dataclass
class TagIdleState:
    last_fit_ts: float = 0.0
    last_fit_row_count: int = 0
    last_admitted: bool | None = None

    def to_json(self) -> dict:
        return {
            "last_fit_ts": self.last_fit_ts,
            "last_fit_row_count": self.last_fit_row_count,
            "last_admitted": self.last_admitted,
        }

    @staticmethod
    def from_json(raw: dict) -> TagIdleState:
        return TagIdleState(
            last_fit_ts=raw.get("last_fit_ts", 0.0),
            last_fit_row_count=raw.get("last_fit_row_count", 0),
            last_admitted=raw.get("last_admitted"),
        )


@dataclass
class IdleConfig:
    server_url: str
    learn_spec: Path
    log_dir: Path | None
    backbone: str
    package_version: str
    state_path: Path
    poll_interval_s: float = 30.0
    idle_polls_required: int = 3
    max_queue_depth: int = 0
    retrain_interval_s: float = 3600.0
    min_new_rows: int = 50
    num_threads: int = 1
    nice: int = 19
    tasks: tuple[str, ...] = ()  # empty: every routed tag


def _count_experience_rows(log_dir: Path, task: str) -> int:
    tag_dir = log_dir / task
    if not tag_dir.is_dir():
        return 0
    total = 0
    for path in tag_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            total += sum(1 for line in fh if line.strip())
    return total


def _routed_tags(spec: LearnSpec, tasks: tuple[str, ...]) -> list[TagSpec]:
    tags = [t for t in spec.tags if t.router is not None]
    if tasks:
        tags = [t for t in tags if t.task in tasks]
    return tags


def _due_for_refit(tag: TagSpec, state: TagIdleState, row_count: int, cfg: IdleConfig, now: float) -> bool:
    if now - state.last_fit_ts < cfg.retrain_interval_s:
        return False
    return row_count - state.last_fit_row_count >= cfg.min_new_rows


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".student-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_state(path: Path) -> dict[str, TagIdleState]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {task: TagIdleState.from_json(row) for task, row in raw.items()}


def save_state(path: Path, state: dict[str, TagIdleState]) -> None:
    _write_atomically(path, json.dumps({task: st.to_json() for task, st in state.items()}))


@dataclass
class RefitReport:
    task: str
    attempted: bool
    admitted: bool | None = None
    reasons: tuple[str, ...] = ()
    reloaded: bool = False
    error: str | None = None


def refit_one_tag(
    spec_path: Path, tag: TagSpec, cfg: IdleConfig, state: dict[str, TagIdleState], now: float
) -> RefitReport:
    """Fits ``tag``'s student (`prismyra.learn.fit.run`), and -- only if the result is admitted -- writes it
    atomically over the exact path its own `prismyra.learn.spec.RouterConfig.student_path` names and asks the
    live server to pick it up (`POST /v1/learn/reload`). A fit that is not admitted changes nothing already
    serving: the previous, still-admitted artifact keeps answering until a later fit does better, which is
    the same "never degrade what is already live" property `prismyra.learn.students.StudentHandle`'s own
    admission check gives the *first* load of a student.
    """
    from . import fit as fit_mod

    assert tag.router is not None
    log_dir = cfg.log_dir if cfg.log_dir is not None else spec_path.parent / "experience"
    feedback_dir = log_dir / "_feedback"
    try:
        artifact = fit_mod.run(
            spec_path=spec_path,
            task=tag.task,
            log_dir=log_dir,
            package_version=cfg.package_version,
            backbone=cfg.backbone,
            num_threads=cfg.num_threads,
            feedback_dir=feedback_dir if feedback_dir.is_dir() else None,
        )
    except fit_mod.FitError as e:
        return RefitReport(task=tag.task, attempted=True, error=str(e))

    row_count = _count_experience_rows(log_dir, tag.task)
    state[tag.task] = TagIdleState(last_fit_ts=now, last_fit_row_count=row_count, last_admitted=artifact.admitted)
    if not artifact.admitted:
        return RefitReport(task=tag.task, attempted=True, admitted=False, reasons=artifact.reasons)

    student_path = Path(tag.router.student_path)
    if not student_path.is_absolute():
        student_path = spec_path.parent / student_path
    _write_atomically(student_path, json.dumps(artifact.to_json()))

    reload_url = cfg.server_url.rstrip("/") + "/v1/learn/reload"
    reply = http_post_json(reload_url, {"task": tag.task})
    return RefitReport(
        task=tag.task, attempted=True, admitted=True, reasons=artifact.reasons, reloaded=reply is not None
    )


def run_forever(
    cfg: IdleConfig, *, stop_after: int | None = None, sleep=time.sleep, now=time.time
) -> list[RefitReport]:
    """The poll loop. ``stop_after`` (number of polls; `None` means forever) and the injectable ``sleep``/
    ``now`` are what make this testable without a real clock or a real server -- every production call site
    (`main`) leaves both at their real defaults.
    """
    spec = load_spec(cfg.learn_spec)
    routed = _routed_tags(spec, cfg.tasks)
    if not routed:
        raise IdleError(f"{cfg.learn_spec}: no tag has a router configured (nothing for idle retraining to do)")
    state = load_state(cfg.state_path)
    reports: list[RefitReport] = []
    idle_streak = 0
    polls = 0
    while stop_after is None or polls < stop_after:
        polls += 1
        stats = http_get_json(cfg.server_url.rstrip("/") + "/stats")
        if stats is None or is_busy(stats, max_queue_depth=cfg.max_queue_depth):
            idle_streak = 0
        else:
            idle_streak += 1
        if idle_streak >= cfg.idle_polls_required:
            idle_streak = 0
            t = now()
            due = [
                tag
                for tag in routed
                if _due_for_refit(
                    tag,
                    state.get(tag.task, TagIdleState()),
                    _count_experience_rows(
                        cfg.log_dir if cfg.log_dir is not None else cfg.learn_spec.parent / "experience", tag.task
                    ),
                    cfg,
                    t,
                )
            ]
            if due:
                # One tag per idle window: re-checking `/stats` before the next one keeps a long queue of
                # due tags from monopolising the idle window a caller's own traffic just handed back.
                report = refit_one_tag(cfg.learn_spec, due[0], cfg, state, t)
                reports.append(report)
                save_state(cfg.state_path, state)
        if stop_after is None or polls < stop_after:
            sleep(cfg.poll_interval_s)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prismyra-learn-idle")
    parser.add_argument("--server-url", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--learn-spec", required=True)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--task", action="append", default=[], help="repeatable; default is every routed tag")
    parser.add_argument("--state-path", default=None, help="default: <log-dir>/_idle/state.json")
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--idle-polls-required", type=int, default=3)
    parser.add_argument("--max-queue-depth", type=int, default=0)
    parser.add_argument("--retrain-interval-s", type=float, default=3600.0)
    parser.add_argument("--min-new-rows", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--nice", type=int, default=19, help="os.nice() value this whole process runs at")
    parser.add_argument(
        "--stop-after", type=int, default=None, help="number of polls, for a smoke test; default: forever"
    )
    args = parser.parse_args(argv)
    cap_blas_threads(args.num_threads)

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print('this needs the extra: pip install "prismyra[learn]"')
        return 1

    try:
        os.nice(args.nice)  # raises on some platforms if args.nice < 0 and the caller lacks privilege; that
        # is a configuration mistake worth refusing loudly, not swallowing into "ran at normal priority".
    except OSError as e:
        print(f"could not set nice({args.nice}): {e}")
        return 1

    import prismyra

    log_dir = Path(args.log_dir) if args.log_dir else None
    spec_path = Path(args.learn_spec)
    state_path = (
        Path(args.state_path)
        if args.state_path
        else (log_dir or spec_path.parent / "experience") / "_idle" / "state.json"
    )

    cfg = IdleConfig(
        server_url=args.server_url,
        learn_spec=spec_path,
        log_dir=log_dir,
        backbone=args.backbone,
        package_version=prismyra.__version__,
        state_path=state_path,
        poll_interval_s=args.poll_interval_s,
        idle_polls_required=args.idle_polls_required,
        max_queue_depth=args.max_queue_depth,
        retrain_interval_s=args.retrain_interval_s,
        min_new_rows=args.min_new_rows,
        num_threads=args.num_threads,
        nice=args.nice,
        tasks=tuple(args.task),
    )
    try:
        reports = run_forever(cfg, stop_after=args.stop_after)
    except IdleError as e:
        print(str(e))
        return 1
    for r in reports:
        status = (
            "admitted+reloaded"
            if r.reloaded
            else ("admitted" if r.admitted else ("error" if r.error else "not admitted"))
        )
        print(f"{r.task}: {status}" + (f" ({r.error})" if r.error else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
