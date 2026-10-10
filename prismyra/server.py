"""An HTTP front end: one worker owns the device, callers queue.

Thin on purpose. The interesting decisions are all in the core -- the queue is in `prismyra.queue` because it is a
property of using one device from several threads, not an HTTP concern, and the question types are in
`prismyra.schema` because the wire format and the library's contract must not be allowed to drift apart. What is left
here is translation.

Two choices worth stating:

* **One worker, not a thread pool.** Requests entering the model together are correct but slow in a specific way: they
  share one stream, so all of them finish late instead of the first one finishing first. See docs/PERFORMANCE.md.
* **Queue depth is reported, not hidden.** `/stats` separates waiting from working, because their fixes differ: waiting
  is answered by another device, working only by kernels.

**Stateless.** There is no session, so every request carries its own context and `Context` -- the open context the
library offers -- is not reachable over HTTP. That is a decision, not an omission: a session needs an id, an owner, an
expiry, a memory budget and an eviction rule, and adding those later without breaking callers is easier than taking a
half-built version of them away. In-process callers that want follow-ups use `Prismyra.open_context` directly.
"""

# No `from __future__ import annotations` here, and that is load-bearing. It turns annotations into strings, which the
# web framework then resolves against this module's globals -- and the request models below are defined inside
# `create_app`, so they are not there. The framework finds an unresolvable name, decides the parameter cannot be a
# body, and every request is rejected with "field required" for a field the caller did send. Keeping the annotations
# as real objects is what makes the body a body.
import argparse
import base64
import dataclasses
import uuid
from typing import Any, Literal

from . import decide as decide_mod
from .media import MAX_DECODED_FRAMES, decode_image, decode_video
from .queue import QueueFull, Worker
from .schema import Boolean, Choice, PrismyraError, Question, QuestionError, Result, Scale


def build_questions(payload: list[dict[str, Any]]) -> list[Question]:
    """Turn the wire form into questions, rejecting a bad one before the device is touched.

    Validation happens here rather than inside the worker so that a malformed request costs no device time and cannot
    delay the requests queued behind it.
    """
    out: list[Question] = []
    for i, item in enumerate(payload):
        kind = item.get("kind", "boolean")
        common = {"id": item.get("id") or f"q{i}", "prompt": item.get("prompt", "")}
        if kind == "boolean":
            out.append(Boolean(**common))
        elif kind == "choice":
            out.append(Choice(**common, choices=item.get("choices") or []))
        elif kind == "scale":
            out.append(Scale(**common, low=int(item.get("low", 1)), high=int(item.get("high", 5))))
        else:
            raise QuestionError(
                f"question {common['id']!r} has unknown kind {kind!r}; expected boolean, choice or scale"
            )
    ids = [q.id for q in out]
    if len(set(ids)) != len(ids):
        # Reachable without the caller repeating anything: a missing id is filled from the position, which can collide
        # with one that was given. Answers are keyed by id, so the collision would drop an answer behind a 200.
        raise QuestionError(f"duplicate question ids: {ids}")
    return out


def as_json(result: Result) -> dict:
    """The response. `scoring_version` travels with the numbers, so a stored answer can be compared to a later one.

    The waiting time is read off the result like every other figure, rather than passed in beside it: two places that
    could each hold it is one place too many, and the one that disagrees is the one that gets reported.
    """
    return {
        "answers": {
            a.id: {
                "kind": a.kind,
                "value": a.value,
                "option": a.option,
                "probabilities": {k: round(v, 6) for k, v in a.probabilities.items()},
                **({"read_by": a.read_by} if a.read_by is not None else {}),
            }
            for a in result.values()
        },
        "timing": {
            "queue_ms": round(result.timing.queue_ms, 1),
            "context_ms": round(result.timing.context_ms, 1),
            "readout_ms": round(result.timing.readout_ms, 1),
            "total_ms": round(result.timing.total_ms, 1),
        },
        "model": result.model,
        "scoring_version": result.scoring_version,
        "context_tokens": result.context_tokens,
    }


def with_queue_time(result: Result, queue_ms: float) -> Result:
    """The engine cannot know how long its request waited, so the front end fills that field in."""
    return dataclasses.replace(result, timing=dataclasses.replace(result.timing, queue_ms=queue_ms))


def _batcher_capacity(engine) -> int:
    """The widest single request `Batcher.submit` will accept, mirroring `schedule.Limits.of` without needing a
    built `Batcher` to ask: a `lanes>1` router keeps no `Limits` of its own (every lane does, and they agree), so
    this is computed from the engine instead of read off either shape.

    A document's row budget is `engine.group`, widened to `WIDE_GROUP` exactly when `Batcher.__init__` would widen
    its own -- `engine.wide_group` set and `engine.group` still short of it. `/ask` uses this once, at construction,
    to decide which requests the batched path can take at all; a request wider than this still has somewhere to
    go, because `engine.ask()` itself answers an arbitrarily wide request in several branch passes, which is why
    an over-wide request falls back to the plain queue rather than being refused.
    """
    from .engine import WIDE_GROUP

    return WIDE_GROUP if getattr(engine, "wide_group", False) and engine.group < WIDE_GROUP else engine.group


def _ask_via_batcher(batcher, context: str, questions: list[Question], timeout: float):
    """`Batcher.submit` only enqueues; this adds the wait, so `/ask` can treat the result exactly like
    `Worker.submit`'s: the same three exceptions (`QueueFull`, `PrismyraError`, `TimeoutError`) and, on success, a
    `Job` whose `.result` and `.queue_ms` the response is built from.
    """
    job = batcher.submit(context, questions)  # may raise QueueFull or PrismyraError before anything is queued
    if not job.done.wait(timeout):
        job.cancelled = True
        raise TimeoutError(f"no answer within {timeout}s")
    if job.error is not None:
        raise job.error
    return job


def _namespaced(questions: list[Question]) -> tuple[list[Question], dict[str, str]]:
    """Give every question in this one request a batcher-wide-unique id, and the map back to what the caller sent.

    `/ask`'s own docstring promises a request is stateless and carries its own context -- which a caller reasonably
    reads as "my ids only have to be unique within my own request body", the same promise `build_questions` already
    enforces one request at a time. `Batcher` does not keep that promise on its own: `schedule.py`'s `_answer`
    docstring says two callers asking about the *same* document in one pass are "merged instead -- one document,
    both callers' questions, one set of rows", and the merge is `dict.setdefault(handle, []).extend(...)` -- a
    union, not a disjoint union. Two different callers who happen to both call their question `"ok"` about a
    document they both already had open get one one of their two rows silently answering for both ids once merged
    (reproduced here: a 64-document pool at a sustained arrival rate puts the same document in more than one
    forming pass's company, and every repeat used the same caller-chosen ids by construction, costing several
    requests a 422 for "duplicate question ids" that named a *different* caller's question, not this request's
    own). Renaming here, before `submit`, and renaming back in `_restore_ids` after the answer comes back, is
    local to the HTTP front end and changes nothing the model sees: `question.id` is read only for error text and
    for keying the answer dict (`engine.py`'s `q.id: _answer_for(...)` sites), never for row order or any
    computation, so this cannot move an answer.
    """
    prefix = uuid.uuid4().hex[:12]
    renamed = [dataclasses.replace(q, id=f"{prefix}:{q.id}") for q in questions]
    return renamed, {renamed_q.id: original.id for renamed_q, original in zip(renamed, questions, strict=True)}


def _restore_ids(result: Result, id_map: dict[str, str]) -> Result:
    """Undo `_namespaced`, keeping only *this* request's own answers.

    The namespacing alone is not the whole fix. `Batcher`'s merge for two callers sharing a document (same
    `_answer`'s `asked.setdefault(handle, []).extend(...)`) hands the *same* merged `Result` -- every companion's
    answers included, keyed by their own namespaced ids -- back to every job that named that document, not a
    per-job slice of it (`_answer`'s own `return [answers[self._resident[job.payload.digest]] for job in
    formed.jobs]`: one shared `Result` object, read once per job sharing the handle). Filtering by `k in id_map`
    is what turns "the whole document's answers" back into "the answers to the questions this request asked" --
    without it, two concurrent callers who happen to send the identical context text see each other's answers
    (and, before ids were namespaced at all, could see a `KeyError` here instead: a companion's own namespaced id
    is not a key this request's `id_map` ever had).
    """
    return dataclasses.replace(
        result,
        answers={id_map[k]: dataclasses.replace(a, id=id_map[k]) for k, a in result.answers.items() if k in id_map},
    )


#: Hard limits on one request. An endpoint with none lets a single caller hold the device for as long as it likes, and
#: the failure arrives as everyone else's latency rather than as that caller's error. The context limit is in tokens,
#: not characters, because that is what both costs are a function of -- and a character count is a proxy that varies
#: by a factor of two between languages. Time grows with the context, and so does the key-value cache: at the default
#: group of 32 on the supported model, 32,000 tokens is about 20 GiB. `engine.cache_bytes(tokens)` gives the exact
#: figure, and the engine refuses a context that will not fit.
MAX_CONTEXT_TOKENS = 32_000
MAX_QUESTIONS = 512

#: Media limits. Separate from the token limit because an image's cost in tokens is only known after the processor has
#: sized it, which is after the bytes have been accepted -- so the bytes are what has to be bounded.
MAX_MEDIA_BYTES = 64 * 1024 * 1024
MAX_IMAGES = 16
MAX_VIDEOS = 2


def create_app(
    model: str,
    *,
    max_queue: int = 512,
    request_timeout: float = 120.0,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_questions: int = MAX_QUESTIONS,
    max_media_bytes: int = MAX_MEDIA_BYTES,
    max_video_frames: int = MAX_DECODED_FRAMES,
    batcher: bool = False,
    linger_ms: float | None = None,
    lanes: int = 1,
    **engine_kwargs,
):
    """A FastAPI application with the engine and its worker already running.

    The model loads at construction, not at the first request. A first request that pays for loading would report a
    latency no later request can reproduce, and a health check would pass before the server could answer anything.

    **`batcher`** routes a text-only `/ask` request through `prismyra.schedule.Batcher` instead of answering it alone:
    several callers' documents and questions share passes, the way `docs/PERFORMANCE.md`'s "The scheduler" section
    measures. Off by default -- a caller who wants the plain one-request-at-a-time queue this always was still gets
    exactly that, unchanged, and nothing below changes shape for them. On, a request still goes to the plain queue
    when the batched path cannot take it: it carries media (`Batcher` is text-only, like `open_batch`) or it asks more
    questions than one pass holds (`_batcher_capacity`) -- `engine.ask()` answers that in several branch passes on its
    own, which `Batcher.submit` does not attempt. Either way the wire contract is the same request in, the same
    response shape out; which internal path answered it is not part of what a caller can observe.

    `batcher` requires the paged storage (`Batcher.__init__` refuses without it), so it is set to `True` here when
    the caller has not already said otherwise -- there is no CLI flag for `paged` on its own, and a caller passing
    `batcher=True` plainly wants the one precondition it has.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    from . import Prismyra, __version__

    if batcher:
        engine_kwargs.setdefault("paged", True)
    engine = Prismyra(model, **engine_kwargs)
    worker = Worker(
        lambda payload: engine.ask(payload[0], payload[1], images=payload[2] or None, videos=payload[3] or None),
        max_queue=max_queue,
    ).start()

    text_batcher = None
    batcher_capacity = 0
    if batcher:
        from .schedule import Batcher

        text_batcher = Batcher(engine, max_queue=max_queue, linger_ms=linger_ms, lanes=lanes).start()
        batcher_capacity = _batcher_capacity(engine)

    @asynccontextmanager
    async def lifespan(_):
        yield
        worker.stop()
        if text_batcher is not None:
            text_batcher.stop()

    class QuestionIn(BaseModel):
        id: str | None = None
        prompt: str
        kind: Literal["boolean", "choice", "scale"] = "boolean"
        choices: list[str] | None = None
        low: int = 1
        high: int = 5

    class AskIn(BaseModel):
        context: str = ""
        questions: list[QuestionIn] = Field(min_length=1)
        #: Base64 of the file's own bytes. A path would name a file on this machine rather than on the caller's, and a
        #: URL would make the server fetch whatever it is pointed at.
        images: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)
        videos: list[str] = Field(default_factory=list, max_length=MAX_VIDEOS)

    class DecideIn(BaseModel):
        """JEV's own `DecideRequest` shape (`autotrust/JEV-27B-VL`'s `serve_decide.py`), accepted as-is so a client
        built against JEV needs no changes to its request body. `state` is `str | dict` rather than JEV's
        `str | dict | list`: the list form carries image/video parts, which `decide.py` refuses by name rather than
        silently reading only the text half of it -- see `decide._state_to_context`.
        """

        kind: Literal["noul", "score", "choice"]
        state: str | dict = ""
        question: str
        options: list[str] | None = None
        #: Not part of JEV's protocol. Only meaningful in a batched request (a JSON array body): it is echoed back
        #: so a caller can match a response item to the request item that produced it without relying on list order,
        #: and it is what `merged_with` names the other items in a group by.
        id: str | None = None

    app = FastAPI(
        title="prismyra",
        version=__version__,
        lifespan=lifespan,
        description="Read one context once, then answer many typed questions about it.",
    )

    @app.post("/ask")
    def ask(body: AskIn) -> dict:
        # Refused before admission, so an oversized request costs the queue nothing. 413 rather than 422: the request
        # is well formed, there is just too much of it. Tokenising to find out is cheap next to what admitting it
        # costs.
        if not body.context.strip() and not body.images and not body.videos:
            raise HTTPException(status_code=422, detail="a request needs a context, an image or a video")
        encoded = body.images + body.videos
        total = sum(len(blob) for blob in encoded)
        if total > max_media_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"the media is {total} encoded bytes and the limit is {max_media_bytes}",
            )
        try:
            images = [decode_image(base64.b64decode(blob, validate=True)) for blob in body.images]
            videos = [
                decode_video(base64.b64decode(blob, validate=True), max_frames=max_video_frames) for blob in body.videos
            ]
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=422, detail=f"could not read the media: {e}") from e

        # Text tokens only. What the images add is known only after the processor has sized them, and the engine
        # refuses a context that will not fit on the device by name, which is the limit that actually binds.
        tokens = len(engine.tokenizer(body.context)["input_ids"])
        if tokens > max_context_tokens:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"the context is {tokens} tokens and the limit is {max_context_tokens}; it would need "
                    f"{engine.cache_bytes(tokens) / 1024**3:.1f} GiB of key-value cache"
                ),
            )
        if len(body.questions) > max_questions:
            raise HTTPException(
                status_code=413,
                detail=f"{len(body.questions)} questions were asked and the limit is {max_questions}",
            )
        try:
            questions = build_questions([q.model_dump() for q in body.questions])
        except QuestionError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        id_map = None
        try:
            # The batched path only when it can actually take the request: no media (`Batcher` is text-only) and
            # not more questions than one pass holds. Everything else -- including every request when `batcher` is
            # off -- takes the queue this endpoint always had, unchanged.
            if text_batcher is not None and not images and not videos and len(questions) <= batcher_capacity:
                namespaced, id_map = _namespaced(questions)
                job = _ask_via_batcher(text_batcher, body.context, namespaced, request_timeout)
            else:
                job = worker.submit((body.context, questions, images, videos), timeout=request_timeout)
        except QueueFull as e:
            # The work was never started, so 503 with Retry-After is the honest answer. Its own exception type matters
            # here: the engine raises `RuntimeError` too, and reporting that as an overloaded server would send the
            # caller to retry a request that will fail the same way.
            raise HTTPException(status_code=503, detail=str(e), headers={"Retry-After": "1"}) from e
        except TimeoutError as e:
            # Admitted and still running. The device is not free, so a retry should not arrive immediately.
            raise HTTPException(status_code=504, detail=str(e), headers={"Retry-After": "5"}) from e
        except PrismyraError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        result = _restore_ids(job.result, id_map) if id_map is not None else job.result
        return as_json(with_queue_time(result, job.queue_ms))

    @app.post("/v1/decide")
    def decide(body: DecideIn | list[DecideIn]) -> dict:
        """A JEV-compatible read-out (see `prismyra.decide`'s own module docstring for the contract and why
        a `choice` question is always relabelled). The body is a single JEV-shaped decision -- answered and
        returned exactly as JEV's own `/v1/decide` would shape one -- or a JSON array of them, Prismyra's own
        extension: items that share one `state` are answered in one `ask()` call instead of one each, and the
        response is wrapped with `num_model_requests` so that saving is something a caller can see rather than
        take on faith.

        Goes through the same single-worker queue `/ask` uses when `--batcher` is off, not through `Batcher`:
        this endpoint's own saving is merging *one request's own* decisions sharing a state, which needs no
        more than the queue already serialising access to the device; cross-request batching is `Batcher`'s
        job and this does not attempt it.
        """
        if isinstance(body, DecideIn):
            single = True
            raw_items = [body.model_dump()]
        else:
            single = False
            raw_items = [b.model_dump() for b in body]

        def ask(context: str, questions: list[Question]):
            job = worker.submit((context, questions, None, None), timeout=request_timeout)
            return job.result

        try:
            responses, num_requests = decide_mod.decide_many(engine, raw_items, ask)
        except decide_mod.DecideError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except QuestionError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except QueueFull as e:
            raise HTTPException(status_code=503, detail=str(e), headers={"Retry-After": "1"}) from e
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e), headers={"Retry-After": "5"}) from e
        except PrismyraError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        if single:
            return responses[0]
        return {"results": responses, "num_model_requests": num_requests}

    @app.get("/health")
    def health() -> dict:
        return {"ok": worker.alive, "depth": worker.depth}

    @app.get("/stats")
    def stats() -> dict:
        out = {"engine": engine.stats(), "queue": worker.stats()}
        if text_batcher is not None:
            out["batcher"] = text_batcher.stats()
        return out

    app.state.engine = engine
    app.state.worker = worker
    app.state.batcher = text_batcher
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prismyra-serve")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-queue", type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=MAX_DECODED_FRAMES,
        help="how many frames to decode from a clip at most; the processor then samples at the clip's own rate",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="how long a request may wait for the device before it is answered with 504",
    )
    parser.add_argument(
        "--require-kernels",
        action="store_true",
        help="refuse to start without the faster kernels rather than serving at a quarter of the speed",
    )
    parser.add_argument(
        "--no-short-graphs",
        action="store_true",
        help="read every single-question request eagerly instead of replaying the startup recordings; frees the "
        "device memory they hold (about 1.7 GiB on the supported model) and starts faster",
    )
    parser.add_argument(
        "--heads",
        default=None,
        help="a JSON spec of option-set heads (see prismyra.heads); questions whose options match none are unaffected",
    )
    parser.add_argument(
        "--batcher",
        action="store_true",
        help="answer a text-only /ask alongside whatever else is already waiting, through prismyra.schedule.Batcher, "
        "instead of alone; off by default. Implies --paged (there is no separate flag for it: Batcher refuses "
        "without the paged storage). A request with media, or more questions than one pass holds, still answers "
        "through the plain queue this flag leaves otherwise unchanged.",
    )
    parser.add_argument(
        "--linger-ms",
        type=float,
        default=None,
        help="with --batcher, how long a forming pass waits with nothing new arriving before it runs (default: the "
        "scheduler's own, 2 ms, bounded by what the engine has measured a pass to cost -- see schedule.Limits)",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=1,
        help="with --batcher, how many independent pass pipelines run under one Batcher; more than one lets a "
        "second pass start while the first is still running, at the cost of a second shelf's own device memory",
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print('the server needs its extra: pip install "prismyra[server]"')
        return 1

    app = create_app(
        args.model,
        max_queue=args.max_queue,
        request_timeout=args.request_timeout,
        max_context_tokens=args.max_context_tokens,
        max_questions=args.max_questions,
        max_video_frames=args.max_video_frames,
        require_kernels=args.require_kernels,
        **({"short_graphs": False} if args.no_short_graphs else {}),
        heads=args.heads,
        batcher=args.batcher,
        linger_ms=args.linger_ms,
        lanes=args.lanes,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    # So that `python -m prismyra.server` works as well as the `prismyra-serve` script. Without it the module imports,
    # defines `main`, and exits silently with a zero status -- which reads as a server that started and stopped.
    raise SystemExit(main())
