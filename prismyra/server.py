"""An HTTP front end: one worker owns the device, callers queue.

Thin on purpose. The interesting decisions are all in the core -- the queue is in `prismyra.queue` because it is a
property of using one device from several threads, not an HTTP concern, and the question types are in `prismyra.schema`
because the wire format and the library's contract must not be allowed to drift apart. What is left here is translation.

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

from __future__ import annotations

import argparse
import dataclasses
from typing import Any, Literal

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


#: Hard limits on one request. An endpoint with none lets a single caller hold the device for as long as it likes, and
#: the failure arrives as everyone else's latency rather than as that caller's error.
#:
#: The context limit is in tokens, not characters, because that is what both costs are a function of -- and a character
#: count is a proxy that varies by a factor of two between languages. Time grows with the context, and so does the
#: key-value cache: at the default group of 32 on the supported model, 32,000 tokens is about 20 GiB.
#: `engine.cache_bytes(tokens)` gives the exact figure, and the engine refuses a context that will not fit.
MAX_CONTEXT_TOKENS = 32_000
MAX_QUESTIONS = 512


def create_app(
    model: str,
    *,
    max_queue: int = 512,
    request_timeout: float = 120.0,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_questions: int = MAX_QUESTIONS,
    **engine_kwargs,
):
    """A FastAPI application with the engine and its worker already running.

    The model loads at construction, not at the first request. A first request that pays for loading would report a
    latency no later request can reproduce, and a health check would pass before the server could answer anything.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    from . import Prismyra, __version__

    engine = Prismyra(model, **engine_kwargs)
    worker = Worker(lambda payload: engine.ask(payload[0], payload[1]), max_queue=max_queue).start()

    @asynccontextmanager
    async def lifespan(_):
        yield
        worker.stop()

    class QuestionIn(BaseModel):
        id: str | None = None
        prompt: str
        kind: Literal["boolean", "choice", "scale"] = "boolean"
        choices: list[str] | None = None
        low: int = 1
        high: int = 5

    class AskIn(BaseModel):
        context: str = Field(min_length=1)
        questions: list[QuestionIn] = Field(min_length=1)

    app = FastAPI(
        title="prismyra",
        version=__version__,
        lifespan=lifespan,
        description="Read one context once, then answer many typed questions about it.",
    )

    @app.post("/ask")
    def ask(body: AskIn) -> dict:
        # Refused before admission, so an oversized request costs the queue nothing. 413 rather than 422: the request is
        # well formed, there is just too much of it. Tokenising to find out is cheap next to what admitting it costs.
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
        try:
            job = worker.submit((body.context, questions), timeout=request_timeout)
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
        return as_json(with_queue_time(job.result, job.queue_ms))

    @app.get("/health")
    def health() -> dict:
        return {"ok": worker.alive, "depth": worker.depth}

    @app.get("/stats")
    def stats() -> dict:
        return {"engine": engine.stats(), "queue": worker.stats()}

    app.state.engine = engine
    app.state.worker = worker
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
        require_kernels=args.require_kernels,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
