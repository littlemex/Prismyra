# Prismyra

Read one context once, then answer many typed questions about it.

```python
from prismyra import Prismyra, Boolean, Choice

engine = Prismyra("Qwen/Qwen3.6-35B-A3B-FP8")

result = engine.ask(
    context=open("returns-policy.txt").read(),
    questions=[
        Boolean(id="returnable", prompt="Can an opened item be returned?"),
        Boolean(id="within_30", prompt="Is there a thirty day limit?"),
        Choice(id="who_pays", prompt="Who pays return shipping?", choices=["seller", "buyer"]),
    ],
)

result["returnable"].value        # False
result["who_pays"].option         # "buyer"
result["who_pays"].probabilities  # {"seller": 0.19, "buyer": 0.81}
result.timing.total_ms            # 229.5
```

## What this is for, and where it is the wrong tool

Prismyra answers **closed questions about one shared context**. It reads the context once and each question then costs
almost nothing, so the more questions you ask of the same text, the better it does. It generates no text at all.

That shape has a hard edge, and here it is as a number. On the machine in
[`benchmarks/results/`](benchmarks/results/), with a context of about 5,000 tokens:

| questions | Prismyra | vLLM |
|---|---|---|
| 1 | 226 ms | **134 ms** |
| 2 | 227 ms | 181 ms |
| 3 | 228 ms | **228 ms** |
| 4 | 229 ms | 275 ms |
| 16 | 228 ms | 839 ms |
| 64 | 320 ms | 3,095 ms |
| 128 | 503 ms | 6,103 ms |

**Below four questions, use a general serving engine.** Prismyra pays for one traversal of the model up front and only
earns it back by amortising over questions; a single question pays the whole thing for one answer. Above four, the gap
widens quickly, because a traversal costs the same whether it carries one question or thirty-two.

Use it for: classifying, routing, scoring or extracting many fields from the same document, transcript, ticket or page.
Do not use it for: generating text, chat, one-off questions, or anything where the context differs per question.

## Install

```bash
pip install prismyra              # runs, and leaves every borrowed kernel on its fallback
pip install "prismyra[fast]"      # the kernels: needs vLLM and Triton
```

Without the `fast` extra the model still answers, and `engine.stats()["kernels"]` says what was skipped. See
[docs/KERNELS.md](docs/KERNELS.md) for what each kernel is worth.

## Asking more than once

`ask` reads the context and throws it away. When a follow-up question depends on an earlier answer, keep the context
open instead -- the reading is the expensive half and it is not repeated:

```python
with engine.open_context(ticket_text) as context:
    triage = context.ask([Choice(id="queue", prompt="Which queue?", choices=["billing", "technical", "other"])])
    if triage["queue"].option == "billing":
        detail = context.ask([Boolean(id="refund_ok", prompt="Does this qualify for a refund?")])
```

**An open context holds device memory until it is closed.** Each branch carries its own copy of the context's keys and
values -- that is what lets branches be written independently -- so an open context costs the context's key-value cache
multiplied by the group:

| context | group 8 | group 32 |
|---|---|---|
| 1,000 tokens | 0.23 GiB | 0.92 GiB |
| 5,000 tokens | 0.84 GiB | 3.36 GiB |
| 20,000 tokens | 3.13 GiB | 12.52 GiB |
| 100,000 tokens | 15.34 GiB | 61.35 GiB |

Modest at ordinary lengths and not at long ones. `engine.cache_bytes(tokens)` gives the figure for your configuration,
and a context that will not fit is refused by name rather than by an allocator. Lowering `group` lowers this
proportionally at the cost of one extra traversal per group of questions. Use `close()` or a `with` block.

## Several requests at once

```python
from prismyra import Request

results = engine.ask_many([
    Request(context=a, questions=[...]),
    Request(context=b, questions=[...]),
])
```

Each result is a `Result` or the error for that slot: one bad request does not fail the others. Nothing is shared
between requests -- different contexts share no state.

There is a server with that already wired:

```bash
pip install "prismyra[server,fast]"
prismyra-serve --require-kernels
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{
  "context": "Returns are accepted within thirty days.",
  "questions": [{"id": "thirty", "prompt": "Is there a thirty day limit?", "kind": "boolean"}]
}'
```

`GET /stats` reports queue depth and latency, with waiting separated from working -- their fixes differ, and one latency
figure hides which one is binding. `--require-kernels` refuses to start rather than serve at a quarter of the speed.

Inside a process, put a `prismyra.queue.Worker` in front of the engine directly. One worker owns the device
and callers queue. Requests entering the model together are correct but slow in a particular way: they share one stream,
so all of them finish late instead of the first one finishing first. Measured, at eight arriving together: median
latency 2,113 ms interleaved against 945 ms queued, and the first answer home at 2,113 ms against 214 ms.

## What a probability means

Each option's single token is scored by the model's own output embedding at the branch's final position, and those
scores are softmaxed **over the options you declared**. So a probability is a relative preference among your options. It
is not calibrated and it is not comparable across different option sets. There is deliberately no `confidence` field.

`result.scoring_version` travels with the numbers; store it if you store answers. Full statement:
[docs/READOUT.md](docs/READOUT.md).

Two things are refused rather than answered wrongly: an option that is more than one token with a leading space, and two
options whose first token is the same. Both need the model's tokenizer, so they are refused by `engine.validate` before
the context is read -- early enough to cost no device time, but not as early as construction.

## Supported models

One, for now: `Qwen/Qwen3.6-35B-A3B-FP8`. The faster kernels are applied through a per-model adapter that checks it
found the number of modules it was measured against, and fails rather than quietly leaving the slow path in place.
Another checkpoint will load and answer; it will not get the kernels until an adapter is written and measured for it.

## Performance

The cost model, at about 5,000 context tokens:

```
total ~= 138 ms  +  92 ms x ceil(questions / 32)
```

Both constants grow with the context length. Numbers come from
[`benchmarks/results/`](benchmarks/results/) and are refreshed by hand on the machine each file names --
CI has no GPU and **cannot catch a performance regression**. On such a machine:

```bash
prismyra-bench sweep --require-kernels
prismyra-bench compare --against benchmarks/results/qwen3_6_35b_a3b_fp8__rtx_pro_6000.json
```

`compare` exits non-zero on any point more than ten per cent slower. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

| | |
|---|---|
| [docs/FORK.md](docs/FORK.md) | How one context serves many questions, and the asymmetry it rests on |
| [docs/READOUT.md](docs/READOUT.md) | What a probability is, exactly |
| [docs/KERNELS.md](docs/KERNELS.md) | Each replacement, what it is worth, and what was rejected |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | How the numbers were measured and how to reproduce them |

## Licence

Apache 2.0.
