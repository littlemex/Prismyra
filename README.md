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

**And one thing the speed table does not tell you.** Those figures are worth something only if the answers are right,
so they were measured on published labels. On RACE-middle the read-out scores 94.8% against 95.3% for the same model
generating the answer, and on BoolQ 89.5% against 88.8% -- both differences with a 95% interval spanning zero. The two
methods cannot be separated on accuracy, which is what makes comparing their speed worth doing.

**That equivalence is a short-context result.** Both of those tasks give a few hundred tokens of context. Bury the same
RACE questions in ten thousand tokens of other articles and the read-out scores 82.4% against 87.4% for generation, a
difference of -5.0% with an interval of [-8.6%, -1.3%] that **excludes zero**. Some of that is the model -- generation
falls too, by 6.9 points -- but the gap between the two methods opens with length, and this is the one comparison that
separates them. [docs/ACCURACY.md](docs/ACCURACY.md) has the three lengths and what is and is not established about
why.

**The crossover is real and it is measurable on those same tasks.** RACE asks 3.9 questions per article and the read-out
is 2.2x faster there; BoolQ asks one per passage and the read-out is **0.7x -- slower**. One question and this is the
wrong tool, exactly as the table above says.

**Where it does not work is sharper than where it does.** On a task whose interesting class is rare -- LexGLUE's unfair
terms-of-service clauses, where 1.5% of the answers are yes -- the read-out reaches 60% recall at 6% precision. It says
yes to nearly everything, and the same model generating the answer does the same, so this is not the read-out's limit
but the absence of a decision point: taking the larger of two probabilities stands it at 0.5 when it belongs near
0.97. A
closed-question classifier that works out of the box on a rare class is not what this is.

**What fixes it is a decision point, and it costs about eight hundred labelled documents.** `prismyra.thresholds` fits
one number per question from answers you already logged beside what turned out to be true -- no gradient, no device, no
second model. On those clauses it takes precision from 5.8% to 39.1% and F1 from 10.5% to 47.4% with recall unchanged.
Below roughly eight hundred documents it declines to fit rather than guessing, and above them it stops improving: two
and a half times the labels buys nothing, which says the rest of the gap is in how the read-out orders the probabilities
rather than where they are cut. [docs/ACCURACY.md](docs/ACCURACY.md) has all of it, including what did not
help and one figure this project published and then withdrew.

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

**An open context holds device memory until it is closed**, and the context itself is held once however many questions
read it:

| context | group 8 | group 32 | group 32, before the context was stored once |
|---|---|---|---|
| 1,000 tokens | 0.10 GiB | 0.33 GiB | 0.92 GiB |
| 5,000 tokens | 0.17 GiB | 0.41 GiB | 3.36 GiB |
| 20,000 tokens | 0.46 GiB | 0.69 GiB | 12.52 GiB |
| 100,000 tokens | 1.99 GiB | 2.22 GiB | 61.35 GiB |

Measured on one 48 GiB card: **19 contexts of 3,040 tokens can be open at once**, where three could before. The last
column is what it cost when each branch held its own copy of the context, which is what changed; the answers did not, to
the bit. `engine.cache_bytes(tokens)` gives the figure for your configuration. Use `close()` or a `with` block.

**Holding a context is the cheaper half.** Working on one costs several times as much, transiently: reading a context of
24,327 tokens allocates about 4.8 GiB above the 0.9 GiB it leaves behind, and answering thirty-two questions about it
peaks at 5.0 GiB. `engine.reading_bytes(tokens)` and `engine.answering_bytes(tokens, questions)` report both, from
figures this engine has measured on itself rather than from a formula. A context that will not fit is refused by name;
one whose *reading* will not fit is refused too, and the difference matters because asking fewer questions does not make
a read smaller -- only a shorter context does. There is therefore a longest context this card can read at all, and
`prismyra-bench ceiling` finds it.

## Images and video

An image or a clip goes in the context, which is exactly where this design wants it: the vision tower runs once and the
frames then behave like any other context token, so the questions after them are nearly free.

```python
from PIL import Image

with engine.open_context("A product photograph.", images=[Image.open("chair.jpg")]) as context:
    result = context.ask([
        Boolean(id="damaged", prompt="Is the item visibly damaged?"),
        Boolean(id="assembled", prompt="Is the item assembled?"),
        Choice(id="room", prompt="Which room is this for?", choices=["kitchen", "bedroom", "office"]),
    ])
```

`videos=` takes frames the same way, and the placeholders are assembled for you.

**A clip has to carry its timing.** Frames alone do not say how fast they run, and the processor then assumes a rate,
decides the clip is shorter than it is, and answers about a clip that does not exist. Nothing looks wrong: everything
visual is still right. Measured on the supported model, asked how long a six second clip is, it answers six when told
the rate and two when not. So `prismyra.media.decode_video` returns a `Clip` carrying the rate of the frames it hands
back, and passing that `Clip` is what makes timing questions mean anything:

```python
from prismyra.media import decode_video

clip = decode_video(open("delivery.mp4", "rb").read())
with engine.open_context("A doorway camera recording.", videos=[clip]) as context:
    result = context.ask([
        Boolean(id="person", prompt="Does a person appear?"),
        Choice(id="when", prompt="When does the parcel arrive?", choices=["beginning", "middle", "end"]),
        Scale(id="seconds", prompt="Roughly how many seconds long is this clip?", low=1, high=9),
    ])
```

A bare array of frames is still accepted; the processor guesses the rate, and the guess is its own rather than one made
here and presented as a fact.

The frame cap -- `decode_video(..., max_frames=...)`, and `--max-video-frames` on the server -- bounds decoding work
and host memory, not what the model sees. The processor does the real sampling, at a couple of frames per second of
the clip's own duration. Lowering the cap raises the stride and leaves the duration alone, which is why the six second
clip still reads as six at a cap of 32.

Measured on the supported model: reading a 336 by 336 image costs about 290 ms and a twelve frame clip about 300 ms,
after which a group of questions costs what it costs for text. A clip read backwards answers backwards, which is the
check that the three-axis positions media needs are being continued correctly.

Over HTTP, media arrives as base64 of the file's own bytes. A path would name a file on the server rather than on the
caller's machine, and a URL would send the server fetching whatever it was pointed at.

```bash
python3 - chair.jpg > request.json <<'EOF'
import base64, json, sys
print(json.dumps({
    "context": "A product photograph.",
    "images": [base64.b64encode(open(sys.argv[1], "rb").read()).decode()],
    "questions": [{"id": "damaged", "prompt": "Is the item visibly damaged?", "kind": "boolean"}],
}))
EOF

curl -s localhost:8000/ask -H 'content-type: application/json' --data-binary @request.json
```

Video files are decoded by the server, which is what the decoder in the `server` extra is for. Frames passed in process
need no decoder.

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
| [docs/ACCURACY.md](docs/ACCURACY.md) | Whether the answers are right, on public labels, and where they are not |

## Licence

Apache 2.0.
