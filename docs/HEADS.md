# Option-set heads

The read-out is zero-shot: each declared option is scored by the model's own output embedding (docs/READOUT.md). That is
the right default, and for a question you ask once it is the only sensible one. A question you ask on every request is
different -- the same options, usually the same prompt, thousands of times a day. For such a question a small head fitted
to the hidden state the read-out already sees can be much more accurate than the embedding rows of the option words,
and it costs one small matrix product per answer.

An option-set head is that, and nothing more:

- it reads the **same final hidden state** the read-out reads, at the same position;
- it answers **only** the questions that declare **exactly its set of options**;
- every other question is read exactly as without heads -- the same tensors, the same arithmetic, bit-identical
  probabilities. The backbone is never touched, so the context is read exactly as before too.

The engine knows nothing about what a head is for. Which questions it answers is data: the option set in its spec.

## Registering heads

```python
engine = Prismyra("Qwen/Qwen3.6-35B-A3B-FP8", heads="heads.json")
```

or `prismyra-serve --heads heads.json`. The spec is a JSON list, weights paths relative to the spec file:

```json
[{"name": "ticket-priority-v1", "options": ["low", "normal", "urgent"], "form": "mlp",
  "weights": "ticket-priority-v1.safetensors"}]
```

| field | meaning |
|---|---|
| `name` | reported in every answer the head gives, as `Answer.read_by` (and `read_by` in the server's JSON) |
| `options` | the option set the head answers; row i of its output scores the i-th option listed here |
| `form` | `linear` or `mlp` |
| `weights` | a safetensors file with the tensors below |

| form | tensors | logits |
|---|---|---|
| `linear` | `W` (k, hidden), `b` (k,) | `h W^T + b` |
| `mlp` | `W1` (m, hidden), `b1` (m,), `W2` (k, m), `b2` (k,) | `gelu(h W1^T + b1) W2^T + b2` |

`hidden` is the model's hidden size and `k` the number of options. `h` is the final hidden state as the engine holds it
(after the model's last norm); if you standardised features while fitting, fold the mean and scale into `W` / `W1` and
`b` / `b1` before saving. The head runs in float32. Its softmax over `k` logits is the answer's probabilities.

A spec that cannot be right is refused when the engine is built, not when a question arrives: a form that is not one of
the two, a missing tensor, a head that reads a different hidden size or scores a different number of options than it
declares, an option listed twice, and two heads for the same option set.

## Which questions a head answers

A question is answered by a head when its options are the head's options as a **set**. The order in which a caller
declares them does not matter, for the same reason it does not matter to the read-out: the rendered question lists its
options sorted, so two declarations of one set reach the same hidden state. The head's probabilities are returned in the
question's own declared order.

Everything else is untouched. A yes/no question, a lettered question, a question with one more or one fewer option, a
question with one option renamed: all are read from the output embedding, and `read_by` is `None`.

The prompt is not part of the key. A head is fitted to the hidden state of the questions it was trained on; asked a
different question over the same options, it still answers, and nothing guarantees that answer means anything. Keep
the prompt fixed for the questions a head serves, or give the head an option set no other question uses.

## What a head's probability means

Not what docs/READOUT.md defines. A head's probabilities are its own softmax, fitted to whatever labels it was trained
on -- possibly calibrated to them, possibly not. That is why every answer says which read it came from:
`Answer.read_by` is the head's name, or `None` for the output-embedding read-out. Store it beside the answer, as you
store `scoring_version`.

With `calibrate=True`, the priors are subtracted from the output-embedding scores only; a head's answer is its own and
no prior is applied to it.

## Fitting a head

A head is fitted outside the engine, on the hidden states of the questions it will answer. `record_hidden` collects
them at exactly the point a head is applied, so they are the features the head will see when served:

```python
import torch
from prismyra import Choice
from prismyra.heads import record_hidden

question = Choice(id="priority", prompt="How urgent is this ticket?", choices=["low", "normal", "urgent"])
with record_hidden(engine, question.options) as rec:
    for text in training_texts:
        engine.ask(text, [question])
features = torch.stack(rec.rows)          # (len(training_texts), hidden)
```

Fit a multinomial logistic regression or a small MLP on those features against your labels, fold any standardisation
into the first layer, and save the tensors in the layout above. Choose its strength on held-out examples, not on the
set you will report. Answers are not changed while recording.

## Cost

One `(1, hidden) x (hidden, m)` and one `(1, m) x (m, k)` product per answered question, or one `(1, hidden) x
(hidden, k)` for a linear head, in float32, after the branch pass. Against a branch pass of tens of milliseconds it does
not show in the timing.

## A measured example: how hard a coding-assistant request is

One head has been fitted and measured end to end. It answers a fixed question -- *how hard is this request for an AI
coding assistant: `simple`, `medium` or `complex`* -- with the request given as the context (`{"message": ...}`) and a
fixed rubric as the prompt. The question is the three-tier routing judgement of a public article on routing coding
requests by difficulty; the article's twenty Japanese requests are the first test below. The head's weights are not
published here; what follows is what it does and how far the measurements carry.

**What it is.** A two-layer head (`mlp`, 2,048 -> 256 -> 3, GELU) on the final hidden state of the published 36-layer
checkpoint, registered for the option set `{simple, medium, complex}`. It was fitted on 2,500 synthetic requests in
Japanese and English, written by Kimi K3 to cover one-liners to long requests with pasted code, logs and configuration,
and labelled by Kimi K3 twice with the rubric; only requests whose two labels agreed were kept (92.7% did). The tiers
were capped at 1.5 times the smallest, training stopped on a tenth of the training set, and the head was chosen on 278
further synthetic requests, not on any set below. Nothing in the backbone changed, so on the five decision sets the checkpoint
is checked against before release (reading comprehension, yes/no questions, typed policy decisions and two held-out
families) every probability was the same, bit for bit, with the head registered as without it.

**Results.** Accuracy against the stated truth; the three systems answered the same items. The 20 requests have
accepted alternatives per item, from the article; the other two sets are judged by large language models, because no
human labels exist for them.

| set | items | truth | 36-layer, no head | with the head | Strands Decider 2B | head - Decider, 95% interval |
|---|---|---|---|---|---|---|
| the article's requests (Japanese) | 20 | the article's accepted answers | 17 / 20 | 19 / 20 | 20 / 20 | -- |
| T300: synthetic, written by another generator | 300 | Kimi K3, majority of three | 67.0% | 86.7% | 64.3% | +22.3 [+16.7, +28.0] |
| T300 | 243 | Claude Opus 5.5 and GPT-6 Astra agree | 57.2% | 84.8% | 62.1% | +22.6 [+16.5, +28.8] |
| H300: WildChat-1M first turns (ODC-BY) | 250 | Claude Opus 5.5 and GPT-6 Astra agree | 77.6% | 87.6% | 69.6% | +18.0 [+12.4, +24.0] |

Intervals are paired bootstrap intervals over items (4,000 resamples). On H300 the margin over Decider stays above
eight points whichever single judge, or all three agreeing, is taken as the truth.

**Served.** `prismyra-serve --heads <spec>` with the short-input recordings and autotune pinning on, one request at a
time over HTTP on one L40S, gave on H300 the same answer for every item as the in-process measurement above (largest
difference in a probability 5e-7, the server's rounding). A request took 45-53 ms at the median (p90 53-82 ms) against
Decider's 90-95 ms (p90 98-153 ms) on its own L40S; 639 of the 642 requests replayed a recording. The server was ready
127 s after launch, 15 s of it taking the recordings, and held 35.6 GB of device memory, 1.8 GB of it the recordings.
On T300 the served run differs from the in-process one on 12 items (86.0% and 84.0% against the two truths): the
in-process run predates autotune pinning and came from a process whose block-inverse kernel had picked the other
configuration (docs/KERNELS.md), which is the difference pinning removes.

**Limits.** The labels on T300 and H300 come from language-model judges, not people; T300's requests were themselves
written by Kimi K3, the model whose labels the head was fitted to, which is why the Opus-and-Astra rows are given. Only
19 of H300's agreed items are Japanese, too few to say anything about Japanese. WildChat is people talking to a chat
assistant, not instructing an agent that works in their repository, so H300 is the nearest public stand-in rather than
the deployment's own traffic. The head answers only this exact option set; asked a different prompt over the same
three options it still answers, and nothing measured here says that answer is right.
