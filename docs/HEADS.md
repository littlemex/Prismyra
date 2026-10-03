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
