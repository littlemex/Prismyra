# `POST /v1/decide`: a JEV-compatible read-out

`autotrust/JEV-27B(-VL)` serves its own decision read-out at `POST /v1/decide` (its own `serve_decide.py`,
shipped inside the checkpoint). Prismyra serves the same request shape at the same path, so a caller already
built against JEV's API can point it at Prismyra instead, and answers it through `ask()` rather than JEV's
lm_head-LoRA decision head. The implementation is `prismyra/decide.py`; this is the contract.

## Request

A single object, or a JSON array of them (Prismyra's own extension -- see "Batching" below):

```json
{"kind": "noul", "state": "a document, or a JSON object", "question": "a yes/no question", "options": null, "id": null}
```

| field | type | meaning |
|---|---|---|
| `kind` | `"noul"` \| `"score"` \| `"choice"` | which closed vocabulary the question is read against |
| `state` | string or JSON object | the context. A string is used as-is; an object is rendered the same way JEV's own request parser renders one (`json.dumps`, not re-serialised to match Prismyra's own conventions) |
| `question` | string | the question text |
| `options` | list of strings, `choice` only | 2 to 256 named options, in the order the response reports them |
| `id` | string, optional | echoed back in the response and in `merged_with`; only meaningful in a batched (array) request |

`kind="noul"` and `kind="score"` ignore `options` -- their vocabularies are fixed (below). JEV's own protocol
also allows `state` to be a list mixing text and image parts, for a multimodal decision; Prismyra's two served
checkpoints are text-only, and a list `state` is refused by name rather than silently reading only its text.
`state` must also be non-empty: JEV allows a decision from the question alone, Prismyra's `ask()` requires a
context, and that is a real difference between the two servers, not an oversight.

## The three vocabularies

| `kind` | options | native type |
|---|---|---|
| `noul` | `["false", "true"]` | yes/no |
| `score` | `["0", "1", "2", "3", "4", "5"]` | 0-5 |
| `choice` | the caller's own `options`, 2-256 | one of N named options |

`noul` is read through Prismyra's own `Boolean` (`options=["no","yes"]`, the pair every other test in this
project already depends on existing) and relabelled on the way out -- the token actually scored on the device
is Prismyra's, not JEV's; the two label strings carry the same meaning, not the same bytes. `score` is read
through `Scale(low=0, high=5)`, which already renders exactly these six verbalizers, so there is no
relabelling at all.

`choice` is always read through a relabelling: every option is given a single-token symbol (`A`, `B`, `C`,
... then `AA`, `AB`, ... once the alphabet is exhausted), the question's prompt lists `A) option0`,
`B) option1`, ... the way JEV's own prompt already does, and the symbol scored is translated back to the
caller's own option text in the response. This runs the same way whether or not the caller's own wording
would have passed Prismyra's one-token-and-distinct-first-token rule unaided (`schema.py`'s account of that
rule) -- one code path, not a check-then-maybe-relabel branch, so a `choice` question's cost and behaviour do
not depend on the caller's wording in any way that would be silently different between two near-identical
requests. The symbol pool is found once per tokenizer with `prismyra.decide.choice_labels`, which reuses
`prismyra.readout.plan` -- the exact check Prismyra's own engine runs -- rather than a second, hand-rolled
tokenisation rule that could drift from it.

JEV's own ceiling is 256 options (its trained A-P head extended with Q-Z, then two-letter labels). Prismyra's
own ceiling, `schema.MAX_OPTIONS`, is 255: one `Question` cannot declare more. A `choice` request for 256
options is refused with both numbers named, not silently truncated to 255.

## Response

```json
{"id": "0", "kind": "noul", "options": ["false", "true"], "probabilities": [0.0011, 0.9989],
 "choice_index": 1, "choice": "true", "value": true, "adaptation": "native", "symbols": null,
 "protocol": "prismyra-decide-v1", "model": "...", "merged_with": []}
```

| field | meaning |
|---|---|
| `options` | the options in the order `probabilities` reports them: JEV's own fixed order for `noul`/`score`, the caller's own order (not Prismyra's internally-sorted rendering order) for `choice` |
| `probabilities` | a list, aligned to `options` -- not a dict, matching JEV's own shape |
| `choice_index`, `choice` | the argmax, by position and by option text |
| `value` | Prismyra's own typed read: `bool` for `noul`, `int` for `score`, the chosen option's own text for `choice`. JEV's protocol has no equivalent field |
| `adaptation` | `"native"` for `noul`/`score` (Prismyra's own tokens are scored directly); `"symbol-labelled"` for every `choice` question (see above) |
| `symbols` | `choice` only: the option-text -> label map actually used, for transparency. `null` otherwise |
| `protocol` | `"prismyra-decide-v1"` -- an identifier, not a claim of wire-level interchangeability with JEV's own `"jev27-bare-v1"` |
| `merged_with` | the `id`s of the other items in this request answered in the same `ask()` call as this one (see "Batching") |

**What a probability means here is Prismyra's own answer** (`SCORING` in the top-level README), not a
reproduction of JEV's. JEV applies a trained bias and a per-kind temperature before its own softmax
(its `decision_head.json`'s own note: "decision logits = lm_head-LoRA logprobs of verbalizer ids + bias, then
per-kind temperature"). Prismyra applies the analogous correction only when the engine was built with
`calibrate=True` (`prismyra.calibration.Calibration`, a different mechanism: a measured, content-free prior
subtracted before the softmax) and returns the plain softmax over the declared options otherwise. Nothing
here invents a bias or a temperature to make the two numbers mean the same thing.

## Batching: several decisions about one record cost one read, not several

JEV's own protocol is one call in, one decision out. A caller that wants several decisions about the same
`state` -- several `kind="choice"` fields on one record, say -- pays for that record's context exactly as
many times as it asks about it under JEV, because each call is its own forward pass.

Send a JSON array instead of a single object and Prismyra groups the items by their own (exact) `state` and
answers every group in one `ask()` call -- the context read once, however many kinds of decision were asked
about it:

```json
[
  {"kind": "noul", "state": "the record", "question": "Is it urgent?", "id": "urgent"},
  {"kind": "score", "state": "the record", "question": "How clear is it?", "id": "clarity"}
]
```

```json
{"results": [
  {"id": "urgent", "merged_with": ["clarity"], "...": "..."},
  {"id": "clarity", "merged_with": ["urgent"], "...": "..."}
], "num_model_requests": 1}
```

`num_model_requests` is the number of `ask()` calls the whole request actually cost -- equal to the number of
distinct states, which is what to compare against the number of items to see the saving. A batched request is
all-or-nothing: one invalid item refuses the whole request before anything reaches the device, the same policy
`/ask` already applies to a malformed question.

This goes through the same single-worker queue `/ask` uses when `--batcher` is off, not through
`prismyra.schedule.Batcher`: the saving here is merging *one request's own* decisions sharing a state, which
needs nothing more than the queue already serialising access to the device. Batching *across* different
callers' requests is `Batcher`'s own job (see the top-level README's "The scheduler" section), and `/v1/decide`
does not route through it.
