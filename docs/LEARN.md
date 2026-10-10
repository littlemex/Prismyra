# Tags, experience, and an offline student (stage 1/2)

`prismyra.learn` is the smallest version of DISTILL-RL-DESIGN-v2.md's distillation plan that can be called
"learning" without touching the serving path: a closed, operator-registered vocabulary of decisions ("tags"), a
bounded-queue log of what Prismyra answered under those tags, and a hand-run CLI that fits a small, CPU-only
student from the log and evaluates it offline. Off by default, and the default costs nothing: see
[Guarantees](#guarantees).

The design calls for more than this -- a router that shifts traffic to a student, head reinforcement learning, a
LoRA update loop -- in later stages. None of it is here. Importing it from this package does not work, by
design.

## Registering tags

```json
[{"task": "phishing-v1", "question": "Is this email a phishing attempt?", "options": ["no", "yes"],
  "kind": "noul", "normalize": "strip+lower", "retain_days": 30, "keep_hidden": false,
  "eval_set": "evals/phishing-v1-gold.jsonl"}]
```

A JSON list, one entry per tag, every field required and checked in full before the server starts
(`prismyra.learn.spec.load_spec`; see that module for exactly what each field must be).

| field | meaning |
|---|---|
| `task` | the tag's name; unique across the file |
| `question` | the question text a request must match (after `normalize`) to be tagged, unless it names `task` directly |
| `options` | the option set; compared as a set, order does not matter |
| `kind` | an opaque label compared for exact equality -- `/ask`'s `boolean`/`choice`/`scale` or `/v1/decide`'s `noul`/`score`/`choice`, an operator's choice, never interpreted |
| `normalize` | `none`, `strip`, `lower`, or `strip+lower`, applied to both the request's and the entry's own question text before comparing |
| `retain_days` | how long logged experience for this tag is kept before it is deleted |
| `keep_hidden` | whether a hidden state may be retained alongside this tag's experience (stage 1/2 never captures one; see below) |
| `eval_set` | a path, relative to the spec file, to the tag's gold-labelled evaluation set (`prismyra.learn.fit.load_eval_set`) |

A request is tagged in exactly one of two ways (`prismyra.learn.tagging.tag_for`): it names a registered `task`
directly (and still has to agree with that entry's `kind` and option set), or its `(kind, question, options)`
matches one entry exactly once normalized. Anything else is untagged, and untagged means unseen -- no log entry,
no feature, no student ever sees it. A document's own text never participates in matching: two different
documents asked the same registered question are the same tag.

## Running with it

```
prismyra-serve --model ... --learn-spec learn.json
```

Optional: `--learn-log-dir` (default: an `experience/` directory next to the spec file), `--learn-backend`
(`local`, the default, or `ray`), `--learn-max-queue` (default 10,000).

`/stats` gains a `learn` key: `{"logged": N, "dropped": N, "queue_depth": N}`. `dropped` counts experience that
could not be written -- the queue was full, or the write itself failed -- never a request that failed.

Experience is written as one JSONL file per `(tag, UTC date)`, under `<log-dir>/<tag>/<YYYY-MM-DD>.jsonl`. Each
line is one `prismyra.learn.experience.Experience`: the tag, the spec's own content-hash version, the context,
the question, the options, the kind, Prismyra's probabilities, who answered (`"prismyra"` or a head's name),
the version tuple (`package`/`backbone`/`spec`), an optional reward-or-label (populated out of band, never on
the request path), and `h` (always `None` in stage 1/2 -- see below). Files older than the tag's `retain_days`
are deleted automatically.

## Fitting a student (stage 2)

```
pip install "prismyra[learn]"
python -m prismyra.learn.fit --learn-spec learn.json --task phishing-v1 \
  --backbone littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l --out students/phishing-v1.json
```

Trains a LightGBM multiclass model on the tag's logged experience (soft labels: Prismyra's own probabilities,
not just its top choice), features built from the input alone -- never the question text, which is constant
within one tag, and never a hidden state (see below) -- by a deterministic hashed TF-IDF
(`prismyra.learn.features`) that handles both a short sentence and a flattened JSON state (a game's board, say)
with one code path.

A student is admitted only if **both**:

1. On `eval_set` -- gold labels, not Prismyra's own -- the student's accuracy is at least Prismyra's own minus
   `--accuracy-margin`. (Needs an `eval_set` item carrying Prismyra's own prediction, collected once, out of
   band; this CLI never loads a model itself.)
2. On a held-out, de-duplicated slice of the tag's own logged experience, the one-sided 95% lower bound of the
   student/Prismyra agreement rate clears `--min-match-rate`.

Agreement with Prismyra alone is deliberately not enough: a student that reproduces Prismyra's own mistakes has
a high agreement rate by construction. The artifact (one JSON file, `StudentArtifact`) also reports a
confidence-threshold (`tau`) sweep -- answer rate versus disagreement rate -- for an operator to pick a point
on, rather than this script picking one. **The artifact is saved, never wired into serving.** Stage 4 is what
would do that, and it does not exist yet.

## Guarantees

- **Off by default.** No `--learn-spec` means `prismyra.learn` is never imported, and nothing on `/ask` or
  `/v1/decide` changes shape: both make exactly the comparisons they always made.
- **One comparison when on, for every request.** A spec loaded means every request pays one `tag(...)` lookup
  (pure Python, no device). Only a *tagged* request pays more than that: a `FastAPI` background task, scheduled
  after the response has already been sent, that enqueues the experience onto a bounded queue a separate thread
  drains.
- **The queue drops rather than blocks.** A full queue increments `dropped` and returns; nothing on the request
  path ever waits on a disk write.
- **`h` is not captured in stage 1/2.** The field exists in `Experience` and in every tag's `keep_hidden` policy,
  ready for the stage that needs it (head reinforcement learning, 6b), but nothing on this path reads a hidden
  state -- stage 2's own student uses input features only. Capturing it would mean wrapping `engine.ask` in
  `prismyra.heads.record_hidden` on a path shared with every other caller's request, including ones `--batcher`
  has merged into the same forward pass, for a feature nothing here consumes yet.
