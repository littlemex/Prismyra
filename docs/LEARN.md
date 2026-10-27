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

A JSON list, one entry per tag, every required field checked in full before the server starts
(`prismyra.learn.spec.load_spec`; see that module for exactly what each field must be). An entry may be
exactly `REQUIRED_FIELDS` plus zero or more of `OPTIONAL_FIELDS`; anything else is a refusal to start.

Required:

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

Optional (omitting any of these keeps the tag meaning exactly what it meant before the field existed):

| field | meaning |
|---|---|
| `features` | the name of a feature builder registered with `prismyra.learn.features.register_features`, used by stage 2's student instead of the hashed-TF-IDF default; an unregistered name is a refusal at fit time, not at spec-load time |
| `min_match_rate` | overrides, for this tag only, the student-admission condition 2 bar below (default: `prismyra.learn.spec.DEFAULT_MIN_MATCH_RATE`, 0.90); a number in `(0, 1]` |
| `min_accuracy_ratio` | overrides, for this tag only, the student-admission condition 1 bar below (default: `prismyra.learn.spec.DEFAULT_MIN_ACCURACY_RATIO`, 0.90); a number in `(0, 1]` |
| `student` | which `prismyra.learn.backends.StudentBackend` trains and predicts for this tag's stage 2 student (`"lightgbm"` or `"xgboost"`; default `"lightgbm"`) -- see [Choosing a student backend](#choosing-a-student-backend-lightgbm-or-xgboost) below |

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

1. On `eval_set` -- gold labels, not Prismyra's own -- the student's accuracy is at least the tag's
   `min_accuracy_ratio` (default 0.90) **times** Prismyra's own accuracy. (Needs an `eval_set` item carrying
   Prismyra's own prediction, collected once, out of band; this CLI never loads a model itself.)
2. On a held-out, de-duplicated slice of the tag's own logged experience, the one-sided 95% lower bound of the
   student/Prismyra agreement rate clears the tag's `min_match_rate` (default 0.90).

Both numbers default to `prismyra.learn.spec.DEFAULT_MIN_MATCH_RATE` / `DEFAULT_MIN_ACCURACY_RATIO` -- defined
once, there, not scattered across this script's own flags. A tag whose traffic needs a different bar sets
`min_match_rate` and/or `min_accuracy_ratio` on its own entry (see the field table above); `--min-match-rate`
and `--min-accuracy-ratio` additionally let an operator override either, for one run only, without editing the
spec file (both default to unset, meaning "use the tag's own value"). Lenient on purpose: distillation's point
is a student that is fast and cheap, not one that reproduces Prismyra's own mistakes on every single board --
0.90/0.90 is "close enough to Prismyra, correct often enough on gold" rather than "indistinguishable from
Prismyra".

Agreement with Prismyra alone is deliberately not enough: a student that reproduces Prismyra's own mistakes has
a high agreement rate by construction, which is why condition 1 is scored against `eval_set`'s gold labels, not
against Prismyra's own answer. The artifact (one JSON file, `StudentArtifact`) also reports a
confidence-threshold (`tau`) sweep -- answer rate versus disagreement rate -- for an operator to pick a point
on, rather than this script picking one. **The artifact is saved, never wired into serving on its own.** See
[Stage 4](#stage-4-switching-traffic-between-a-student-and-prismyra) below for the part that can route traffic
to an admitted artifact.

## Choosing a student backend: LightGBM or XGBoost

`prismyra.learn.backends.StudentBackend` is the seam between `fit`/`students` and the library that actually
trains and predicts. Two implementations ship today, `"lightgbm"` (the default, unchanged behaviour) and
`"xgboost"`. A tag picks one with its own optional `"student"` field:

```json
{"task": "sentiment-finnews-v1", ..., "student": "xgboost"}
```

Omitting `"student"` keeps a tag meaning exactly what it meant before this field existed (`"lightgbm"`).
`python -m prismyra.learn.fit --student-backend {lightgbm,xgboost}` overrides it for one run, the same
per-run-override shape `--min-match-rate`/`--min-accuracy-ratio` already have. `"xgboost"` is an optional
extra (`pip install "prismyra[learn-xgb]"`); neither library is imported unless a tag actually asks for it.

Both backends train on the same soft-label rows `_soft_rows` already builds (one row per non-zero-probability
option, weighted by that option's probability) -- LightGBM as `objective: "multiclass"`, XGBoost as
`objective: "multi:softprob"`, both `deterministic`/CPU-only, both pinned to a fixed `seed` and `num_threads`.
XGBoost's own determinism mechanism is `tree_method="hist"` with no row/column subsampling, which on CPU gives
the same "same build, same input, same seed/threads -> byte-identical model" contract LightGBM's own
`deterministic=True` already promises -- see `prismyra/learn/backends.py` for the one place the two actually
differ (XGBoost, with no subsampling active, does not depend on `seed` at all, which is a stronger guarantee,
not a weaker one). `StudentArtifact.backend`/`.backend_version` record which library and version produced an
artifact; an artifact written before these fields existed reads as `"lightgbm"`/`None`, unchanged.

### Real-task measurement (2026-10-27, one L40S, `littlemex/prismyra-decision-35b-a3b-fp8` v0.5.1)

Two commercially-clear, real 3-class sentiment tasks (`docs/DATA_PROVENANCE.md`'s own cleared list):
`cardiffnlp/tweet_eval` (`sentiment` config) and `zeroshot/twitter-financial-news-sentiment`. 2,500 real
training texts per task were sent through the real engine's `/v1/decide` (`kind: "choice"`) to log experience
the ordinary way; a separate 200-row gold `eval_set` per task was built the same way, carrying Prismyra's own
`prismyra_probabilities`.

At the project's own default admission bar (`min_match_rate`/`min_accuracy_ratio` = 0.90/0.90), **neither
backend cleared admission on either task at this data size**:

| task | backend | eval accuracy (student) | Prismyra accuracy | held-out agreement (95% lower bound) |
|---|---|---|---|---|
| tweet sentiment | lightgbm | 0.670 | 0.755 | 0.467 |
| tweet sentiment | xgboost | 0.645 | 0.755 | 0.480 |
| financial-news sentiment | lightgbm | 0.825 | 0.915 | 0.788 |
| financial-news sentiment | xgboost | 0.775 | 0.915 | 0.774 |

Financial-news headlines (explicit words like "cut", "downgrade") agree with the teacher far more than general
tweets do at this data size; neither reaches the 0.90 bar. This is reported as measured, not adjusted to force
a pass -- a real win on real tasks is left to a future round with more logged traffic, better features
(`prismyra.learn.features.register_features`), or a tag-level override of the bar for traffic that genuinely
needs a lower one.

To exercise the stage 4 routing mechanics end-to-end against this real engine, one further, explicitly labelled
artifact was fit with a one-run-only override (`--min-match-rate 0.3 --min-accuracy-ratio 0.3`) on the
financial-news task's xgboost student -- a demonstration artifact, not a claim that it would be admitted under
the default bar. Measured over real HTTP (loopback, same host as the engine), against the same 200 gold rows,
with a fresh `--learn-log-dir` per router mode (switching a tag's router mode on a shared controller state can
otherwise inherit the previous mode's own `ratio`, including a mode like `"shadow"` that leaves it at `0`):

| router mode | student-served fraction | p50 latency |
|---|---|---|
| none (Prismyra only) | 0% | 81.34 ms |
| `ab`, `ratio: 10000` (lightgbm) | 100% (60/60) | 43.57 ms |
| `ab`, `ratio: 10000` (xgboost) | 100% (60/60) | 42.46 ms |
| `shadow` | 0% (by design -- never serves the student) | -- |
| `confidence`, `tau: 0.5` | 93.3% (56/60) | 46.22 ms |
| `ramp` (5 windows, `ramp_step: 2000`/8s) | 16% -> 40% -> 60% -> 76% -> 100% | 74.7 -> 73.2 -> 44.4 -> 43.9 -> 43.2 ms |

A student-served request is about **1.9x faster end-to-end** than Prismyra's own (43 ms vs 81 ms, HTTP
included) -- both backends land at essentially the same real latency, because the booster's own prediction
cost (sub-millisecond either way, below) is small next to `/v1/decide`'s own per-request overhead for a
`"choice"` question. The same input sent five times in a row, and again after a full server process restart,
answered identically in every mode tested (including which side served it).

Backend-only comparison (`benchmarks/learn_backend_bench.py`, same L40S node's CPU, synthetic data, 3,000 rows,
256 dims, 100 boosting rounds, one row at a time):

| | lightgbm | xgboost |
|---|---|---|
| train time | 5.66 s | 9.67 s |
| predict p50 (one row) | 0.051 ms | 0.254 ms |
| predict p99 (one row) | 0.066 ms | 0.367 ms |
| peak RSS during training | 119 MB | 177 MB |

LightGBM remains the faster, lighter-weight booster on its own; XGBoost is not a speed upgrade over it. What
XGBoost adds is a second, real, interchangeable implementation behind the same artifact format and the same
router -- comparable accuracy on both tasks above, and router-level latency indistinguishable from LightGBM's
once HTTP and request handling are in the measurement, which is what a caller of `/v1/decide` actually pays.

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

## Stage 4: switching traffic between a student and Prismyra

Stage 1/2 (above) only ever logs and trains offline; nothing it does can change what a caller receives.
Stage 4 (`prismyra.learn.router`, `.controller`, `.students`) is the part of DISTILL-RL-DESIGN-v2.md section
4 that can: for a tag that opts in, some fraction of its traffic is answered by its own distilled student
instead of Prismyra, with the device skipped entirely for the fraction that is. A tag that does not opt in
(no `"router"` field) is unaffected by any of this -- it keeps meaning exactly what it meant in stage 1/2.

### Registering a router

```json
{"task": "snake-move-up-v1", "question": "...", "options": ["up", "down", "left", "right"],
 "kind": "choice", "normalize": "none", "retain_days": 30, "keep_hidden": false,
 "eval_set": "evals/snake-move-up-v1-gold.jsonl",
 "router": {"mode": "ramp", "student_path": "students/snake-move-up-v1.json",
            "ramp_step": 500, "ramp_interval_s": 300, "ramp_max_ratio": 10000,
            "rollback_min_match_rate": 0.9, "rollback_max_drift": 0.3}}
```

`student_path` must point at a `StudentArtifact` (`python -m prismyra.learn.fit`'s own `--out`) whose
`admitted` is `true` -- loading one that is not admitted is a refusal at server startup
(`prismyra.learn.students.StudentError`), the same "fail loudly, not silently" discipline the rest of this
package already keeps. `mode` is one of:

| mode | what decides the route | what the audit lane checks |
|---|---|---|
| `ab` | a fixed `ratio` (per 10,000), never changed by this package | a sample of both directions, for monitoring only |
| `ramp` | `ratio` raised by `ramp_step` every `ramp_interval_s`, while the audited comparison keeps passing | same as `ab`, and what lets ramping continue |
| `shadow` | never serves the student; `ratio` is unused | always -- every request, both sides, every time |
| `confidence` | the student's own top probability against `tau`; `ratio` (if set) is an extra eligibility gate | a sample of the student-served subset |
| `bandit` | the controller's own UCB1 pick among `bandit_arms` (ratios), re-evaluated every `ramp_interval_s` | the audited sample attributes reward to whichever arm was active |

`RouterConfig`'s full field list -- all but `mode`/`student_path` optional, every default named once in
`prismyra.learn.spec` -- is `ratio`, `tau` (required for `confidence`), `bandit_arms` (required for
`bandit`), `ramp_step`/`ramp_interval_s`/`ramp_min_ratio`/`ramp_max_ratio`,
`audit_min_fraction`/`audit_max_fraction`, and `rollback_min_match_rate`/`rollback_max_drift`/
`rollback_cooldown_s`.

**Moving one tag between modes across a restart.** `prismyra.learn.controller.Controller`'s state (`ratio`,
among other things) is keyed by tag name alone and persists under `--learn-log-dir`, independent of which
`mode` wrote it. A tag moved from `"shadow"` (whose own `ratio` is unused and defaults to `0`) straight to
`"confidence"` on the same log directory inherits that `0` rather than starting from `confidence_ratio_gate`'s
"unset means no gate" reading, which only applies to a tag with no saved state yet -- the student then never
clears `eligible = bucket < ratio` regardless of `tau`. Set `"ratio": 10000` explicitly on the new mode's own
entry when switching, rather than relying on the field's default.

### The determinism contract

> 経路は決定的にする: `hash(tag, 正規化した入力) mod 10000 < ratio` かつ、生徒の最大の確率が `tau` 以上なら
> 生徒、それ以外は Prismyra。同じ入力と同じ表の版なら、いつも同じ経路を通る。

`prismyra.learn.router.bucket_for` is exactly this hash (SHA-256, not Python's own salted `hash()`), and
`decide()` is a pure function of `(tag, normalized_input, ratio, audit_fraction, student's own prediction)`
-- the same four inputs always produce the same `RouteOutcome`. What moves `ratio`/`audit_fraction` over
time (`ramp`'s own climb, `bandit`'s own arm choice, a rollback cutting either back down) is
`prismyra.learn.controller.Controller`'s job, never the per-request decision's own: a "表の版" is one
`(ratio, audit_fraction)` snapshot, named by `prismyra.learn.router.params_version` and carried on every
logged row (`route_params_version`), so a later reader can tell whether two rows were routed under
comparable conditions.

### The audit lane, and how a rollback fires

Independent of the primary routing decision, a second deterministic hash (`bucket_for(..., salt="audit:")`)
decides whether this request is also checked against the side that did not serve it: a student-served
request gets Prismyra's own answer for comparison (paying the device, for that one request); a
Prismyra-served request gets the student's own cheap, CPU-only prediction. `audit_fraction` is sized by
`Controller._audit_fraction_locked` to target about 20 audited requests per `ramp_interval_s`-sized window,
shrinking as a tag's own traffic grows (so a busy tag is not over-audited) and widening toward
`audit_max_fraction` for a cold one (so a brand-new tag is never left unauditable).

Every tick (at most once per `ramp_interval_s`, on whichever request happens to arrive after that much time
has passed -- no background thread), `Controller` checks the audited window's agreement rate against
`rollback_min_match_rate` and the accumulated drift histogram (see below) against `rollback_max_drift`
(`features.population_stability_index` between a frozen reference window and the current one). Either
failing cuts `ratio` to `ramp_min_ratio` (0, unless raised) and halts any further raising for
`rollback_cooldown_s` -- `ab`'s own fixed ratio is not exempt from this: a flat split that regresses is
still a regression.

### What a routed response carries

A response to a tagged, routed question or decision item gains two fields nothing else does:
`served_by` (`"prismyra"` or `f"student:{version}"`, `prismyra.learn.students.student_version`'s own name
for the exact artifact that answered) and `request_id` -- the same id `POST /v1/feedback` (below) and the
logged `Experience` row both use. An untagged request, and a tagged request whose tag has no `router`,
gain neither field: the response is unchanged from stage 1/2.

### `POST /v1/feedback`: a gold label or a reward, after the fact

DISTILL-RL-DESIGN-v2.md section 2 always described "報酬かラベル (あれば)" as collected "out of band"
without saying how. This is that path:

```
POST /v1/feedback
{"request_id": "<id a routed response carried>", "label": "up"}
-- or --
{"request_id": "<id>", "reward": 0.73}
```

Exactly one of `label`/`reward`; `prismyra.learn.hook.LearnHook.submit_feedback` refuses both or neither.
Written to a separate bounded queue (`prismyra.learn.experience.FeedbackLog`), off the request path, the
same drop-rather-than-block contract every other sink in this package keeps. 404 when `--learn-spec` was
not given at all; there is nowhere for this to join into.

At training time, `prismyra.learn.fit.run`'s own `feedback_dir` parameter (`--feedback-dir` on the CLI)
joins every logged row against a feedback record by `request_id`. A matched row trains on the gold label
directly -- one row, weight 1.0, for that option -- instead of the teacher's own (possibly wrong)
probabilities for the same row (`prismyra.learn.fit._soft_rows`). A `label` outside the tag's own option
set is dropped, not guessed at; a `reward` (a scalar, not a class) is left for a later stage -- turning one
into a classification target is a different, unmeasured question.

### Privacy: `retain_context`

A tag's `"retain_context"` (optional; default `"full"`) controls how much of a request's own `context`
survives into its logged `Experience`, independent of whether that tag routes any traffic at all:

| `retain_context` | what is stored | what is not |
|---|---|---|
| `"full"` | the raw context, exactly as stage 1/2 always stored it | -- |
| `"features_only"` | the already-vectorised feature vector (`precomputed_features`), from the tag's own loaded student's `Vectorizer` -- only once that student exists | the raw context |
| `"none"` | nothing but the drift histogram (`feature_summary`, below) | the raw context, and no `precomputed_features` either |

`feature_summary` -- a small (32-bucket), unweighted, hashed token-count histogram
(`prismyra.learn.features.feature_summary`) -- is kept regardless of `retain_context`: it is lossy (it
cannot be un-hashed back into text) and it is what `Controller`'s drift check compares across time, so a
tag set to `"none"` for privacy still feeds the one safety mechanism that depends on seeing *some* signal
about what is being asked.

### Hot-swapping a student: `POST /v1/learn/reload`

```
POST /v1/learn/reload
{"task": "snake-move-up-v1"}
```

Re-reads `student_path` from disk and swaps the in-memory `StudentHandle` the router uses -- no restart.
The operator's own responsibility is writing a new, already-*admitted* artifact to exactly that path first,
atomically (a temp file in the same directory, then `os.replace`, never a partial write a reader could
observe mid-write) -- this endpoint never fits anything itself, and a not-admitted artifact at that path is
refused the same way the initial load at server startup is.

### Idle-time retraining: `prismyra-learn-idle`

A standalone console script, not a thread inside `prismyra.server` (DISTILL-RL-DESIGN-v2.md section 5's own
"独立した段階として作る"): it polls the server's own `GET /stats` over HTTP, and only when the queue (and,
with `--batcher`, the batcher's own queue) has been empty for `--idle-polls-required` consecutive polls does
it consider refitting one due tag -- due meaning enough new experience (`--min-new-rows`) has accumulated
since the last fit, and enough wall-clock time (`--retrain-interval-s`) has passed. One tag per idle window,
never more: a long queue of due tags does not monopolise an idle window a caller's own traffic just handed
back.

```
prismyra-learn-idle --server-url http://127.0.0.1:8000 --learn-spec learn.json \
  --backbone <model id> --nice 19 --num-threads 1
```

`--nice` (default 19) lowers this whole process's own OS scheduling priority; `--num-threads` bounds how
many CPU threads one `fit.run()` call is allowed. Both exist because this is CPU contention a background
job must not impose on a GPU server sharing the same box -- `/stats`-polling alone answers "is the device
idle", not "is it safe to also burn CPU here", which is what these two knobs bound instead. A fit that is
not admitted changes nothing already serving; a fit that is gets written atomically to `student_path` and
handed to the live server via `POST /v1/learn/reload`, exactly as an operator driving both endpoints by
hand would.

`--num-threads` alone is not enough to bound the CPU cost, and this was found and fixed on real hardware,
not in review: `fit.run()`'s own numpy usage (`_soft_rows`'s `np.asarray`, `Vectorizer.transform`) is built
against a BLAS backend (OpenBLAS/MKL) that picks its *own* thread count from `OMP_NUM_THREADS`/
`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS`, independently of
LightGBM's own setting. Measured on a real L40S node (48 vCPUs): with `--num-threads 1` and these left at
their default, one fit call drove 23-24 cores above 20% utilisation at once; `main()` now calls
`cap_blas_threads` (setting all five, via `os.environ.setdefault`, before anything that could import numpy)
immediately after parsing arguments, which brought that down to the single core `--num-threads 1` actually
promises.

**Measured impact on serving, with that fix in place** (real L40S, `Qwen/Qwen3-0.6B`, a continuous
closed-loop `/ask` load, 5 idle-off and 5 idle-on 20-second windows alternating, a synthetic tag with 3,000
logged rows so one fit call takes several seconds of real CPU work):

| | off | on (whole window) | on (only while the fit call itself ran) | budget |
|---|---|---|---|---|
| p50 | 63.68 ms | +0.12% | +0.02% | +2% |
| p99 | 71.95 ms | +4.08% | +4.50% | +5% |
| throughput | 15.58 req/s | -0.07% | -0.12% | - |

Both within budget. A second run forced continuous back-to-back fitting through the entire "on" window
(`prismyra-learn-idle` itself never does this -- one fit per idle window reached, then it waits for the
next poll) as a deliberately pessimistic upper bound: p50 +0.09%, p99 +3.23%, still within budget. Under a
sustained load that kept the worker's own queue non-empty the whole time, `prismyra-learn-idle` attempted
zero fits across 15 polls (`is_busy`'s own gate stayed tripped throughout) and the process's own CPU usage
stayed low (3.8% mean, 9.3% max of a 48-core machine) while it did nothing but poll -- see
`benchmarks/idle_latency_bench.py` for the harness and the project's own run log for the full numbers.
Testing this at 16 and 64 questions per request hit an unrelated, pre-existing incompatibility between the
engine's multi-question branch path and the `transformers` versions this project's own range allows
(reproduces with `--batcher` on or off, on both the small and the 35B reference model); the single-question
measurement above is what this round could complete.

## The read/write interfaces: `ExperienceSink` and `ExperienceSource`

Where a tagged request's `Experience` goes, and where `prismyra.learn.fit`/`.idle` read it back from, are
each one small interface (`prismyra.learn.experience.ExperienceSink`/`.ExperienceSource`), not a fact baked
into `LearnHook`'s own logic. Every sink and every source agree on one schema -- the same
`Experience.to_json()`/`.from_json()` shape, versioned (`schema_version`, currently 2; a row with none is
read as 1, stage 1/2's own shape, with every stage-4 field defaulted to absent) -- so switching which one a
server uses never changes what a later reader has to parse.

| sink | writes to | enabled by |
|---|---|---|
| `LocalExperienceLog` (default) | one JSONL file per `(tag, UTC date)`, local disk | nothing extra -- this is stage 1/2's own, unchanged |
| `RayExperienceLogActor` | the same, inside a Ray actor | `--learn-backend ray` |
| `prismyra.learn.otel_sink.OtelExperienceSink` | an OTel log record per experience, OTLP to a collector | `--learn-otel-endpoint <url>` (needs `pip install "prismyra[otel]"`) |

More than one at once is `prismyra.learn.experience.MultiSink`: `--learn-otel-endpoint` *adds* the OTel
sink alongside whichever `--learn-backend` chose, it does not replace it. Every sink keeps the same
contract stage 1/2 established -- a bounded queue, a full one drops rather than blocks, writing happens off
the request path.

| source | reads from | ships in this package as |
|---|---|---|
| `LocalExperienceSource` (default) | the same on-disk layout `LocalExperienceLog` writes | `prismyra.learn.fit.run`'s own default when no `source=` is given |
| `prismyra.learn.http_source.HttpPullExperienceSource` | a generic paginated `GET {url}?task=...&since=...&page=...` contract (see that module's own docstring) | `run(..., source=HttpPullExperienceSource(url))` |

A destination-specific reader (CloudWatch Logs, a Prometheus range query) is meant to be built *on top of*
the same generic pull contract `HttpPullExperienceSource` already implements, not as a reason to add that
destination's own SDK as a hard dependency of this package -- see `prismyra.learn.http_source`'s module
docstring for exactly where that line is drawn. Whichever source feeds it, `POST /v1/feedback`'s own
`request_id` join (above) works identically: `prismyra.learn.experience.load_feedback` and
`prismyra.learn.fit._soft_rows` both operate on `Experience.request_id` alone, never on how the row reached
them.

## `prismyra.learn` and `prismyra.telemetry` stay separate streams

`prismyra.telemetry` (`docs/OBSERVABILITY.md`) is this project's *general* observability layer -- server
metrics, traces, logs about operation, never about a request's own content. `prismyra.learn`'s experience
log is the opposite: it exists specifically to carry a registered tag's request content, and only when
`--learn-spec` turned that on. The two can share OTel SDK bootstrap machinery when both happen to be
enabled (`OtelExperienceSink` can be pointed at a `logger_provider` someone else already built), but never
data: a different OTel `Logger` name (`"prismyra.learn.experience"` vs `prismyra.telemetry`'s own
`"prismyra.telemetry"`), a different enable flag, no code path that reads one's own data into the other's.
Turning on `--otel` alone emits zero `Experience` rows; turning on `--learn-spec` alone emits zero spans or
metrics through `prismyra.telemetry`. `tests/test_telemetry_isolation.py` is what checks both directions,
together with the untouched stage 1/2 case (neither on) and the case where both are.
