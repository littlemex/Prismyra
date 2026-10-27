# Changelog

## 0.4.10

Three fixes to stage 4, found running the student path on real hardware a second time:

- **`prismyra.decide.choice_labels` is now cached per tokenizer.** It was being recomputed on every single
  `/v1/decide` request with a `kind: "choice"` item, including one a router answers from the student alone
  without ever touching the engine -- on this project's own real tokenizer the uncached cost was 21.46 ms
  (p50), almost all of a measured 43 ms student-served round trip. Cached, it costs 0.33 microseconds.
  Memoised with a `weakref.WeakKeyDictionary` keyed on the tokenizer object itself, so a garbage-collected
  tokenizer's cache entry cannot be served to an unrelated object that happens to reuse its `id()`.
- **A tag's router can now switch `mode` without carrying over a stale `ratio`.** Running `shadow`, then
  `confidence`, against the same `--learn-log-dir` used to leave `confidence` reading back `shadow`'s own
  `ratio=0` literally, instead of `confidence_ratio_gate`'s "unset ratio means no extra gate" rule -- a rule
  that previously only applied to a brand-new tag. `prismyra.learn.controller.Controller` now records which
  mode last initialised a tag's state, and reinitialises it through `_initial_ratio` the moment the mode
  resolving it changes.
- **A text-feature tag can now ask for character n-grams instead of whole words.** `"text_tokenizer": "charN"`
  on a tag's own entry (default unchanged, `"words"`) selects `prismyra.learn.features.char_ngrams` for the
  hashed-TF-IDF branch -- self-implemented rather than a new dependency. Measured on a new real task
  (`banking77-intent-v1`, 53-way intent classification): it badly hurt LightGBM and left XGBoost roughly
  unchanged, a backend-dependent finding, not a universal improvement.

See [docs/LEARN.md](docs/LEARN.md#round-2-2026-10-27-same-day-where-the-43-ms-actually-went-a-reused-task-that-really-admits-and-character-n-grams) for the real-machine measurements behind all three, a new task measured honestly as not
admitted (`banking77-intent-v1`), and a reused task (`snake-move-up-v1`, learn3/learn4's own structured
features, unmodified) that admits for both student backends on a fresh sample.

## 0.4.9

Stage 2's student can now be trained and served by XGBoost as well as LightGBM. A tag picks one with its
own optional `"student"` field (`"lightgbm"`, the default and unchanged behaviour, or `"xgboost"`);
`python -m prismyra.learn.fit --student-backend` overrides it for one run. XGBoost is an optional extra
(`pip install "prismyra[learn-xgb]"`) and is never imported unless a tag asks for it. Both backends train on
the same soft-label rows, are pinned to a fixed seed and thread count, and promise the same contract: the
same input and the same artifact give the same answer, including across a server restart. A
`StudentArtifact` now records which backend and library version produced it (`backend`/`backend_version`);
an artifact written before these fields existed reads as `"lightgbm"`, unchanged. See
[docs/LEARN.md](docs/LEARN.md#choosing-a-student-backend-lightgbm-or-xgboost) for a real-task measurement
against both backends, over real traffic, on a real engine.

## 0.4.8

Stage 4 of the learning layer ([docs/LEARN.md](docs/LEARN.md)) can now switch a registered tag's live
traffic between Prismyra and its own distilled student. Five routing strategies share one deterministic
rule (`hash(tag, normalized input) mod 10000 < ratio`): a fixed split, a ratio that ramps up over time, a
shadow mode that always runs the student for comparison but never serves its answer, a confidence threshold
that only lets the student answer when its own top probability clears a bar, and a bandit that picks among
a configured set of ratios by their own observed agreement rate. An audited sample of both directions feeds
a rollback: a tag's ratio is cut back to zero and held there for a cooldown period the moment the audited
agreement rate or the input's own drift (a population stability index over a hashed feature histogram)
crosses a configured bound. A routed response carries `served_by` and `request_id`; `POST /v1/feedback`
attaches a gold label or a reward to that id after the fact, and a labelled row trains on the label directly
instead of the teacher's own probabilities. `POST /v1/learn/reload` hot-swaps a newly admitted student
without restarting the process, and the new `prismyra-learn-idle` console script drives that loop on its
own: it polls the server's queue depth, waits for it to be idle, and refits one due tag at a time in a
separate, deliberately deprioritised process. None of this changes an unrouted tag's behaviour, and an
unrouted request remains byte-identical to this layer not existing at all.

The experience log's write and read sides are now each one small interface
(`prismyra.learn.experience.ExperienceSink`/`ExperienceSource`) rather than a fact built into how a server
logs or a student trains: a local JSONL file is still the default on both sides, an OTel log sink and a
generic paginated HTTP pull source are the second implementation of each, and more than one sink can run at
once.

A new `prismyra.telemetry` package adds optional, off-by-default observability for the server's own
operation -- request counts and latency, queue depth, rejected-request counts, and a span per request --
over OTLP to an operator's own collector, plus a dependency-free `/metrics` endpoint that always works. It
never carries a request's own content, and it shares no data path with the learning layer's own experience
log even when both are turned on at once.

## 0.4.7

An optional, off-by-default learning layer lets an operator register a closed, versioned
vocabulary of decision tags and have a request tagged only on an exact, normalized match --
never by the document itself. Without `--learn-spec`, this layer is never imported and the
serving path makes exactly the comparisons it always made; with it, a tagged request's
experience is logged in the background, after the response has already been sent, through a
bounded queue that drops rather than blocks a request.

`python -m prismyra.learn.fit` trains a small, deterministic LightGBM model offline from a
tag's logged experience. The default input goes through a hashed-TF-IDF representation; a tag
may instead register a hand-written feature builder for structured input (such as a flattened
game-board state), which reached a materially higher accuracy and agreement with the teacher at
every training-set size tried, already ahead of the hashed default's best result at the
smallest size tried. A student is admitted only if its accuracy on a gold-labelled evaluation
set clears a configurable fraction of the teacher's own accuracy on that same set, and its
one-sided confidence lower bound on held-out agreement with the teacher clears a configurable
threshold; both bars default to 90% and either can be overridden per tag.

`recipes/decision-lora` now mixes two licence-clear, public dataset families (sentiment, and
causal/multi-step reasoning) into the training pool at a small, fixed fraction, closing a
measured gap on sentiment judgments against a comparable general-purpose model with no measured
regression on the project's production-safety item set or any other held-out item at the
confidence levels used. An earlier attempt at a larger mixed-in fraction improved sentiment by
more but also regressed the production-safety set; halving the fraction kept the sentiment gain
and removed the regression. The reasoning gain pointed the same direction but did not reach
significance at the row counts tried.

## 0.4.6

The 22.5 GiB budget that fell short of the 16,000-token, 32-question case by a small margin in
the previous release now fits it at the engine's full default width, with no narrower,
slower configuration needed.

- A gated-delta-net layer's branch-pass state buffer, sized at the engine's full group width, is
  now freed the moment a request that only needs one such group is done reading it, instead of
  being held until the whole cache drops. This removes a steady climb in held memory across a
  long pass; on the supported checkpoint, that climb used to add about 2.2 GiB from the first
  layer read to the last and now adds about 0.45 GiB. A multi-group request is unaffected by this
  change and keeps the prior behaviour.
- The formula admission uses to budget that same layer's held state priced its recurrent half at
  the model's own compute dtype, but the framework always keeps that half in `float32` regardless
  of the model's dtype, so the formula was pricing it at half its real width. The fix prices each
  half at its own dtype instead of a flat multiplier, which stays correct whatever dtype the model
  computes at. This is a correction to admission's own arithmetic only; every answer this engine
  gives is unaffected and bit-identical to the version before this change.

## 0.4.5

A 16,000-token context and 32 questions at once now fit on a 24 GiB class card with no extra
configuration, and on a smaller, 22.5 GiB budget with one explicit setting.

- Two opt-in weight placements, `PRISMYRA_LM_HEAD=lazy` and `PRISMYRA_EMBED_TOKENS=lazy`, keep the
  output and input embedding matrices in host, pinned memory between requests instead of on the
  device, gathering only the rows a forward pass actually names. Together they free roughly
  1.9-2.1 GiB on the supported 36-layer checkpoint. Both are off by default and have no effect
  unless set.
- A context longer than 4,096 tokens is now read in 4,096-token pieces fed through the same cache
  one after another, instead of one call over the whole length; each piece leaves the cache in
  exactly the state the same tokens would have left it in read in one pass. Whether a given read
  splits is decided per request from the engine's own memory accounting; there is no flag to set.
- That accounting had a blind spot under a per-process memory cap: a process given a software
  fraction of a card smaller than the whole of it had that fraction enforced by the allocator, but
  neither the automatic-chunking decision nor the engine's own admission check ever saw it. A new
  function corrects the free and total figures for the per-process fraction a caller may have set;
  a process with no fraction set is unaffected.
- A branch pass's read no longer copies the context once per question. The read now shares the
  context's pages across every question's table instead of copying them into it, and each
  question privately holds only its own tokens plus a copy of the context's own partial last
  page, bounded by a small, fixed page size rather than by the context's length. Where the full
  default width already fits a budget, the pass answers about 6% faster than before; combined with
  a narrower width for a tighter budget, the free memory left over roughly doubles and the latency
  cost of that narrower width falls by about half.

A 22.5 GiB budget and the engine's full default width together still fall short of the
16,000-token, 32-question case by a small margin, for a reason unrelated to the pieces above and
not addressed by this release. Building the engine with a narrower width closes that gap at the
cost of answering some of the wider requests in more than one internal pass.
