# Roadmap

What is measured today, where each line of work is stuck, and what the next step is. Every number here
names the condition it was measured under; anything not measured is called out as not measured, not
implied. This page does not record how any of it happened -- only where things stand and what is next.

**Status.** `v0.5.1` is the published line and is intended to stay the one shipped build for now. A
fp8 candidate described under [Accuracy](#accuracy) below improves on two specific weak spots of
`v0.5.1` and has passed its own checks, but is not itself a release plan; it is kept here as a
documented candidate for whenever the project next revisits the published line.

## Speed on a single question

One question about one document, read once, is the hardest case to make fast: there is nothing to
share across requests. Under a fresh document on every call -- the condition that gives neither this
project's checkpoints nor a competing engine any benefit from a warmed cache -- `fp8` on one L40S
answers a single question about a 3,446-token document in roughly 150 ms, and at that condition
already beats the two external checkpoints it is compared against in `recipes/decision-lora/README.md`
at every question count measured, including one question. An earlier internal target near 80 ms came
from a comparison that (unintentionally) gave the competing engine a warmed key/value cache and not this
project's own call; that comparison has been corrected in the recipe document, and chasing that
particular number is no longer the goal. The remaining interest is in where the ~150 ms itself goes, on
its own terms.

Of that single forward pass, by measured per-kernel device time on one such read: the routed-expert
layer (MoE) takes about 44.6 ms, the FP8 dense projections about 25.7 ms, and the recurrent (gated delta
rule) layers about 17 ms. The dense-projection figure is the one place this project's own kernel is
slower than a comparable kernel in a well-known serving library on the same hardware (roughly 33 ms
against 22 ms in that comparison); the gap has been traced to how the matrix scale is laid out in memory
rather than to any numerical difference, and nothing has closed it yet.

Three attempts to let `torch.compile` fuse the backbone into fewer, larger kernel launches have each
been measured and none has shipped:

- A first attempt at compiling the whole backbone recompiled per-layer instead of once, which made it
  slower than the uncompiled path by about a third; the actual cause (one function body shared across
  many layer instances forcing a separate specialisation per layer) was found and fixed with one
  configuration change.
- After that fix, the hand-written kernels this project relies on for exact, reproducible numbers had to
  be marked so the compiler would not try to trace through them. Doing that the simple way makes the
  compiler split the computation into over forty separate fragments, each paying its own bookkeeping
  cost on every call; measured with the compiler's kernel-fusion stage turned off (to isolate this cost
  specifically), that bookkeeping alone made the compiled path 52% slower than the uncompiled one, and
  turning the fusion stage back on crashed with an internal compiler error.
- Registering those same kernels as proper custom operators (so the compiler can treat each as one node
  instead of a fragment boundary) reduced that penalty but did not remove it -- the compiled path is
  still 25% slower than the uncompiled path with fusion isolated out -- and, measured independently,
  registering the operators this way makes the ordinary, uncompiled path 6.7% slower on its own, which
  disqualifies it as a plain win regardless of what compilation does with it.

A separate, already-implemented change fuses the per-layer residual addition into the following
normalisation step, using a well-known serving library's own fused kernel. It is not bit-exact with the
unfused path (expected: fusing changes how many times a number is rounded), so it was checked against
the accuracy bar this project applies to any such change, and passed on one supported card. On the
other supported card, in the specific case of a request asking more than one question of a document
through the shared-storage request path, it was found to break an existing bit-for-bit guarantee between
that path and the plain one-question path on the project's own fixed regression check (about 98% of
those paired answers moved instead of the usual roughly 1%, on one of the two supported cards). It has
not shipped, and is off by default.

A separate opt-in mode that trades the determinism guarantee for speed (fusing the attention
rotation and output-gate math into a `torch.compile`'d region, rather than the hand-written exact
kernels used by default) is implemented, evaluated against a held-out accuracy check with its own
pass bar, and not yet merged into the published line.

Extending the fixed set of pre-recorded replay lengths (currently up to 2,048 tokens) past that ceiling
was checked before building it: at the current ceiling the saving from replaying a recorded pass instead
of running it fresh is already down to about 3% on one supported card and under 1% on the other, so a
document longer than the ceiling -- which is the common case for this task's documents -- would gain
close to nothing from extending it, and it was not built.

**Next step.** No version of compiling the backbone has yet beaten the hand-written kernel path; the
next candidate worth trying is wrapping every hand-written kernel as a custom operator the way a
well-known serving library does it, in a way that does not pay the dispatch cost on the ordinary,
uncompiled path (today's attempt applied the same wrapped calls unconditionally). Shipping the
residual-fusion change is blocked on finding a way to restore the broken multi-question guarantee on
the affected card, most likely by moving the rounding back into a hand-written kernel this project
controls, the way the rotation/gate fusion already does.

## Long documents

The longest supported document is 65,536 tokens; a document twice that length (131,072 tokens) was
attempted and did not work.

A real bug was found and fixed first: the check that decides whether a document fits in memory predicted
free memory instead of confirming it, so once a process's memory pool grew to answer one long document,
later documents were rejected using a stale, never-rechecked prediction. The fix (confirm free memory
before rejecting, the same way a different memory check elsewhere in the engine already did) is merged
and keeps the pool from growing without bound. It does not, by itself, make a second 65,536-token
document fit on the smaller of the two supported cards (32 GB): after the fix, that card is about 0.4 GiB
(roughly 5%) short of what a second such document needs, which is a real capacity limit of that
card/checkpoint/length combination, not a software defect.

Extrapolating the measured cost of a 65,536-token document (not a fresh measurement at 131,072 tokens,
and stated as an extrapolation only) suggests a document twice as long would need on the order of 17 GiB,
which neither supported card reaches with headroom to spare. The limiting resource is not the
attention layers' key/value cache, which is small and already discussed in
[docs/PERFORMANCE.md](PERFORMANCE.md); it is the recurrent layers' own working memory, which the
project's existing memory-saving techniques (splitting a long read into chunks, and a separate,
experimental FP8 key/value-cache flag that is not on the published line) do not reduce enough to close
this gap -- the experimental flag only touches the attention layers, and extrapolating its own
already-measured saving at shorter lengths puts it far short of what 131,072 tokens would need.

On a long-document variable-tracking check at 65,536 tokens (a 25-question sample, explicitly too small
to call any of the following a significant difference): the published `fp8` and `nvfp4` checkpoints tie
at 64%, and the un-adapted base checkpoint scores 60% on the same sample -- so this project's own
training did not damage the base model's ability at this task, and the base model itself was never
strong at it. A competing model's publicly reported 94% on a similar task is more likely a property of
that model's own architecture or training than something this project's quantisation or training
degraded; unlike that competing model, which the same note records losing badly when quantised (94% down
to 48%), this project's own checkpoint shows no such quantisation cost on this check.

**Next step.** No change available today (chunked reading, the experimental key/value-cache flag) closes
the gap to 131,072 tokens; doing so would need a reduction in how much working memory the recurrent
layers hold per token, which is an architecture-level change, not a configuration one.

## Accuracy

A fp8 candidate exists that adds BoolQ and QNLI examples as self-distillation anchors, aimed
specifically at a known weak spot of the published line (BoolQ accuracy on one internal regression
check had drifted down as other anchors were added). It has been checked against the published line on
a previously unused evaluation collection built specifically so this check would not be contaminated by
whichever data had already been used to pick the candidate:

- Across the whole 10,000-question collection, the candidate's pooled score does not move measurably
  against the published line (within a few tenths of a point either way, with the interval excluding
  anything worse than half a point).
- Within the collection's own BoolQ-style subset, the candidate's point estimate is higher, but that
  subset's own confidence interval touches zero, because only a couple of hundred BoolQ-style questions
  survived this collection's own de-duplication against earlier training data -- not enough on their own
  to call the improvement clearly established by a strict reading of the pass bar. A separate,
  roughly three-times-larger same-task sample from the project's development evaluation set clears that
  same bar outright in the same direction, and a third, independent check agrees in direction and size;
  taken together the improvement looks real, even though the smallest of the three checks alone lacked
  the statistical power to say so by itself.

An nvfp4 (4-bit) version of this same candidate has not been built or evaluated.

On the long buried-context evaluation (ordinary RACE questions answered after being buried inside
several thousand tokens of other articles), the published line still trails a competing model on this
project's own head-to-head comparison in `recipes/decision-lora/README.md` (89.17% against that model's
94.27% on one such slice; the project's own calibration-quality figures, Brier score and expected
calibration error, are also slightly behind the same competitor on the same comparison). Several data
recipes aimed directly at this gap have been tried and found not to move it reliably (synthetic buried
text, financial-document padding, auxiliary multiple-choice data), each moving some other evaluation
slice instead, for better or worse. One recipe -- program-generated data combining variable-tracking and
multi-step fact-chaining -- reliably closed a related long-context gap instead (bringing this project's
own score on a long-context variable-tracking check from the low 60s up to parity with the same
competitor, in the mid-90s, on two independent training runs) without closing the buried-context gap
itself. The newest BoolQ/QNLI candidate above also moves buried-context in the right direction on both
of its two training runs, by a small amount confirmed to sit outside the range normal training variation
produces on this metric -- worth noting, though it was not the reason that candidate was selected.

Calibration is shipped, off by default: a single scoring-time temperature correction, applied only to
four-choice questions (the one question-shape with enough rows in the project's own benchmark, just
over half of it, to trust fitting a correction against). Correcting every question uniformly, or by
task/source instead, was tried first and found to make the pooled result worse, because a correction
that helps one small group overfits noise once every group is pooled back together. On a fresh
evaluation collection the four-choice correction did not clearly carry over -- the overall calibration
error barely moved, improving on some task families and getting worse on one -- and the project's own
documentation says plainly that the within-benchmark improvement does not clearly generalise to an
unseen collection. One check from this change's own release process is still outstanding: its
hardware-specific regression check has passed on one supported card and has not yet been run on the
other, for lack of a free slot on the one attempt made.

**Next step.** Continue the self-distillation search specifically for the buried-context gap (the
newest candidate's small, two-for-two positive move is a lead, not yet a fix); build and evaluate an
nvfp4 version of the newest BoolQ/QNLI candidate; run the still-outstanding hardware regression check
for calibration on the second supported card.

## Paths that answer the same question differently (`--batcher` on or off)

With the optional request-batching flag on, a single-question request is answered through a different
computation than the same question asked without it or through the plain library call -- a known,
intentional fork rather than a bug: the batching path needs every request to be able to share
underlying storage with others, which only exists once a request is treated that way from the start, and
a lone question cannot be special-cased without making its answer depend on what else happens to arrive
in the same narrow window, which this project does not otherwise allow.

This has been measured candidly and written down in `README.md`, `docs/PERFORMANCE.md` and the model
cards: on a development set of real HTTP requests, about 1 in 60 answers flip their decision between the
two paths, the largest single probability move seen is about a quarter, and roughly 8% of answers move
more than one existing regression test's fixed tolerance for this -- while aggregate accuracy across the
same set does not move by a statistically detectable amount. What is and is not promised is now stated
plainly: bit-for-bit repeatability for a fixed input and a fixed way of starting the server is unaffected
either way; agreement between the two paths on one single question's exact decision is only guaranteed
within the original benchmark that was first measured, not promised more broadly.

Two ways to remove the difference were evaluated and both rejected. Routing the single-question path
through the batching path's shared storage would break an existing, already-relied-on bit-for-bit
guarantee for multi-question requests. Special-casing "this request happens to be alone right now"
inside the batching path would make the answer depend on arrival timing, which is a form of
non-determinism this project does not otherwise accept for a fixed input and fixed startup.

**Next step.** None planned under the current design. This would only be revisited if the
single-question read path itself is redesigned for some other reason (for example, as part of the speed
work above).

## A small, fast student model

A small decision-tree model (LightGBM or XGBoost, CPU-only) first imitates this project's own answers
(distillation), and is meant to be improved further afterwards with reinforcement learning once its own
feedback loop exists.

Distillation (the first stage) is implemented and shipped behind an explicit admission bar: a candidate
student must match this project's own answers closely enough, both on an evaluation slice and on rows
never used to fit it, before any live traffic is ever routed to it. Two reusable fixes shipped alongside
it: a bug where switching between traffic-routing modes let a stale ratio leak from one mode into the
next is fixed, and a second way of turning text into features (character-level rather than word-level)
was added for tasks that do not split cleanly into words. Once a student is admitted and actually
routed to, it answers in about 2 ms against roughly 54 ms for reading the question with the full model,
measured over real HTTP.

Two tasks have been tried against the real admission bar. A game-move-classification task passed
outright on both tested tree libraries. A 53-option intent-classification task failed clearly: the
student's own held-out accuracy sat around a quarter against the full model's own accuracy of about a
third on the same questions, nowhere near the required agreement, across both tree libraries and both
ways of turning text into features tried (four combinations, none passing), and the gap did not close
predictably as more training rows were added.

The second stage -- reinforcement learning on top of an already-distilled student, using the feedback a
caller reports back -- exists only as a design today; no code for it has been written.

**Next step.** Implement the second (reinforcement-learning) stage. Separately, the failed
intent-classification task needs either a different task where the signal is actually learnable from
simple text features, or a different feature representation; more training rows alone did not fix it.

## Judging safety, and reading a document once during training too

Two directions are design-only today, part of this project's longer-range distillation and
reinforcement-learning design, and not yet implemented.

**A safety judgment.** A recent published method (arXiv:2610.07935) uses only a written policy -- no
labelled examples -- to generate hard safety cases, train on a teacher's demonstrations, and reinforce
against a rubric; reported results on public safety benchmarks cut an unsafe-agent-action rate from
79.1% to 3.8% and a harm-benchmark score from 22.6 to 14.8, without reducing ordinary ability. The shape
fits this project well: reading a conversation once and asking many policy-clause questions as parallel
branches is already what this project does, and a deterministic, bit-reproducible probability is
auditable in a way a method that reasons out loud is not. The acknowledged limitation: the published
method reasons before answering, and this project's one-pass read-out would have to learn from a
teacher's already-reasoned-through label instead, which may lose accuracy on judgments that genuinely
need deliberation. Any training data for this would be built only from commercially licensed sources,
with teacher labels from models already approved for that use, and only kept where two independent
teacher models agree.

**Next step.** Before building any training data: measure the current, untrained checkpoint directly
on a public safety benchmark (accuracy and calibration), and compare against existing purpose-built
safety-guard models, to decide whether training data is worth building at all.

**Reading one document once during training, the way inference already does.** During training, when
many training rows share the same underlying document, read that document's shared portion once and
treat the rows as parallel branches -- mirroring the inference-time mechanism this project already has,
instead of re-reading the document from scratch once per row. A recent published method (arXiv:2609.33848)
demonstrated this kind of shared-prefix training to cut redundant computation. Open questions already
named in the design: keeping branches from attending to each other, copying the recurrent layers' state
per branch the same way the inference-time mechanism does, summing gradients correctly across branches
that share a prefix, and keeping the whole thing bit-deterministic; how much of the existing
inference-time implementation can be reused has not been checked.

**Next step.** A CPU-only first measurement, before any training code is written: on an existing
training dataset, count how many training rows actually share a document, to put a hard upper bound on
the possible saving.

## Comparisons still outstanding

Two models from outside this project are part of its standing comparison set and have not been run yet.

**A 26B decision model with a Gemma-4-based LoRA adapter.** The model's own card requires a pre-release
build of a serving library plus a bundled patch for running a LoRA adapter against a tied output layer.
An earlier attempt reproduced a shape-mismatch crash on load with that adapter active, regardless of how
the model is split across GPUs; the base model loads without the adapter, so the mismatch is specific to
this adapter against the serving library version available here. Not retried this round, since the
cause already points at a version mismatch between what the card assumes and what is installed, rather
than at anything a retry under the same versions would change.

**A 27B decision model requiring two large GPUs.** Blocked twice, for two different reasons. First
attempt: building a throwaway Python environment on shared network storage was too slow to finish in a
reasonable time; a fix (building it on local disk instead) was prepared. Second attempt, to try that
fix: no two-GPU slot was free anywhere in the shared cluster for over an hour, and a parallel attempt to
reserve capacity elsewhere also failed within its own time budget. The environment-setup fix therefore
remains untested, and the model remains unmeasured, for lack of capacity rather than for any problem
with the approach.

**Next step.** Retry the 27B model once two large GPUs are free at the same time; the setup and
smoke-test scripts are already written and ready. The 26B model needs someone to read the serving
library's own LoRA-layer code to find exactly where its shape convention diverged from what the card
assumes, before a retry is worth the GPU time.

## Observability

Operational metrics, traces and logs about the server itself (`--otel`) are implemented, off by default,
and kept deliberately separate from the learning/experience log in [docs/LEARN.md](LEARN.md). Currently
covered: request counts and durations by endpoint, rejected-request counts by status code, one trace
span per request reconstructed from already-measured timings, every numeric field of the stats endpoint
exported as a gauge, and an always-on Prometheus text endpoint independent of `--otel`.

One tradeoff is already written down rather than hidden: a trace span is built after a request is
already answered, from timings already measured, rather than kept live while the request is in flight --
deliberately, to avoid touching request-handling code paths this project has measured and tested
carefully elsewhere. No concrete extension beyond today's coverage has turned up a measured need yet.

**Next step.** None identified from measurement so far. If a live (in-flight) version of the same trace
is ever wanted, it carries more implementation risk than the reconstructed one in place today, and
should be judged against that risk before being built.
