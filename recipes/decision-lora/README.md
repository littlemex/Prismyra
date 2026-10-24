# Decision LoRA: training the read-out Prismyra serves, and folding it in before serving

Prismyra answers a typed question by reading the probability of each declared option's token at the last position of
a branch. This recipe trains a LoRA adapter on exactly that quantity -- the log score of the gold option's token,
softmaxed over the declared options only -- and then folds the adapter into the FP8 checkpoint ahead of time.
The served checkpoint has the same shapes, the same dtypes and the same routed experts as the base, so every kernel
Prismyra installs on the base is installed on the result and the request path does no extra work for the adapter.

## Why a merged adapter and not a served one

An adapter applied at request time adds two small matmuls to every adapted projection on every pass. Folding it in
removes them: the adapted matrices are dequantised, `W + (alpha / r) B A` is added, and the result is re-quantised with
the checkpoint's own scheme (e4m3, one scale per 128x128 block, stored as `weight_scale_inv`). The routed experts are
never adapted, so the MoE kernels see byte-identical expert weights.

`merge.py` was checked two ways. With an adapter whose `B` is zero, the re-quantisation error was 0.0 and every answer
on a 579-question set was identical to the base, with a largest probability difference of 0.0 -- which shows the
round trip is lossless, not that the arithmetic on a non-zero update is right. The trained adapter is the second check:
its gains below were measured on the merged checkpoint, so a wrong scale convention would have shown up as lost
accuracy rather than as the long-context rows moving by ten points.

## Training

The FP8 weights stay FP8 in memory. For training only, the block-FP8 matmul is replaced by "dequantise the block and
multiply in bf16", which autograd can differentiate, and a mixture-of-experts layer dequantises all its experts once
and runs one grouped matmul per projection. The weights themselves receive no gradient; only the adapter does.

Targets: the attention and gated-delta-net projections and the shared expert (250 modules, 19.2M parameters at rank
16). Peak memory was 40.7 GiB per GPU on 44 GiB cards with sequences up to 12,500 tokens.

```bash
torchrun --nproc_per_node 2 train.py --model Qwen/Qwen3.6-35B-A3B-FP8 --data data/train_v1_7k.json --out lora.pt
python merge.py /path/to/Qwen3.6-35B-A3B-FP8 lora.pt /path/to/merged
```

Then serve the merged directory like any checkpoint: `Prismyra("/path/to/merged")`.

A trained adapter is published with `python export.py lora.pt <dir>` (safetensors plus a config), and `merge.py` accepts
that directory in place of the `.pt`.

## Data

Every row is rendered the way the evaluation renders it: a context, then one lettered choice or a yes/no question.
`EVAL_ITEMS` lists the evaluation files every row is deduplicated against, and `DATA_DIR` is where rows are written.

```bash
export DATA_DIR=data EVAL_ITEMS=items/race150.json:items/boolq400.json:items/race40_bury10k.json:items/race40_bury7k.json:kev-transfer-v4-test.jsonl
python build_mixture.py
python subset.py
python build_long.py
python build_families.py
```

| Script | Rows | What |
|---|---|---|
| `build_mixture.py` | 16,681 | RACE train (short, and buried in 2k-12k tokens of other train articles), BoolQ, MMLU auxiliary train, SciQ, QNLI, tweet_eval offensive, three synthetic policy families |
| `subset.py` | 7,018 | the per-family subset that was trained (v1), written as `train_v1_7k.json` |
| `build_long.py` | 3,233 | RACE and BoolQ items buried in 4k-12k tokens of other train text |
| `build_families.py` | 2,300 | AG News, DBpedia, MNLI, CommonsenseQA, ARC-Challenge, OpenBookQA, HellaSwag; writes the union as `train_v3.json` |

Three families are held out of every script: emotion classification, paraphrase detection (PAWS), and policies with an
exception clause. No sentiment task and no paraphrase-detection task of any source is added (QNLI and MNLI are
entailment tasks), so "unseen family" stays unseen.

## Measured result (v1: `subset.py`'s 7,018 rows, one epoch)

One L40S per evaluation, 35B-A3B FP8 with the merged adapter, group 32. Each figure is compared with the base
checkpoint read the same way.

| Set | Base | Merged v1 |
|---|---|---|
| RACE validation (150 articles, 579 questions) | 95.16 | 95.85 |
| BoolQ validation (400) | 89.50 | 90.25 |
| RACE buried in ~7k tokens (40 articles, 157 questions) | 85.35 | 93.63 |
| RACE buried in ~10k tokens (the same 157) | 83.44 | 94.27 |
| Kev transfer-v4 test (764) | 81.28 | 85.34 |
| of which the three held-out families (228) | 69.74 | 77.19 |

The long-context rows moved most; the held-out families moved as well, which says the adapter did not only learn the
trained families' formats. Single runs; the sets are small enough that differences of a point or two on the short rows
are within noise.

## Three published checkpoints, and why the 36-layer one is the default

The adapter above is trained against the full 40-layer base and published on its own as
[`littlemex/prismyra-decision-lora-qwen3.6-35b-a3b`](https://huggingface.co/littlemex/prismyra-decision-lora-qwen3.6-35b-a3b).
Three checkpoints with an adapter already folded in are published separately, at 40, 36 and 32 of the base model's 40
layers: [`...-fp8-40l`](https://huggingface.co/littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-40l),
[`...-fp8-36l`](https://huggingface.co/littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l) and
[`...-fp8-32l`](https://huggingface.co/littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-32l). The 36- and 32-layer
checkpoints are not the 40-layer adapter folded into a truncated base; each has its own LoRA, trained on the same
7,018 rows, distilled from the 40-layer checkpoint's own output distribution. `truncate_layers.py` produces the
truncated base a checkpoint's LoRA is then trained against.

Measured on the same five sets, one L40S, group 32, each Choice question's options read in the order the source data
gives them ("perm 1"):

| set | 40l | 36l | 32l | bar | which competitor |
|---|---|---|---|---|---|
| RACE validation (579) | 95.85 | 96.20 | 95.51 | 93.78 | Decider-4B |
| BoolQ validation (400) | 90.25 | 90.25 | 89.25 | 89.50 | Decider-4B |
| RACE buried in ~7k tokens (157) | 93.63 | 94.27 | 94.90 | 91.08 | Decider-4B |
| RACE buried in ~10k tokens (157) | 94.27 | 92.36 | 95.54 | 91.08 | Decider-4B |
| Kev transfer-v4 test (764) | 85.47 | 85.73 | 82.85 | 82.85 | Lux-9B |

The bar for each set is the better-scoring of two other decision models of comparable active parameter count,
Decider-4B (4B dense) and Lux-9B (9B dense, against this checkpoint's 35B total / 3B active MoE); which one wins
changes by set, and Lux-9B is the stronger of the two on Kev specifically. Averaging each Choice question's four
cyclic option orderings ("perm 4", which adds branch passes but no second reading of the context) instead of the one
order the data gives moves most sets by up to a few points either way -- BoolQ has no option order to average over,
so it is unchanged by construction:

| set | 36l, perm 4 | 32l, perm 4 | bar |
|---|---|---|---|
| RACE | 96.03 | 96.03 | 93.78 |
| BoolQ | 90.25 (unchanged) | 89.25 (unchanged) | 89.50 |
| RACE buried ~7k | 94.90 | 95.54 | 91.08 |
| RACE buried ~10k | 94.27 | 92.99 | 91.08 |
| Kev | 84.95 | 84.29 | 82.85 |

**Read the margins in questions, not just percentage points, and read plainly rather than round past.** On BoolQ, 40l
and 36l are 3 of 400 questions above the bar (361 against 358); 32l is 1 of 400 below it (357 against 358). A
single-run margin this size, in either direction, is a tie with the bar, not a clear win or a clear loss -- treat all
three as "at the BoolQ bar." On Kev, 32l ties the bar exactly (633 of 764 both), and 36l scores 2 questions above the
40-layer checkpoint (655 against 653) -- not evidence that 36l is more accurate than 40l, but evidence that cutting
four layers and retraining did not cost accuracy on this set either.

**32l is not simply the two longer checkpoints minus a bar.** It beats both 40l and 36l on the two long-context
buried sets (150 correct against 148 and 145 on the 10k-token version), so "faster but weaker" does not hold set by
set -- it holds only in the sense below, on a benchmark none of the three trained on.

## JevBench: a held-out benchmark this adapter never trained on

[JevBench](https://github.com/fstandhartinger/jevbench)'s 231 publicly licensed (MIT) questions -- 72 original, 48
easy, 111 hard -- are closed-form (true/false, multiple choice, or a small ordinal scale), so every one is expressible
as a Prismyra `Boolean` or `Choice` with no exclusions. Multi-word labels are remapped to a single letter with the
original label and its rubric kept in the prompt text, the accommodation
[TypeLLM](https://github.com/RadixArk/TypeLLM) makes for the same one-token constraint in its own prompts -- the
prompts are not identical between the two systems, since TypeLLM's answers below are quoted from its own run rather
than elicited with this evaluation's prompt.

Measured on one L40S, one question at a time:

| checkpoint | correct / 231 | accuracy | Brier (lower is better) | ECE (lower is better) |
|---|---|---|---|---|
| base, untrained, 40 layers | 178 | 77.06% | 0.2904 | 0.0732 |
| 40l | 202 | 87.45% | 0.1859 | 0.0405 |
| 36l | 203 | 87.88% | 0.1992 | 0.0757 |
| 32l | 181 | 78.35% | 0.3134 | 0.0784 |

A sign test over the 231 per-question outcomes finds no significant difference between the 40- and 36-layer
checkpoints (p = 1.000). It also finds no significant difference between the 32-layer checkpoint and the **40-layer**
untrained base (p = 0.795) -- there is no untrained 32-layer checkpoint to compare against, so this does not by
itself say the adapter's effect at 32 layers has vanished, only that 32l's score here is not distinguishable from the
40-layer base's. Read a non-significant result as "no difference detected in this test," not as demonstrated
equivalence: on the five sets above, 32l is close to or ahead of 40l and 36l on four of five, so JevBench and the
five sets disagree about how well the 32-layer checkpoint generalises, and that disagreement is reported rather than
resolved. What JevBench does add is calibration: 36l's Brier and ECE here are both worse than 40l's, and 36l's ECE
(0.0757) is worse than the untrained base's (0.0732) -- on a library whose whole mechanism is a probability, that is
a real cost of the 36-layer checkpoint that the five sets above do not surface.

**Against TypeLLM's own published answers for the same 231 questions** (`no_thinking/answers.jsonl`, quoted rather
than re-run): TypeLLM without a thinking step answers 84.42%, and a sign test finds no significant difference from
either the 40- or the 36-layer checkpoint (p = 0.230 and p = 0.152; both checkpoints score a few questions ahead by
point estimate, 202/203 against 195). This is a citation, not a reproduction, and the two systems are not otherwise
comparable: TypeLLM serves a different model (`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`) on a different GPU (an RTX
PRO 6000 Blackwell, per TypeLLM's own README) in a different quantisation format (NVFP4 against this checkpoint's
FP8). A non-significant result under those conditions supports "reading out a probability was not detectably worse
than a constrained generation method on this benchmark" and nothing about which *library* is better -- model, GPU,
prompt and method all vary at once, so a difference could not be assigned to any one of them, and no equivalence is
claimed. TypeLLM's thinking-enabled run (98.70%) beats every checkpoint here by a wide and significant margin
(p = 2.2e-07 against 40l, p = 4.2e-07 against 36l); Prismyra generates no text and has no comparable mode, so that
comparison is not a fair one and is reported only for completeness.

## Compared with general-purpose LLMs on the same five sets

The same five sets and 2,057 questions above were also put to four general-purpose LLMs over Bedrock -- Claude Opus
5.5, Claude Fable 5.1, GPT-6 Astra and Claude Haiku 4.5 -- each given the same context, the same question, the same
options, and an instruction to answer with a single letter. GPT-6 Astra was used in place of GPT-6 Sol, for which no
published per-token Bedrock price was found. This is not the read-out's usual speed comparison -- an unbatched decode
loop on the same weights is the fair opponent for that, and it is in [docs/ACCURACY.md](../../docs/ACCURACY.md) --
it answers a different question: how accurate is a 36-layer, decision-tuned checkpoint against models with no
one-token restriction, at what it costs to ask them the same 2,057 questions.

| set | 36l | Opus 5.5 | Fable 5.1 | GPT-6 Astra | Haiku 4.5 |
|---|---|---|---|---|---|
| RACE (579) | 96.20% | 97.41% | 97.93% | 97.24% | 95.16% |
| BoolQ (400) | 90.25% | 92.25% | 92.75% | 92.50% | 89.75% |
| RACE buried ~7k tokens (157) | 94.27% | 98.73% | 97.45% | 96.82% | 89.17% |
| RACE buried ~10k tokens (157) | 92.36% | 97.45% | 98.09% | 96.82% | 88.54% |
| Kev transfer-v4 test (764) | 85.73% | 94.11% | 88.87% | 90.31% | 69.76% |
| **all 2,057** | **90.71%** | **95.28%** | **93.53%** | **93.68%** | **83.71%** |

**This checkpoint loses on accuracy overall, and most clearly on Kev** -- a set built, by its own description, to
need knowledge from several source datasets and multi-step reasoning rather than reading comprehension. A sign test
over matched questions finds all three of the costlier models significantly ahead of the 36-layer checkpoint there:
p = 6.2e-12 against Opus, p = 0.017 against Fable, p = 1.0e-04 against GPT-6 Astra. Only Haiku, the cheapest of the
four, scores below the checkpoint overall (83.71% against 90.71%). Most of that is one failure mode: told to answer
with a single letter, Haiku still wrote out reasoning on Kev's harder questions and ran past its output limit before
producing one, on 64 of 764 Kev questions (8.4%); the other three models did this on five questions or fewer across
every set combined. Those runs are scored as wrong under the stated rule, not excluded or re-scored.

What is traded for that accuracy is cost, and by how much depends heavily on context length:

| set | 36l ($, GPU-time estimate) | Opus 5.5 | Fable 5.1 | GPT-6 Astra | Haiku 4.5 |
|---|---|---|---|---|---|
| per 1,000 questions, RACE | $0.03 | $2.08 | $4.87 | $3.43 | $0.35 |
| per 1,000 questions, BoolQ | $0.05 | $1.82 | $4.08 | $2.30 | $0.22 |
| per 1,000 questions, RACE buried ~7k | $0.05 | $40.58 | $101.04 | $69.97 | $7.48 |
| per 1,000 questions, RACE buried ~10k | $0.07 | $53.68 | $133.79 | $92.83 | $9.90 |
| per 1,000 questions, Kev | $0.05 | $1.95 | $3.63 | $2.22 | $0.29 |

The checkpoint's figure is what its own measured per-question time costs at the on-demand rate of the single
L40S-backed instance it ran on, crediting it as billed and fully utilised for exactly that time and nothing else --
an estimate of GPU-time cost under ideal utilisation, not a like-for-like per-token unit price, and not a bound in
every direction: it does not by itself say how much of that time is already shared across a batch of questions read
from the same document, which would push the true marginal cost per question lower still. The general-purpose
models' prices are Bedrock's published per-token rates, and they scale with input tokens, so the two long-context
sets are where the gap is largest: from about 4x cheaper (Haiku on BoolQ, the smallest gap measured) up to about
2,000x cheaper (Fable on the 7k-token buried set, the largest). Running the five-set, four-model comparison in full
cost $95.79 in Bedrock charges.

The general-purpose models' latency in this evaluation includes a network round trip to a shared service; the
checkpoint's does not. Some part of the latency gap below is that network round trip rather than the read-out
mechanism against generation -- the mechanism-level comparison against an unbatched decode loop, with no network
involved on either side, is in [docs/ACCURACY.md](../../docs/ACCURACY.md):

| set | 36l | Opus 5.5 | Fable 5.1 | GPT-6 Astra | Haiku 4.5 |
|---|---|---|---|---|---|
| median time per question, RACE | 48.6 ms | 1,392 ms | 2,078 ms | 786 ms | 440 ms |
| median time per question, Kev | 95.6 ms | 2,240 ms | 2,207 ms | 843 ms | 439 ms |

## Speed against other decision models, same day and GPU

Three other decision models -- Decider-4B, Kev-9B, and Lux-9B in the mode that branches from one context read across
several questions -- were re-measured on the same L40S on the same day as the 36-layer checkpoint, at 1, 16 and 64
questions about a ~5,400-token document made fresh (a one-line, never-repeated prefix) on every call. Lux-9B also
has a slower, unbatched mode that reads about 45 seconds at 64 questions; it is not in this table. The three
competitors were measured through their own HTTP servers; the checkpoint was called directly from Python at its
default request width, `group=32` -- the competitors' own internal request widths are not controlled by this
parameter and were not independently varied here:

| model | 1 question | 16 questions | 64 questions |
|---|---|---|---|
| Decider-4B (HTTP) | 512.9 ms | 534.3 ms | 999.7 ms |
| 36l (direct call) | 243.6 ms | 360.6 ms | 646.8 ms |
| Kev-9B (HTTP) | 638.6 ms | 838.9 ms | 1,534.9 ms |
| Lux-9B, branching from one read (HTTP) | 664.8 ms | 1,010.4 ms | 2,288.3 ms |

Each figure is the median of 5 timed calls after 2 warm-up calls. **This ranks what was actually measured; it does
not establish a fair one-mechanism-to-another ranking**, because the checkpoint's number excludes a network round
trip that every other row includes -- serving the checkpoint behind `prismyra-serve` and repeating the comparison
over HTTP is the like-for-like version of this table, and has not been done. It is still the number that matters for
deciding what to run in this project's own serving path, which calls Prismyra directly.

An earlier attempt at this same comparison read the 36-layer checkpoint at 1,066.3 ms for 64 questions -- slower than
Decider-4B -- and traced to two compounding measurement errors rather than one. `docs/PERFORMANCE.md` has the general
finding: a document made fresh on every call costs no more than the same document repeated, once both are measured
at the same request width; `tests/test_gpu.py::test_a_first_seen_context_costs_no_more_than_a_repeated_one` pins it.
The corrected figures above are what that fix looks like applied to this comparison.

Measured with the same script and GPU on the *same* document repeated rather than made fresh on every call --
[docs/PERFORMANCE.md](../../docs/PERFORMANCE.md) is why that distinction costs no more than noise once warmed up --
all three published checkpoints and the untrained base at both depths, at 64 questions about a ~5,300-token document:

| model | 64 questions |
|---|---|
| base, untrained, 40 layers | 733.4 ms |
| 40l | 716.8 ms |
| 36l | 650.3 ms |
| 32l | 584.7 ms |
| base, untrained, 9B dense | 2,050.1 ms |

Folding the adapter in costs nothing measurable against the untrained base at the same depth (716.8 ms against
733.4 ms is within the run-to-run noise this project treats as no difference); the speed difference across the three
published checkpoints tracks their layer count, as it should. Raw data:
[`benchmarks/results/decision_checkpoints_vs_base__l40s__repeated_document.jsonl`](../../benchmarks/results/decision_checkpoints_vs_base__l40s__repeated_document.jsonl).

## Against autotrust/JEV-27B-VL and LiquidAI/d1-3B

Two more decision models, measured against `fp8-36l` the same way as the three above: same gold labels, same
2,057 questions across the five sets this project's own README describes, each model read through its own
native API rather than forced through the other's. `JEV-27B-VL`'s FP8 quantisation (`Atlas3D/JEV-27B-VL-FP8`, a
community release, not an official one) was used in place of the unquantised 27B original -- it does not fit a
48 GB-class card -- after confirming its own model card's claim that it answers text questions the same as the
unquantised text-only sibling it is derived from. Reproduce the speed half with
[`benchmarks/compare_jev_d1.py`](../../benchmarks/compare_jev_d1.py); the two competitors' own numbers below were
read through their own serving code, not through this script -- see that script's own docstring for why.

**Accuracy** (sign test per set, Holm-corrected across the five; "tie" means not significant after correction):

| set | `fp8-36l` | JEV-27B-VL | d1-3B | `fp8-36l` vs JEV | `fp8-36l` vs d1-3B |
|---|---:|---:|---:|---|---|
| RACE (579) | 96.37% | 96.37% | ~90.2% | tie (p = 1.000) | **36l wins** (p < 0.0001) |
| BoolQ (400) | 90.50% | 90.25% | ~85.8% | tie (p = 1.000) | **36l wins** (p = 0.0091) |
| RACE buried ~7k (157) | 94.90% | 93.63% | ~79.6% | tie (p = 1.000) | **36l wins** (p = 0.0001) |
| RACE buried ~10k (157) | 92.99% | 91.08% | ~77.7% | tie (p = 1.000) | **36l wins** (p = 0.0003) |
| Kev transfer-v4 (764) | 87.04% | 79.97% | ~77.9% | **36l wins** (p < 0.0001) | **36l wins** (p < 0.0001) |
| **pooled, 2,057** | **91.40%** | 88.48% | 83.08-83.13% | **36l wins by 2.92pt** | **36l wins by 8.3-8.4pt** |

`fp8-36l` ties JEV-27B-VL on four of five sets and beats it on the fifth (Kev, this project's own hardest
transfer set); it beats d1-3B significantly on every one of the five. **Neither competitor beats this checkpoint
on accuracy on any set measured.**

**Probability quality** (2,057 questions, lower is better for both):

| | `fp8-36l` | JEV-27B-VL | d1-3B |
|---|---:|---:|---:|
| Brier | **0.135** | 0.174 | 0.234-0.235 |
| ECE (15 bins) | 0.043 | **0.037** | **0.025-0.027** |

`fp8-36l` wins on Brier against both; both competitors are better calibrated by ECE, d1-3B markedly so --
reported rather than smoothed over, since this project's own answer is confident more often than it is right by
this one measure.

**Speed** (one document, held just over 3,401 Prismyra tokens; d1-3B's own tokenizer counts it slightly
differently; 2 warm-up calls then median of 7). The one-question row is a same-day, same-GPU (L40S)
measurement for both `fp8-36l` and JEV-27B-VL, interleaved in one run; the 16- and 64-question rows for
`fp8-36l` are from the same session's own version-to-version comparison on the identical GPU. JEV-27B-VL's
16- and 64-question figures and all of d1-3B's figures are not from that session -- they are carried over
from an earlier comparison round on the same hardware class, reused because neither competitor's served
checkpoint has changed since that round:

| questions | `fp8-36l` | JEV-27B-VL | d1-3B (RTX PRO 4500, compiled) | d1-3B (L40S, compiled) |
|---|---:|---:|---:|---:|
| 1 | 170.5 ms | **154.8 ms** | 159.3 ms | **98.1 ms** |
| 16 | **229.9 ms** | 1,423.7 ms | 203.1 ms | 130.1 ms |
| 64 | **335.3 ms** | 5,719.5 ms | 298.6 ms | 190.5 ms |

Three different shapes of result, not one verdict:

- **JEV-27B-VL is 9.2% faster than `fp8-36l` at one question**, in this same-day, same-GPU measurement -- a
  reversal from an earlier comparison round, which had reported `fp8-36l` ahead at one question by the
  then-current figures on both sides. The cause of that reversal is not established; the tuned MoE tile
  pinned in v0.4.3 was checked directly on this hardware and confirmed to still be loading and taking effect,
  which rules it out as the cause. At 16 and 64 questions `fp8-36l` is unchanged in kind and still wins
  decisively (6.2x and 17.1x), because JEV's own API answers one decision per call and pays for the document
  again on every one -- a design limit, not a tuning gap, and not something a later JEV release closes
  without changing that API.
- **d1-3B is faster than `fp8-36l` at every width measured, on both cards it was measured on.** A 3B dense
  model with no routed experts has less to compute than a 35B-A3B MoE at any width; this is the honest trade
  this project's own accuracy numbers above are made against, not a result this release changes.
- **JEV-27B-VL is not bit-deterministic; this project's own checkpoints are built to be.** The same input, read
  repeatedly in one process, across companions in one batch, and across process restarts, returned at least
  three distinct probability values for JEV-27B-VL in an eleven-trial sample. This release adds a construction-
  time self-check (`_check_fp8_determinism`, [docs/KERNELS.md](../../docs/KERNELS.md)) specifically for the
  one hardware-level gap in `fp8-36l`'s own determinism story that unseeded kernel probes could not see --
  confirmed on real hardware on both of this project's served cards (L40S and RTX PRO 4500) as part of this
  release.

**Memory**: `fp8-36l` holds about 32.7 GiB on an L40S; JEV-27B-VL's FP8 quantisation holds about 34.2 GiB
(similar, despite the regular-dense-27B-against-MoE-35B difference, because JEV also carries a vision encoder
this comparison's text-only questions never use); d1-3B holds about 6 GiB on either card, light enough to run
several copies beside everything else on one GPU. Pick on this axis is a question about how many other things
need to share the card, not about the model's own ability to answer questions.

## The commercially-clear line (v0.5.0) against the same five sets

Two more checkpoints, from the training line described in
[`docs/DATA_PROVENANCE.md`](../../docs/DATA_PROVENANCE.md#v050-a-commercially-clear-training-line-prismyra-decision-35b-a3b-fp8-revision-3c88c546-prismyra-decision-35b-a3b-nvfp4-revision-9d08c9b4):
`prismyra-decision-35b-a3b-fp8` and `prismyra-decision-35b-a3b-nvfp4`, called `fp8` and `nvfp4` below to tell them
apart from the `-36l` checkpoint above. Measured on the identical 2,057 questions across the same five sets as the
section above. Scoring throughout this section: sign test per set, matched question by question, Holm-corrected
across the five sets in one family; "tie" means the Holm-corrected p-value is at or above 0.05.

This section has three subsections, each its own self-contained measurement round -- read the heading before
reading a number, since `fp8` and `nvfp4`'s own figures differ slightly between them (run-to-run noise on this
benchmark, consistent with the "tie" result the quantisation-difference table below reports in both rounds):
against `fp8-36l` (the previous line, not re-measured since), against `JEV-27B-VL` (an earlier round, kept for
the record), and against `JEV-35B`/`JEV-35B-NVFP4` (the current round). Two more models this project tried to
include in the current round, `autotrust/GEV-26B-Decide` and `autotrust/AutoTrust-JEV-27B-v1.1`, could not be
served in this project's environment and are not measured anywhere in this comparison -- this reports what ran,
not every model that exists.

### Against `fp8-36l`, the previous line

| set | `fp8` − `fp8-36l` | Holm p | `nvfp4` − `fp8-36l` | Holm p |
|---|---:|---:|---:|---:|
| RACE (579) | -1.38 | 0.0975 (tie) | -1.21 | 0.2121 (tie) |
| BoolQ (400) | -0.50 | 0.6374 (tie) | +0.25 | 0.8772 (tie) |
| RACE buried ~7k (157) | -5.73 | **0.0266 (loses)** | -5.10 | **0.0456 (loses)** |
| RACE buried ~10k (157) | -3.82 | 0.2176 (tie) | -1.91 | 0.8772 (tie) |
| Kev transfer-v4 (764) | -3.27 | **0.0068 (loses)** | -4.32 | **0.0001 (loses)** |
| **pooled** | **-2.43pt** | | **-2.43pt** | |

(`fp8` pools to 89.16%, `nvfp4` to 89.16%, `fp8-36l` to 91.59%, on this round's 2,057 questions.) Both
checkpoints in this line lose to `fp8-36l` on the long-buried-context set and on Kev, and tie it on the other
three; read plainly, that is two losses and three ties each, not an across-the-board loss. This line's adapter is
trained only on sources confirmed clear for commercial use and for training a derivative model (see
`docs/DATA_PROVENANCE.md`); the pooled 2.43-point gap above is that restriction's measured cost on this
benchmark.

### Against JEV-27B-VL (an earlier round, kept for the record)

`autotrust/JEV-35B` is a newer model from the same vendor; see the next subsection for a comparison against it,
measured in a later round against `fp8` and `nvfp4` remeasured the same day as JEV-35B itself. The numbers below
are from the earlier round and are not reused there.

| set | `fp8` − JEV-27B-VL | Holm p | `nvfp4` − JEV-27B-VL | Holm p |
|---|---:|---:|---:|---:|
| RACE (579) | -1.21 | 0.3249 (tie) | -1.04 | 0.4719 (tie) |
| BoolQ (400) | -0.25 | 1.0000 (tie) | +0.50 | 1.0000 (tie) |
| RACE buried ~7k (157) | -4.46 | 0.2088 (tie) | -3.82 | 0.3331 (tie) |
| RACE buried ~10k (157) | -1.27 | 1.0000 (tie) | +0.64 | 1.0000 (tie) |
| Kev transfer-v4 (764) | +4.06 | **0.0428 (wins)** | +3.01 | 0.2806 (tie) |
| **pooled** | **+0.68pt** | | **+0.68pt** | |

(`fp8` and `nvfp4` both pooled to 89.16% in this earlier round; JEV-27B-VL pooled to 88.48%; d1-3B, not
re-measured either round, pools to 83.08-83.13% on the earlier round's own report.) Read this plainly: `fp8`'s
point estimate is behind JEV-27B-VL on four of the five sets (RACE, BoolQ, and both buried sets), and the Holm
correction only clears 0.05 on the fifth, Kev -- the largest set (764 of 2,057 questions), which is why the
pooled number is positive. `nvfp4` is weaker still: none of its five per-set comparisons against JEV-27B-VL
clears the Holm bar, including Kev (p = 0.2806).

**Probability quality, speed and determinism against JEV-27B-VL** (same earlier round; lower is better for
Brier and ECE):

| | `fp8` | `nvfp4` | `fp8-36l` | JEV-27B-VL | d1-3B |
|---|---:|---:|---:|---:|---:|
| Brier | 0.1631 | 0.1620 | **0.1347** | 0.1742 | 0.234-0.235 [reused] |
| ECE (15 bins) | 0.0282 | **0.0217** | 0.0422 | 0.0373 | 0.025-0.027 [reused] |

| questions | `fp8` (L40S) | `nvfp4` (RTX PRO 4500) | `fp8-36l` (L40S) | JEV-27B-VL (L40S) | d1-3B (RTX PRO 4500) | d1-3B (L40S) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 152.8 ms | 154.8 ms | 153.6 ms | 156.2 ms | 159.3 ms [reused] | 98.1 ms [reused] |
| 16 | 217.8 ms | 188.6 ms | 219.6 ms | 1,573.3 ms | 203.1 ms [reused] | 130.1 ms [reused] |
| 64 | 305.3 ms | 272.3 ms | 306.2 ms | 6,346.0 ms | 298.6 ms [reused] | 190.5 ms [reused] |

At one question, `fp8`, `fp8-36l` and JEV-27B-VL are within 2% of each other on this round -- under this
project's own 5% bar for calling something a real difference. At 16 and 64 questions, JEV-27B-VL is 7-21x
slower than either Prismyra checkpoint: its own `/v1/decide` answers one decision per call and pays for the
document again every time, a design limit this project's own checkpoints do not share. JEV-27B-VL is also not
bit-deterministic, measured once in an earlier round and not independently re-tested since.

### Against JEV-35B and JEV-35B-NVFP4 (current round)

`autotrust/JEV-35B` (a fine-tune of `Qwen/Qwen3.5-35B-A3B`) and its NVFP4 sibling, measured the way each
checkpoint's own card documents reading a decision (its `/v1/decide` request shape, applied directly rather
than over HTTP), against `fp8` and `nvfp4` remeasured the same day. **A hardware note before the numbers**:
JEV-35B's own card states it needs one GPU with 80 GB or more; it does not fit one 46 GB L40S, so it was served
across two L40S with vLLM's tensor-parallel splitting (`--tensor-parallel-size 2`) -- the same bf16 weights,
sharded across two GPUs, not a different quantisation. `fp8` ran on one L40S; `nvfp4` and JEV-35B-NVFP4 each ran
on one RTX PRO 4500. The speed table below is what each checkpoint needed to serve its own published format, not
a GPU-count-matched comparison.

| set | `fp8` | `nvfp4` | JEV-35B (bf16) | JEV-35B-NVFP4 |
|---|---:|---:|---:|---:|
| RACE (579) | 95.68% | 95.16% | 96.55% | 93.96% |
| BoolQ (400) | 90.50% | 89.75% | 90.75% | 87.25% |
| RACE buried ~7k (157) | 89.17% | 89.81% | 94.27% | 81.53% |
| RACE buried ~10k (157) | 89.81% | 91.08% | 93.63% | 82.17% |
| Kev transfer-v4 (764) | 83.25% | 83.12% | 78.80% | 69.76% |
| **pooled, 2,057** | **89.11%** | **88.92%** | **88.43%** | **81.82%** |

**`fp8` against JEV-35B:**

| set | diff(pt) | Holm p |
|---|---:|---:|
| RACE (579) | -0.86 | 0.2633 (tie) |
| BoolQ (400) | -0.25 | 0.8527 (tie) |
| RACE buried ~7k (157) | -5.10 | **0.0456 (loses)** |
| RACE buried ~10k (157) | -3.82 | 0.2498 (tie) |
| Kev transfer-v4 (764) | +4.45 | **0.0307 (wins)** |
| **pooled** | **+0.68pt** | |

The same structure as the earlier round's result against JEV-27B-VL above: `fp8` loses the long-buried-context
set, wins the largest set (Kev), and ties the other three. The competitor's model changed; the shape of this
line's result against it did not.

**`nvfp4` against JEV-35B-NVFP4:**

| set | diff(pt) | Holm p |
|---|---:|---:|
| RACE (579) | +1.21 | 0.0825 (tie) |
| BoolQ (400) | +2.50 | 0.0825 (tie) |
| RACE buried ~7k (157) | +8.28 | **0.0086 (wins)** |
| RACE buried ~10k (157) | +8.92 | **0.0019 (wins)** |
| Kev transfer-v4 (764) | +13.35 | **0.0000 (wins)** |
| **pooled** | **+7.10pt** | |

`nvfp4` wins three of five sets outright here, a clearly different result from the near-tie against JEV-35B's
own bf16 weights above. The reason is JEV-35B-NVFP4 itself, not `nvfp4` improving: measured head to head, bf16
JEV-35B beats its own NVFP4 sibling on **every one of the five sets**, all Holm-significant (+2.59 to +12.74pt,
pooled +6.61pt) -- a quantisation cost on this benchmark with no counterpart in this project's own fp8-to-nvfp4
comparison, which ties on every set both this round and the earlier one (+0.19pt pooled here, +0.00pt pooled in
the earlier round; see the two-row table below).

| set | `fp8` − `nvfp4` (this round) | Holm p |
|---|---:|---:|
| RACE (579) | +0.52 | 1.0000 (tie) |
| BoolQ (400) | +0.75 | 0.4163 (tie) |
| RACE buried ~7k (157) | -0.64 | 1.0000 (tie) |
| RACE buried ~10k (157) | -1.27 | 1.0000 (tie) |
| Kev transfer-v4 (764) | +0.13 | 1.0000 (tie) |
| **pooled** | **+0.19pt** | |

JEV-35B-NVFP4's weakness is not only accuracy. The same input, asked 11 times in one process, returned **8
different probability pairs** (range 0.453-0.640 on the chosen option's own probability) -- `fp8`, `nvfp4` and
JEV-35B (bf16) each returned the identical value all 11 times. Two restarts of the process gave matching answers
for all four checkpoints, including JEV-35B-NVFP4, but with the same-process result above, one matching pair of
restarts does not establish that this checkpoint is actually reproducible rather than coincidentally close on
that one probe.

**Probability quality** (2,057 questions, lower is better for both):

| | `fp8` | `nvfp4` | JEV-35B | JEV-35B-NVFP4 |
|---|---:|---:|---:|---:|
| Brier | 0.1633 | 0.1610 | **0.1598** | 0.2528 |
| ECE (15 bins) | 0.0255 | 0.0208 | **0.0194** | 0.0546 |

JEV-35B (bf16) has the best Brier and ECE of the four; JEV-35B-NVFP4 has the worst of both by a wide margin,
consistent with its accuracy and determinism results above.

**Speed** (one document, 3,446 tokens by each checkpoint's own tokenizer, 2 warm-up calls then median of 7):

| questions | `fp8` (1x L40S) | `nvfp4` (1x RTX PRO 4500) | JEV-35B (2x L40S, TP=2) | JEV-35B-NVFP4 (1x RTX PRO 4500) |
|---|---:|---:|---:|---:|
| 1 | 150.8 ms | 177.2 ms | **99.2 ms** | **82.3 ms** |
| 16 | **252.3 ms** | **211.6 ms** | 688.6 ms | 412.5 ms |
| 64 | **369.4 ms** | **302.1 ms** | 2,759.2 ms | 1,675.1 ms |

**At one question, the JEV-35B family is faster.** At 16 and 64 questions, both Prismyra checkpoints are
faster, by a widening margin (JEV-35B takes 2.76 seconds at 64 questions against `fp8`'s 0.37). This round
answers several questions about the same document in one batched call rather than one call per question -- a
different implementation than JEV-27B-VL's one-call-per-decision API above -- yet JEV-35B's own time still grows
close to linearly with the number of questions, where this project's own checkpoints do not; this round did not
trace that difference to a specific cause.

**Long context.** This project's engine allocates a fixed token budget at start-up; `fp8` and `nvfp4` both
answer one question about a 65,536-token document correctly, and both refuse a 128,726-token document
(`a context of 128726 tokens is longer than this engine allocates for (65536)`) -- a configured limit, not a
model-capacity ceiling, and neither checkpoint was measured past it this round. JEV-35B and JEV-35B-NVFP4's own
cards state native support to 262,144 tokens; at 65,536 tokens, on 175 sampled questions across three Decision
RULER tasks (simple and harder needle retrieval, and multi-variable tracking):

| task | `fp8` | JEV-35B | JEV-35B-NVFP4 |
|---|---:|---:|---:|
| needle retrieval, simple (25) | 100.0% | 100.0% | 100.0% |
| needle retrieval, harder (25) | 100.0% | 100.0% | 100.0% |
| variable tracking (125) | 63.2% | **94.4%** | 48.0% |

(`nvfp4`'s own 65,536-token result is not comparable here: on the 32 GB card it ran on, the first question in a
session succeeds but every later one in the same process fails on memory, so only one question was scored --
not a quantisation accuracy result, a single card's memory ceiling at this context length.) The two simple
retrieval tasks are saturated for every model measured; variable tracking is where they separate, and JEV-35B
(bf16) is clearly ahead of `fp8` there (94.4% against 63.2%), which is itself clearly ahead of JEV-35B-NVFP4
(48.0%) -- the same bf16-beats-its-own-nvfp4-sibling pattern as the production benchmark above. On a further
sample of 16 question-parts from Decision LCR (real documents of roughly 88,000-122,000 tokens, built from
`ArtificialAnalysis/AA-LCR`'s own extracted text, scored per part rather than by this benchmark's own official
all-or-nothing rule, so not comparable to a published Decision LCR score), JEV-35B scores 62.5% and
JEV-35B-NVFP4 56.2%; `fp8` and `nvfp4` were not measured here, for the same budget reason as above.

**Reproducing this comparison**: the accuracy and long-context halves score each item the same way as the
earlier-round subsection above (argmax of the returned probabilities against the gold label; sign test per set
with a Holm correction across the family being compared; Brier as mean squared distance from the one-hot gold
label; ECE from 15 confidence bins). The speed half follows the same method as
[`benchmarks/compare_jev_d1.py`](../../benchmarks/compare_jev_d1.py) (2 warm-up calls, median of 7, a filler
document built to a target token count); JEV-35B's own `/v1/decide` request shape and tensor-parallel serving
requirement mean this round's harness calls its weights directly through its own documented API rather than
through that script unmodified. RACE and BoolQ are public (`race`, `boolq` on Hugging Face); the two
buried-context sets, Kev transfer-v4, and the Decision RULER and Decision LCR samples drawn from them are this
project's own held-out construction or a third-party long-context suite assembled locally, and are not bundled
in this repository.

## Weak-category mixture: sentiment and causal/multi-step reasoning

A held-out panel of public benchmarks this checkpoint had never been evaluated on before turned up two
judgment types it loses on against a comparable general-purpose model: sentiment classification and
causal/multi-step reasoning. Looking at the trained families, neither type has a match -- the checkpoint
had never seen a training example of either, while the judgment types it ties or wins on each have at
least one matching family in the trained pool.

`build_weak_category_mixture.py` adds four licence-clear public datasets covering the two missing types
(financial and general-tweet sentiment; causal and social-situation multi-step reasoning) and mixes them
into an existing pool at a small, fixed fraction. The fraction matters more than it looks: mixing the same
four families in at roughly 7% with no other change measurably improved sentiment but also regressed a
production-safety set this checkpoint is not allowed to regress on, concentrated on the items closest to
the new rows in topic or already closest to a 50/50 call before training -- not a data leak, but ordinary
interference from the one thing the new rows lacked that every row in the existing pool has: a
self-distillation target holding the step toward the model's own prior output as well as toward the label.
Halving the fraction, with nothing else changed, removed the regression and kept the sentiment gain.

Measured on 400 held-out sentiment questions and 400 held-out causal/multi-step-reasoning questions (two
public benchmarks the mixture never trained on), against the base checkpoint, paired sign test:

| mixed-in fraction | sentiment (400) | reasoning (400) | production-safety set, no item clearly regressed? |
|---|---|---|---|
| ~7.4% | **significant gain** (p < 0.01 before correction) | gain, not significant | **no** -- one item's paired 95% CI fell entirely below zero |
| ~4.0% | **significant gain** (p < 0.01, Holm-corrected across both types) | gain, not significant | **yes** -- no item's CI fell entirely below zero |

The reasoning gain was in the right direction at both fractions but did not clear significance either
time; the sentiment gain held at both fractions, and only the smaller one kept the production-safety set
clean. A paired bootstrap over 14 production and held-out items, each compared against the base
checkpoint, found zero items with a confidence interval entirely below zero at the smaller fraction,
against one at the larger fraction -- the same comparison, same items, same statistics, differing only in
how much of the new data was mixed in.
