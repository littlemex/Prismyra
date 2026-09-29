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
