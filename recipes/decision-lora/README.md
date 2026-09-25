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
