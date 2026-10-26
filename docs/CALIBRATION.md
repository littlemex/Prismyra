# A temperature in the engine's own read-out, not after it

`docs/ACCURACY.md` already covers `prismyra.Temperature`: one scalar, fitted by a caller on labelled
answers and applied after `ask()` has already returned, that halves the gap between a claimed
probability and the rate it is actually right. This page is about a second, separate thing: the same
idea wired into the engine's own read-out (`Prismyra(..., calibrate_temperature=...)`,
`readout.score`'s `temperatures` argument), so an engine can ship pre-calibrated instead of asking
every caller to remember to rescale. It is **off by default**, and this page is the record of why it
corrects by option count rather than with one number for everything, on what data that one number was
chosen, and what did and did not survive checking it honestly.

## The question: does one scalar for everything still work, wired this way?

`docs/ACCURACY.md`'s own measurement fits and scores on the same benchmark (RACE-middle, BoolQ), cross
-fitted two ways on a few hundred answers each. The engine's read-out is used across five production
sets at once (RACE, BoolQ, and three harder sets: two "buried" long-context sets and a mixed
contrastive/compositional set called Kev), 2,057 answers total, round11c (the v0.5.1 fp8 checkpoint).
Four grouping schemes were compared, each with a 5-fold cross-validation (fit on four folds, score the
held-out fifth, pool the five held-out folds) rather than the fit-and-score-on-everything numbers a
single train/test split would give:

| scheme | what shares one T | pooled ECE (raw -> cross-validated) | pooled Brier (raw -> cross-validated) |
|---|---|---|---|
| global | every answer | 0.0179 -> 0.0184 (worse) | 0.1633 -> 0.1635 (worse) |
| per kind | `noul` / `choice` / `score` | 0.0179 -> 0.0184 (worse) | 0.1633 -> 0.1640 (worse) |
| per tag | each set's own provenance (race150, boolq400, bury7k, bury10k, and Kev's eleven sub-tags) | 0.0179 -> 0.0194 (worse) | 0.1633 -> 0.1626 |
| per option count | 2/3/4/5/6/7-choice questions each get their own T | 0.0179 -> 0.0203 (worse) | 0.1633 -> 0.1635 |

**All four lose, pooled, to doing nothing.** A scalar that helps one sub-population and hurts others
nets negative once they are pooled back together, and the "per option count" row is the clearest
example: it looks identical to the "selective" row below in spirit, but fitting (and shipping) a T for
*every* option count -- including 3-choice (n=40, mostly Kev's small contrastive sets) and 7-choice
(n=12) -- lets a scalar fit to noise on a 12-row group move the pooled number the wrong way. This is
the overfitting the 5-fold check exists to catch, and it would not have shown up fitting and scoring
on the same split.

## What does survive: one option count, the one with enough rows to trust

Restricting the correction to the single option count this benchmark has enough of -- 4-choice
questions, 1,101 of the 2,057 answers (54%), mostly RACE and Kev's MMLU slice -- and leaving every
other question exactly as it is (T=1, untouched) does survive the same 5-fold check:

| | pooled ECE | pooled Brier | argmax changed |
|---|---|---|---|
| raw (n=2,057) | 0.0179 | 0.1633 | -- |
| T=0.9111 on 4-choice only, else untouched | **0.0162** | 0.1634 | 0/2,057 |

Per-fold, the improvement is real but not uniform: four of five folds improve (ECE 0.0251->0.0203,
0.0480->0.0462, 0.0179->0.0179, 0.0270->0.0236) and one gets worse (0.0200->0.0325). This is reported
rather than smoothed over -- it is why this ships off by default rather than as a universal
improvement, and why the spec format below names exactly one option count rather than implying it
covers every shape a question could take.

T=0.9111 (fit with `Temperature.fit`'s grid search, minimising negative log likelihood, on all 1,101
4-choice rows of the 2,057-question set) is what `prismyra/calibration_temperature.json` ships. Below
one means this read-out was already slightly *under*-confident on this slice -- the same direction
`docs/ACCURACY.md`'s own RACE number (0.90) found, for the reason that page gives: softmaxing over only
the options a question declared starts closer to honest than reading letters out of the whole
vocabulary would.

### Vector scaling was measured too, and disqualified on a different ground

A per-position additive bias (Platt/vector-scaling style, one bias per option-count group, fitted by
gradient descent on the same 5-fold splits) scored *better* than the 4-choice-only scalar on this same
pooled set (Brier 0.1577 against 0.1634) -- and is not shippable regardless, because it moved the
argmax on 52 of 2,057 rows (2.53%). A temperature is a positive scalar dividing every option's score in
a row; it cannot reorder them. A per-position bias can and did. The engine's contract is that
`calibrate_temperature` never changes which option wins (checked by `test_readout.py`'s
`test_a_temperature_cannot_change_which_option_wins` and the real-hardware check below), so this
candidate is reported and then set aside rather than chased for its better number.

## Checking it on data it was not fit on

The 2,057-question set is what chose T=0.9111; a fair check needs a slice that number never saw.
`lock16_rows.json`'s `"choice"` rows with exactly four options (785 of them, across
`lock16_prod_mmlu`, `lock16_contrast_cosmos`, and `lock16_contrast_sciq`) are a different project's
locked regression slice and were not part of the fit. Scored through the real engine
(`Prismyra("bigrun2_relv05_fp8", ...)`, the same checkpoint, on an EKS GPU node):

| | pooled ECE | pooled Brier |
|---|---|---|
| off (raw) | 0.0180 | 0.2133 |
| on (T=0.9111) | 0.0182 | 0.2126 |

Pooled, this is a wash within noise -- ECE a hair worse, Brier a hair better -- and the per-family
picture is mixed the same way the per-fold picture above was: `lock16_contrast_cosmos` improves
(0.0598 -> 0.0401) and `lock16_prod_mmlu` improves slightly (0.0228 -> 0.0214), while
`lock16_contrast_sciq` moves the other way (0.0279 -> 0.0286). **This correction does not clearly
generalise past the benchmark it was fit on.** That is the second, independent reason it ships off by
default and names its provenance in the spec file rather than presenting itself as a general-purpose
fix: a caller who has their own labelled slice should check it there before turning it on, the same way
this page just did.

## What the engine guarantees, checked on real hardware

Three properties, checked against `bigrun2_relv05_fp8` (round11c) on an EKS GPU node, not only in the
CPU unit tests:

1. **Off by default is bit-identical to before this feature existed.** The same 785 lock16 4-choice
   rows, scored once by this engine's own `origin/main` (no `calibrate_temperature` parameter at all)
   and once by this branch with `calibrate_temperature=None`, produced byte-for-byte identical
   probabilities and gold labels for every row.
2. **The argmax never moves.** Scoring the same 785 rows with `calibrate_temperature` on versus off
   changed which option won on 0 of 785 rows.
3. **It is deterministic.** Eleven repeats of the same five questions, same engine, `calibrate_
   temperature` on, returned the identical probability for every repeat.

## Using it

```python
from prismyra import Prismyra

# off (default): bit-identical to an engine built before this feature existed
engine = Prismyra("your/model")

# on: reads prismyra/calibration_temperature.json, the one candidate this page measured and shipped
engine = Prismyra("your/model", calibrate_temperature="prismyra/calibration_temperature.json")

# or build a spec of your own, the same way prismyra.heads reads its spec:
engine = Prismyra("your/model", calibrate_temperature=[
    {"nchoices": 4, "temperature": 0.9111, "fit_on": "..."},
])
```

A spec entry needs only `nchoices` and `temperature`; anything else (`fit_on`, `fit_date`,
`validated_by`, ...) is kept for a human tracing where a number came from and is never read by the
engine. A question whose option count has no entry is scored exactly as without a spec at all.

See `prismyra/temperature.py` for `GroupedTemperature` (the engine-wired class this page measures) and
`Temperature`/`cross_fit` (the caller-side, single-scalar class `docs/ACCURACY.md` measures).
