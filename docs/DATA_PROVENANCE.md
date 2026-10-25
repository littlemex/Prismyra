# Data provenance

What the LoRA adapter folded into `fp8-36l` and `nvfp4-36l` was trained on, and under what licence, so a caller can
check commercial-use terms without re-deriving the training history from the commit log. This page is additive: new
rounds get a new dated section below the current one, rather than rewriting history.

Every licence below was read from the dataset's own licence file or dataset-card metadata, not from a summary --
where a dataset mixes licences by sub-collection (for example GLUE, which re-distributes several independently
licensed sources), the sub-collection actually used is named, not the umbrella package.

## Round shipped 2026-10 (`fp8-36l` revision `57a7076e`, `nvfp4-36l` revision `53291b60`)

### Core question-answering pool (13 families)

These are the families the adapter's training pool is built from across every revision to date (`recipes/
decision-lora/build_families.py` and the knowledge-distillation pool it feeds). Gold labels are each dataset's own;
see "Teacher models" below for the one exception.

| Family | Source | Licence (SPDX) | Licence text | Commercial use |
|---|---|---|---|---|
| CommonsenseQA | [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa) | MIT | dataset card metadata | yes |
| AQuA-RAT | [`deepmind/aqua_rat`](https://huggingface.co/datasets/deepmind/aqua_rat) | Apache-2.0 | dataset card metadata | yes |
| MMLU (auxiliary-train) | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) | MIT | dataset card metadata | yes |
| Cosmos QA | [`allenai/cosmos_qa`](https://huggingface.co/datasets/allenai/cosmos_qa) | CC-BY-4.0 | dataset card metadata | yes |
| QQP (Quora Question Pairs, via GLUE) | [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) (`qqp` config) | not formally published by Quora | GLUE's own card defers to "the original licenses accompanying each dataset"; Quora's own release page does not state an SPDX licence | **not independently confirmed** -- long-standing community practice treats it as research/ML-use permissive (it is in essentially every open LLM pretraining and GLUE-derived corpus), but no primary licence text exists to cite |
| MNLI (via GLUE) | [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) (`mnli` config) | not formally published as one licence | the MultiNLI project page's own "License" section points only to the data-description paper; the corpus mixes genres (fiction, government reports, transcribed speech, web text) that may carry different source licences | **not independently confirmed**; treat the same way as QQP above pending a per-genre check |
| OpenBookQA | [`allenai/openbookqa`](https://huggingface.co/datasets/allenai/openbookqa) | not formally published for the data | dataset card tag is `unknown`; AI2's code repository for this dataset is Apache-2.0, but that covers the baseline code, not the data file | **not independently confirmed** |
| ARC (ARC-Challenge/ARC-Easy) | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | CC-BY-SA-4.0 | AI2's own dataset page (`allenai.org/data/arc`) and dataset card metadata agree | yes (share-alike: a redistributed *copy* of the data must carry the same licence; this does not require releasing the model weights under CC-BY-SA) |
| QuALITY | [`nyu-mll/quality`](https://github.com/nyu-mll/quality) | CC-BY-4.0 | project leaderboard page (`nyu-mll.github.io/quality`) states "QuALITY is distributed under a CC BY 4.0 License" | yes |
| BIG-bench logical-deduction | [BIG-bench](https://github.com/google/BIG-bench) (`logical_deduction` task) | Apache-2.0 | BIG-bench repository licence | yes |
| HotpotQA (comparison slice) | [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | CC-BY-SA-4.0 | dataset card metadata | yes (share-alike, same note as ARC) |
| SciQ | [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) | **CC-BY-NC-3.0** | dataset card metadata | **no** -- non-commercial only |
| DREAM | [`dataset-org/dream`](https://huggingface.co/datasets/dataset-org/dream) (source: [nlpdata/dream](https://github.com/nlpdata/dream)) | **custom, non-commercial** | the source repository's own `license.txt`: "DREAM dataset is intended for non-commercial research purpose only." | **no** -- non-commercial only |

### Also present in the shipped training pool (older rows, predate the "13 families" framing above)

| Family | Source | Licence (SPDX) | Licence text | Commercial use |
|---|---|---|---|---|
| RACE | [`ehovy/race`](https://huggingface.co/datasets/ehovy/race) | **custom, non-commercial** | CMU's own RACE data agreement restricts use to non-commercial research; already disclosed in both checkpoints' Hugging Face cards | **no** -- non-commercial only |
| BoolQ | [`google/boolq`](https://huggingface.co/datasets/google/boolq) | CC-BY-SA-3.0 | dataset card metadata | yes (share-alike, same note as ARC) |
| AG News (as `typed_topic_agnews`) | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) (source: ComeToMyHead's AG corpus) | **custom, non-commercial** | the original corpus page (`groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html`): "provided ... for research purposes ... any other non-commercial activity. You are encouraged to download this corpus for any non-commercial use." | **no** -- non-commercial only |
| CLINC150, Banking77 (as `typed_intent_*`) | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos), [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-3.0, CC-BY-4.0 | dataset card metadata | yes |
| TREC (as `typed_qtype_trec`), Yahoo Answers Topics (as `typed_topic_yahoo`), QNLI (via GLUE), `tweet_eval`'s `offensive` config | various | not individually re-verified in this pass | -- | **unaudited** -- flagged for follow-up before a stricter compliance review; not re-checked against a primary source here |
| `synth_logic`, `synth_signoff`, `synth_timeliness` | this project's own synthetic rows | n/a (not third-party data) | self-authored | yes |

### Weak-category mixture, added this round (`recipes/decision-lora/build_weak_category_mixture.py`, 4 families)

| Family | Source | Licence (SPDX) | Licence text | Commercial use |
|---|---|---|---|---|
| Financial-news sentiment | [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | MIT | dataset card metadata | yes |
| General-tweet sentiment | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`sentiment` config only) | CC-BY-3.0 | the dataset card's own "Licensing Information" section (the package's umbrella tag is `unknown` because its eleven configs are not uniformly licensed; only the `sentiment` config is used here) | yes |
| Causal reasoning | [`pkavumba/balanced-copa`](https://huggingface.co/datasets/pkavumba/balanced-copa) | CC-BY-4.0 | dataset card metadata | yes |
| Social-situation reasoning | [`allenai/social_i_qa`](https://huggingface.co/datasets/allenai/social_i_qa) | **not independently confirmed in this pass** | the shipped Hugging Face model cards state CC-BY-4.0, but the dataset's own card carries no licence field or tag at all; this document could not independently verify that claim from a primary source | **not independently confirmed** -- flagged for follow-up |

These four rows use their own gold labels only; no teacher model relabelled them.

### Self-distillation

Every row's distillation target (soft probability, hidden-state and top-vocabulary signal) is the *previous shipped
revision's own* 36-layer forward pass -- this project's model, not a third party's. A revision cannot be its own
teacher before it exists, so the very first 36-layer adapter was distilled from the full 40-layer checkpoint's own
output instead, which is this project's own weights as well. No external output is involved at any point in this
step.

### Calibration data (NVFP4 quantisation, QAT)

`nvfp4_calib.json`'s per-layer activation maxima are measured on 128 rows drawn from the same training pool audited
above -- not a separate external corpus, so it adds no licence surface beyond what the pool already carries.

### Teacher models (gold-label correction)

About 7% of the core pool's rows take their gold label from a frontier LLM instead of the pool's own label, used only
where the two disagreed, as a single answer letter -- no generated rationale or free text from any teacher model is
reproduced in either checkpoint's weights or in either Hugging Face card.

| Teacher | Used for | Licence | Commercial-use / distillation terms (read from the licence file itself) |
|---|---|---|---|
| Kimi K3 (`moonshotai/Kimi-K3`) | ~7% of core-pool rows' gold label, where it disagreed with the pool's own label | Modified MIT | Unrestricted use, modification, and creation of derivative works, including training another model on its output. Two revenue-gated conditions, neither of which applies to this project: a separate agreement with Moonshot AI is required only if a "Model as a Service" business built on it clears $20M/12-month revenue, and a UI attribution requirement applies only above 100M monthly active users or $20M/month revenue. Internal use (this project's own training pipeline, not resold as a service) is explicitly exempt from both. |

**No output from a closed, proprietary model -- Claude (Opus, Fable, or any other Anthropic model), GPT, or any other
model whose weights or terms are not publicly available -- was used anywhere in this round's training data.** The
project's standing rule explicitly excludes those providers as data sources regardless of their own usage terms.

### Candidate teacher for a future round, not yet used

`DeepSeek V3.2` (`deepseek-ai/DeepSeek-V3.2`, plain MIT licence, no revenue-gated condition) was evaluated against
the current checkpoint on this project's own held-out panel and found at least as strong as Kimi K3 on the two
judgment types the weak-category mixture above targets, at roughly a quarter of Kimi K3's per-question cost. It has
not been used to generate any row shipped in this round; if a future round adopts it, that round's own section below
will record it the same way Kimi K3 is recorded above.

### Licence summary for this round

Every source added in this round -- the four weak-category datasets, the Kimi K3 single-letter relabelling, and the
self-distillation signal -- is licensed for commercial use and for training a derivative model, with no restriction
on redistributing the resulting weights (one row, `allenai/social_i_qa`, is flagged above as not independently
confirmed and should be treated as unverified until re-checked). Three sources carried over from earlier rounds of
the core pool are **not** commercial-clear: RACE and SciQ are non-commercial research data (already stated in both
checkpoints' Hugging Face "Licence and data terms" sections), and DREAM carries the same restriction (newly found
while writing this page; not previously stated in either card). A fourth, AG News, is also non-commercial-only and
was not previously called out either. None of these four is new to this round -- all four predate it and were
already baked into the training pool this round built on. No output of a closed, proprietary model was used.

A handful of smaller legacy families (TREC, Yahoo Answers Topics, QNLI, and `tweet_eval`'s `offensive` config) were
not individually re-verified against a primary licence source in this pass; treat them as unaudited rather than as
confirmed either way until someone checks them directly.

## v0.5.0: a commercially-clear training line (`prismyra-decision-35b-a3b-fp8` revision `3c88c546`, `prismyra-decision-35b-a3b-nvfp4` revision `9d08c9b4`)

Starting with this round, the project ships a second checkpoint line whose adapter is trained from the unmodified
base model using only sources confirmed clear for commercial use and for training a derivative model. This section
documents that line's pool end to end, rather than describing only what changed, because the pool is a fresh build
and the "additive" convention above does not apply to it.

Three sources that were part of every earlier round's pool are not used here at all, because they carry
research-only terms: **RACE** (custom, non-commercial -- CMU's own data agreement), **SciQ** (CC-BY-NC-3.0), and
**DREAM** (custom, non-commercial -- "intended for non-commercial research purpose only"). A fourth,
**social_i_qa**, is not used because its own Hugging Face card carries no licence field at all and this project
could not independently confirm the commercial-use claim made on it by the earlier cards. Where these sources
previously trained a capability, self-authored synthetic data (below) covers the gap instead.

### Sources already documented above, reused for this line

Same dataset, same licence as already recorded earlier in this page; only the row counts differ.

| Family | Source | Licence (SPDX) |
|---|---|---|
| CommonsenseQA rows | `tau/commonsense_qa` | MIT |
| AQuA-RAT rows | `deepmind/aqua_rat` | Apache-2.0 |
| MMLU / MMLU auxiliary-train rows | `cais/mmlu` | MIT |
| Cosmos QA rows | `allenai/cosmos_qa` | CC-BY-4.0 |
| ARC rows | `allenai/ai2_arc` | CC-BY-SA-4.0 |
| QuALITY rows | `nyu-mll/quality` | CC-BY-4.0 |
| HotpotQA (comparison slice) rows | `hotpotqa/hotpot_qa` | CC-BY-SA-4.0 |
| BIG-bench logical-deduction rows | BIG-bench (`logical_deduction` task) | Apache-2.0 |
| BoolQ rows | `google/boolq` | CC-BY-SA-3.0 |
| Financial-news sentiment rows | `zeroshot/twitter-financial-news-sentiment` | MIT |
| General-tweet sentiment rows | `cardiffnlp/tweet_eval` (`sentiment` config only) | CC-BY-3.0 |
| Causal reasoning rows | `pkavumba/balanced-copa` | CC-BY-4.0 |
| Intent classification rows | `clinc/clinc_oos`, `PolyAI/banking77` | CC-BY-3.0, CC-BY-4.0 |

### New sources for this line

| Family | Source | Licence (SPDX) | Licence text | Commercial use |
|---|---|---|---|---|
| Emotion valence | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Apache-2.0 | dataset card metadata | yes |
| Rule-based logical reasoning | [`tasksource/ruletaker`](https://huggingface.co/datasets/tasksource/ruletaker) | Apache-2.0 | dataset card metadata | yes |
| bAbI-style reasoning (17 sub-tasks) | [`tasksource/babi_nli`](https://huggingface.co/datasets/tasksource/babi_nli) | the card's own tag is the unqualified `bsd` | dataset card metadata; the upstream bAbI project (Meta/Facebook) is BSD-3-Clause, not restated with a clause count on this card | yes |
| Sentence-composition QA | [`allenai/qasc`](https://huggingface.co/datasets/allenai/qasc) | CC-BY-4.0 | dataset card / project page | yes (attribution required) |
| Qualitative-relationship reasoning | [`allenai/quartz`](https://huggingface.co/datasets/allenai/quartz) | CC-BY-4.0 | dataset card body text | yes (attribution required) |
| Poem sentiment | [`google-research-datasets/poem_sentiment`](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) | CC-BY-4.0 | dataset card / project page | yes (attribution required) |
| Wikipedia-derived long-document reading and topic questions | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`), with DeepSeek V3.2 authoring the question and Kimi K3 blind-verifying it | CC-BY-SA-4.0 | Wikimedia's own terms of use | yes (share-alike: a redistributed *copy* of the source text must carry the same licence; this does not require releasing the model weights under CC-BY-SA, and the source text itself is not reproduced in the released weights) |

**Confirmed in a follow-up check against the primary licence text:**

| Family | Source | Note |
|---|---|---|
| Causal-reasoning rows (e-CARE) | e-CARE ([waste-wood/e-CARE](https://github.com/waste-wood/e-CARE); Hugging Face reuploads `12ml/e-CARE`, `sjcfr/e-care`) | MIT, from the original repository's `LICENSE` file (Copyright (c) 2022 HIT-SCIR); commercial use permitted. Human labels only (the dataset's own answer); no teacher output is used for these rows. 1,244 rows. |
| QNLI (via GLUE) | `nyu-mll/glue` (`qnli` config), a reformulation of SQuAD v1.1 | CC-BY-SA-4.0, from SQuAD's own licence statement ([SQuAD explorer](https://rajpurkar.github.io/SQuAD-explorer/): "distributed under the CC BY-SA 4.0 license"); commercial use permitted, and share-alike applies to redistributed copies of the data, not to model weights. 13 rows. |

### Self-authored synthetic replacements

These rows carry no third-party text; DeepSeek V3.2 authors the scenario, question and options (or, for the code
family, the function and its input), and Kimi K3 performs a blind second check before a row is kept. Licence: n/a
(not third-party data); commercial use: yes.

| Family | Replaces | What it covers |
|---|---|---|
| Social-situation commonsense (self-authored) | `social_i_qa` (dropped, licence not independently confirmed) | social-situation reasoning, same question shape as the dropped source |
| Dialogue reading comprehension (self-authored) | DREAM (dropped, non-commercial) | multi-turn dialogue reading comprehension |
| Code-output tracing, execution-verified | an early version of this same family that this project found to be 22.3% wrong (gold decided by two models agreeing, not by running the code); superseded by running the self-authored function in a sandboxed subprocess and keeping only rows where four independent runs agree with the stated output | predicting a function's return value for a given input |
| Tool-use response selection (self-authored) | no prior source (new capability for this project) | choosing the correct tool call among invented, schema-described alternatives |

### Self-distillation

As in earlier rounds, every row's distillation target is this project's own model, not a third party's -- but for
this line specifically, the signal comes from the clean, unmodified 40-layer base checkpoint, read on inputs from
this line's own commercially-clear pool. No earlier adapter (all of which were trained in part on the research-only
sources this line excludes) is used as a teacher, starting point, or initialization anywhere in this line.

### Calibration data (NVFP4 quantisation)

The NVFP4 activation-maxima calibration file for this line was measured on rows drawn from this line's own
commercially-clear pool, not reused from any earlier round's calibration file.

### Teacher models

Only Kimi K3 and DeepSeek V3.2 are used anywhere in this line's data, for scenario/question authoring and blind
verification as described above and in the per-family rows. **No output from a closed, proprietary model --
Claude, GPT, or any other model whose weights or terms are not publicly available -- was used anywhere in this
line's training data.** No generated rationale or free text from either teacher model is reproduced in either
checkpoint's weights or in either Hugging Face card.

### Licence summary for this line

Every source behind this line's adapter is licensed for
commercial use and for training a derivative model, with no restriction on redistributing the resulting weights.
RACE, SciQ, DREAM and social_i_qa -- all four non-commercial or unconfirmed in earlier rounds' pools -- are not used
in this line at all; DREAM and social_i_qa's roles are covered by self-authored synthetic data instead, and RACE and
SciQ's roles are not directly replaced (see each checkpoint's own card for where that shows up in evaluation).

## v0.5.1: long-document variable tracking and fact chaining (`prismyra-decision-35b-a3b-fp8` revision `556e7094`)

This revision adds two new training families on top of the v0.5.0 line's pool above, both aimed at a long-document
weakness that line's own evaluation showed plainly. Neither family uses a third-party dataset of pre-written
questions; both are generated end to end by a program this project wrote, with the correct answer decided
mechanically by that same program rather than by any model.

### New sources for this revision

| Family | What it covers | Source of the document text | Licence (SPDX) | Commercial use |
|---|---|---|---|---|
| Variable tracking | Following several independent chains of variable reassignment through a long document and identifying which variable currently holds a stated value | Filler paragraphs drawn from [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (already listed above for this line), with the variable-chain text itself authored by this project's own generator, not drawn from Wikipedia | CC-BY-SA-4.0 for the filler text (Wikimedia's own terms of use; share-alike applies to a redistributed *copy* of the source text, not to the released model weights, and the source text itself is not reproduced in the weights) | yes |
| Fact chaining | Following a chain of transfer records in the reverse direction (where a fact currently on record originally came from) through a long document | Same filler pool as Variable tracking, same generator, same licence terms | CC-BY-SA-4.0 (filler only, as above) | yes |

Every row in both families is checked by the generator itself, not accepted on trust: every clue sentence and
every answer-choice string the row's correctness depends on must appear verbatim in the assembled document, checked
line by line, and a row that fails this check is discarded rather than kept. No teacher model is called for either
family, and no row in either family carries a distillation signal (see Self-distillation below for why that is a
deliberate choice, not an oversight). This project independently checked that the documents these two families
generate do not overlap the project's own long-context evaluation data (no exact document match, no 80-character
substring match across a stratified sample, and no generated variable or location name appearing verbatim in the
evaluation data).

### Self-distillation (updated for this revision)

As with every other row in this line's pool, the two new families above carry no distillation signal at all --
their correctness comes only from the generator, as described above. For every other row already documented under
v0.5.0, the distillation target changes with this revision: it is no longer the clean, unmodified base checkpoint,
but this project's own previously shipped checkpoint (this same repository's v0.5.0 revision), read on the same
commercially-clear inputs. This is still this project's own model's output, not a third party's and not an earlier
research-trained checkpoint's; it simply moves the reference point from before this line started to the most recent
link in this line's own chain.

### Licence summary for this revision

Both new families add no new third-party dataset: the only external text they use (Wikipedia filler paragraphs) was
already part of this line's licence accounting under v0.5.0, and the variable-tracking and fact-chaining content
built around that filler is this project's own generated text, not sourced from anywhere. Every source behind this
revision's adapter remains licensed for commercial use and for training a derivative model, with no restriction on
redistributing the resulting weights.
