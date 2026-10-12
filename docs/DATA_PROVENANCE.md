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
