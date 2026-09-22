# Accuracy

Every other figure in this repository is a latency figure. A wrong answer can be made arbitrarily fast, so this page is
the one that decides whether any of them mean anything.

```bash
pip install -e ".[dev]" && pip install "datasets>=3.0"
python evals/run.py --task race --limit 150 --seed 11
python evals/run.py --task boolq --limit 400 --seed 11
python evals/run.py --task unfair_tos --limit 250 --seed 7 --methods readout,thresholded,majority
```

## What is compared

**readout** is what this package does: one traversal of the context, then one row per question, scoring the declared
options at the branch's final position.

**generate** is the same weights and the same prompt, generating the answer and reading the text, once per question.
Deliberately not a second serving engine: a second process cannot hold a second copy of these weights on one card, and a
different engine would change the read-out and the implementation at once, leaving any difference unattributable. The
model opens every answer with a `<think>` block, so the prompt closes it -- the read-out does no reasoning either, and a
baseline that reasons is a different and much more expensive method.

**majority** always answers the commonest label, taken from the training split. Taken from the scored slice it would be
an oracle rather than a baseline.

## The read-out answers as well as generating does, on a short context

| task | contexts | questions | read-out | generation | difference, 95% interval | majority |
|---|---|---|---|---|---|---|
| RACE-middle | 150 | 579 | 94.8% | 95.3% | -0.5% [-1.6%, +0.5%] | 27.5% |
| BoolQ | 400 | 400 | 89.5% | 88.8% | +0.8% [-1.3%, +2.7%] | 65.2% |

Both intervals span zero. On these slices the two methods cannot be separated, and that -- not a win -- is the claim
worth making, because it is what makes a comparison of speed a comparison of anything at all.

Read the heading's last four words, though. A RACE article is a few hundred tokens, and that is the only length at
which this was ever measured.

## On a long context they separate, and not in this package's favour

The same 159 questions with the same right answers, asked about the same articles, with each article surrounded by
other articles from the same slice until the context reaches a length. `evals/run.py --bury N` does that. Nothing about
the question changes; only how much competing material is in front of it.

| context tokens | read-out | generation | difference, 95% interval |
|---|---|---|---|
| 407 (median, unpadded) | 95.0% | 94.3% | +0.6% [-1.3%, +3.0%] |
| 6,104 | 83.6% | 86.2% | -2.5% [-5.6%, +0.6%] |
| 10,199 | 82.4% | 87.4% | **-5.0% [-8.6%, -1.3%]** |

**The last interval excludes zero.** It is the first comparison in this project that separates the two methods, and it
separates them against the read-out: burying the answer costs it 12.6 points and costs generation 6.9.

Two things follow and they point in different directions.

Some of the fall is the model's and nothing to do with this design -- generation loses 6.9 points on the same material,
so roughly half of what the read-out loses is simply the difficulty of finding an answer in a haystack. But the
*widening* is the mechanism's. The two methods share everything up to the last position of the branch: the same weights,
the same cache, the same fork. The only difference is how the answer is taken out -- the read-out scores the declared
options at one position, while generation emits tokens that are then parsed, and the parser accepts an exact match, a
delimited one, a prefix, or the option merely being mentioned. A flattening distribution costs a single-position argmax
more than it costs four chances at a string.

That is a hypothesis, not a finding. What would settle it is comparing generation's *first* token against the
read-out's argmax on the same questions: if they agree and the accuracy still differs, the parser's extra chances are
the whole story, and if they disagree the read-out's position is.

**What to do with this.** Padding is an adversarial way to make a context long -- competing articles are harder than a
long single document, which is the ordinary case. So this is a lower bound on long-context accuracy rather than a
measurement of it. But it is the only measurement there is, and the package's speed and memory figures go out to 24,327
tokens while its accuracy figures stopped at a few hundred. If you are asking many questions about a long document, this
is the number to plan against, and the read-out's advantage there is speed bought at a measured cost.

The padding length has a ceiling set by the slice: forty RACE articles of about 400 tokens cannot pad any one of them
past about 16,000, which is why asking for 18,000 produced 10,199.

The interval comes from resampling whole contexts, not questions. Questions about one article share a passage and a
topic, so treating them as independent draws would make every interval several times too narrow. It is also why the
numbers are stated with intervals at all: the same BoolQ measurement at sixty passages and a different seed gave 95.0%
for both methods rather than 89.5% and 88.8%, a ten point swing from the sample alone.

## A decision loop, where the mechanism works and the model does not

Everything above asks about a document. `examples/tetris/` asks about a state the previous answer produced, which is the
other thing a one-pass typed decision gets built into: a loop that looks at a state, picks one of a few dozen moves, and
lives with the consequence. A tetromino has up to 34 legal placements, so one board is one context and every placement
is one row -- the shape this design is best at.

| agent | pieces placed | rows cleared | ms per move |
|---|---|---|---|
| five features and five weights | **261.0 / 300** | **96.0** | 0.4 |
| read-out, placements described | 26.0 / 60 | 0.0 | 555.8 |
| read-out, resulting board shown | 20.5 / 60 | 0.0 | 809.2 |
| random placements | 22.7 / 300 | 0.3 | 0.1 |

**The model agents are at the random floor**, and showing them the board each placement would produce does not help, so
this is not prompt wording. Ranking the placements directly rather than through a game gives a rank correlation with the
arithmetic reference of +0.096 +/- 0.070 and -0.105 +/- 0.083 for the two framings -- neither distinguishable from zero.

The floor is in that table for a reason. Without it, "26 pieces placed at 556 ms a move" reads as a working agent with a
latency figure, and it is a latency figure for not playing. The mechanism did its part: 34 placements scored in one
forward pass, where 34 calls would have cost far more. It cannot make the decision good, and a task where decisions
compound is where that distinction is most expensive to miss. `examples/tetris/README.md` has the framings, the exact
tail probability of the one marginal number, and what a real attempt would change.

## Speed is entirely a question of how many questions share a context

Measured in the same runs, against that same unbatched generation loop:

| task | questions per context | read-out against generation |
|---|---|---|
| RACE-middle | 3.9 | 2.2x faster |
| BoolQ | 1.0 | **0.7x -- slower** |

That is the crossover from the README, measured on real tasks rather than a synthetic sweep. One question per context
and the read-out loses: it pays a context pass and then a branch pass, where generation pays one pass and a few
tokens.

**These ratios are an upper bound on the advantage over generation, not a floor**, and the harness says so where it
prints them. The baseline is a plain Python decode loop with no batching and no prefix sharing. A serving engine that
batched the questions and shared the article's prefix would narrow the gap, and the honest place to compare is against
one of those. That comparison is the missing half of this page.

## Where it does not work, which matters more

On LexGLUE's unfair terms-of-service task -- one contract clause, eight independent unfairness types -- the read-out
does not work, and neither does generation.

| | recall | precision | F1 | accuracy |
|---|---|---|---|---|
| read-out | 60.0% | 5.8% | 10.5% | 84.7% |
| generation | 100.0% | 5.8% | 11.0% | 86.5% |
| majority (train) | 0.0% | -- | 0.0% | **98.5%** |

Read that by F1. About 1.5% of the answers are positive, so answering no to everything scores 98.5% accuracy and beats
every method on the column most people look at first.

Recall of 60% at a precision of 6% says what is wrong: the model says yes to nearly everything. Asked whether a clause
limits liability, it will not rule the clause out. The same model generating the answer does the same, so this is not
the read-out's limit -- it is the absence of a decision point. Taking the larger of two probabilities stands the
threshold at 0.5, and 0.5 is the wrong place to stand when one answer in seventy is yes.

**So a general-purpose closed-question classifier is not something that comes for free, and this is the measurement that
shows it.** Where the answer is in the text and the classes are balanced, the read-out is as good as the model. Where
the interesting class is rare, it needs a decision point, and a decision point needs labels.

## What helps, and what does not

**Subtracting a content-free prior** (`prismyra.calibration`) was the cheapest candidate and it does not pay:

| task | read-out | options only | whole question |
|---|---|---|---|
| RACE, 60 articles | 93.0% | 92.2% | 77.9% |
| BoolQ, 60 passages | 85.0% | 80.0% | 68.3% |
| unfair-ToS, 60 clauses (F1) | 9.9% | 8.0% | 11.3% |

Subtracting the whole question asked against a placeholder context is much worse everywhere, and the reason is that it
is not a bias correction. The question and its options are still in that prompt, so what comes back contains real
knowledge about the answer and subtracting it subtracts the knowledge -- pointwise mutual information between context
and answer rather than a thumb off the scale. Subtracting only the option list, nearer to the token bias alone, is
roughly neutral and costs 4% more time. Neither is on by default.

**Choosing the decision point per question** (`prismyra.thresholds`) is what works. One number per question, fitted on
labelled answers from a split it is not then scored on, on a fitting slice of 800 clauses:

| | recall | precision | F1 | macro F1 | accuracy |
|---|---|---|---|---|---|
| read-out | 60.0% | 5.8% | 10.5% | 39.5% | 84.7% |
| read-out with fitted cuts | 60.0% | **39.1%** | **47.4%** | **50.9%** | 98.0% |
| majority (train) | 0.0% | -- | 0.0% | 0.0% | 98.5% |

Recall does not move. Precision goes from 5.8% to 39.1%, which is the trade a rare class wants: the same positives
found, far fewer things wrongly called positive. Per question it helps five of the eight, ties one, and costs two --
and one of those two has a single positive label in the scored slice, so it is one answer rather than a trend.

This is supervised operating-point selection, which is textbook cost-sensitive decision making rather than a new idea.
What recommends it is the price: no gradient, no device, no second model, and the inputs are answers already given
beside what turned out to be true, which a caller that logs its requests already holds. That makes it the floor an
expensive mechanism has to clear, not something to be impressed by. It is also not label-free -- it needs labels, and
a fine-tune given the same labels will go considerably further.

### How many labels it takes, which is the number worth knowing

The mechanism requires five positive examples per question before it will fit a cut at all, and says which questions it
refused and why. At a 1.5% base rate that guard binds hard:

| clauses fitted on | answers fitted on | questions given a cut | F1 |
|---|---|---|---|
| 200 | 1,600 | 3 of 8 | 11.3% |
| 800 | 6,400 | 8 of 8 | **47.4%** |
| 2,000 | 16,000 | 8 of 8 | 46.6% |

Two things to read off that. Below roughly eight hundred labelled documents this does almost nothing here, because at
two hundred clauses five of the eight unfairness types each appear fewer than five times, and the mechanism declines
rather than inventing a cut from one or two examples.

And **it saturates there**. Two and a half times as many labels buys nothing -- 46.6% against 47.4%, well inside the
noise of thirty positives. That is the more useful half of the finding, because it says where the remaining gap is not:
more labels will not move a threshold that has already found its place. What is left is the order the read-out puts the
probabilities in, and a threshold cannot change an order. Anything aimed at the rest of this gap has to change the
scoring itself -- a trained head on the branch's final hidden state is the obvious candidate -- and the bar it has to
clear is 47.4% pooled and 50.9% macro, not the read-out's 10.5%.

That guard is also what caught a figure this project nearly published. An earlier version of the harness fitted its own
cuts with no such guard and reported F1 rising from 10.5% to 33.7%. Seven of those eight cuts were fitted on between
zero and three positive examples. Two independent reviews found that the harness and the package were running
different code before they found anything else, and the figure was withdrawn rather than explained.

### Read these as preliminary

Thirty positive labels in two thousand answers. At that count one label moves recall by about three points, so the
interval around 47.4% is wide and this page does not pretend otherwise. What would close it: the LexGLUE test split,
scored once, with a paired bootstrap over source documents rather than clauses.

## Honest limits of this page

* Every figure is one run on one L40S. The harness takes `--repeat` and `--seed`; neither has been used in anger.
* Three tasks, chosen for what they expose. Two of them are balanced multiple choice, which is the easy case.
* The generation baseline's latency is not a serving engine's latency, and every ratio built on it is marked as an upper
  bound where it appears.
* The unfair-ToS figures come from a validation slice that also diagnosed the failure the threshold mechanism
  addresses. A slice that shaped a mechanism has become development data, so the mechanism's final figure belongs on the
  test split, scored once.
* Absolute performance on unfair-ToS is far below published supervised systems for that task, and no claim here is a
  claim to compete with them. Once labels are used at all, this is on the supervised spectrum; what is being argued is
  cost, not label-freedom.
