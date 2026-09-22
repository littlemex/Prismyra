# Tetris: a decision loop, and a conclusion this example got wrong once

Every other measurement in this repository asks questions about a document somebody else wrote. This one asks about a
state the previous answer produced, which is the other thing people build with a model that returns a typed decision in
one pass: a loop that looks at a state, picks one of a few dozen moves, and lives with the consequence.

The fit looks ideal. A tetromino has up to 34 legal placements, so one board is one context and every placement is one
row of one batch -- 34 questions per context, which is the case the crossover measurement says the read-out wins
comfortably.

It plays. What it needs is for the state to be in a form it can read, and finding that out took four framings and two
controls, because the first answer this example published was wrong.

```bash
python examples/tetris/play.py --agents heuristic,random --pieces 60 --games 8
python examples/tetris/play.py --agents readout,readout_picture,readout_features,readout_score --pieces 60 --games 3
python examples/tetris/play.py --perception 12     # is the board being read, as text and as an image
python examples/tetris/play.py --diagnose 12       # does the ranking resemble one that plays
```

## What happened

Same seeds, same pieces in the same order. A game ends when a piece will not fit, or when the sixty pieces run out.
`regret` is the mean heuristic value given up per decision against the best placement available, which is a per-decision
measure where pieces placed is a per-game one.

| agent | what it is shown | placed / 60 | rows cleared | regret | against random |
|---|---|---|---|---|---|
| `heuristic` | exact features, fixed weights | **60.0 +/- 0.0** | **20.5 +/- 1.5** | 0.00 | p = 0.0002 |
| `readout_score` | the candidate table **with its score** | 57.3 +/- 2.3 | 9.0 +/- 3.6 | 2.39 | -- |
| `readout_features` | the candidate table, no score | **51.7 +/- 2.1** | **6.0 +/- 1.7** | 4.00 | -- |
| `readout` | the board drawn, placements described | 26.3 +/- 2.7 | 0.3 +/- 0.5 | 14.61 | p = 0.16 |
| `readout_outcome` | the board each placement would produce | 24.0 +/- 6.2 | 0.2 +/- 0.4 | 22.49 | p = 0.68 |
| `readout_picture` | the board **as an image** | 24.5 +/- 4.2 | 0.0 +/- 0.0 | 15.68 | p = 0.88 |
| `random` | -- | 23.7 +/- 3.4 | 0.2 +/- 0.4 | 17.15 | -- |

`readout_features` is the interesting row. The five numbers for every candidate are in the shared context and the score
is not, so the model still has to decide what those numbers are worth -- and it survives 51.7 pieces of 60 and clears
six rows where random survives 23.7 and clears none.

The three rows below it are **not distinguishable from random**: exact two-sided p of 0.16, 0.68 and 0.88 over all
arrangements of their games. That is why `random` is in the table. Without it, "26 pieces placed at 537 ms a move" reads
like a working agent with a latency figure.

`readout_picture` is the one to look at twice. It reads the board -- see below -- and it still cannot play, so better
perception on its own bought nothing here.

## Where the failure actually is, after two wrong answers

This file has published two conclusions and both were wrong. Keeping them is cheaper than the mistakes were.

**First**, from the heuristic, `readout`, `readout_outcome` and `random` alone:

> Showing the resulting board did not help, so the failure is not that the model cannot work out which position a phrase
> produces. It cannot judge the positions.

Both reviewers rejected the inference. Showing a branch the board its placement would produce removes the need to
**imagine** a placement; it does not remove the need to **read a grid of text**.

**Second**, after the features framing and the text perception probe:

> The model can judge Tetris positions. It cannot read an ASCII board.

Half right, and the wrong half was the diagnosis. "Cannot read an ASCII board" is not the same as "cannot see a board"
-- this model has a vision tower and this package puts images in a context, and the board had only ever been handed
over as characters. That question was asked from outside and it was the right one.

So the same probe was run again with the board drawn as an image, same questions, same thresholds, same read-out:

| channel | right on mechanically-answerable questions | answering "no" to everything |
|---|---|---|
| the board written out in characters | 92 / 132 = **69.7%** | 69.7% |
| the board drawn as an image | 112 / 132 = **84.8%** | 69.7% |

The text figure is not near the constant-answer baseline, it **is** the constant-answer baseline, to a tenth of a point
(P(>= 92) = 0.54). The image figure is far above it (P(>= 112) = 0.00005), and beats text on 8 of the 9 boards where
they differed (sign test p = 0.039).

Why a grid written out in text defeats a model that reads images: tokenisation merges runs of dots into pieces whose
boundaries differ from row to row, so nothing indicates that the fifth character of one line sits above the fifth
character of the next; text carries one-dimensional positions in this model's position scheme while image patches carry
two-dimensional ones; and the vision tower is not involved at all when the board is characters.

**And it still does not play.** `readout_picture` places 24.5 +/- 4.2 of 60 and clears nothing -- wins 0.54 of pairings
against random, exact two-sided p = 0.88. Reading the board was necessary and is not sufficient.

That locates the failure precisely, which neither earlier conclusion did:

* it can **see** the board when the board is a picture -- 84.8% on holes, heights and emptiness;
* it can **judge** a position when the aggregates are handed to it -- 51.7 of 60 pieces from the features, with the
  score withheld;
* it cannot **get from one to the other**. Judging a placement needs five aggregates over ten columns computed for each
  of 34 candidates and then compared. Per-question accuracy of 84.8% does not survive that many combinations.

`readout_score` is the positive control that keeps the mechanism out of it: the candidate table carries the weighted
score, so the best placement is the largest number in a column, and the read-out places 57.3 of 60.

## Why the ranking diagnostic did not catch this

`--diagnose` compares the model's ranking of the placements against the heuristic's on boards the heuristic reached.
Over 12 boards the mean rank correlation was +0.096 +/- 0.070 for described placements and -0.105 +/- 0.083 for shown
outcome boards, neither distinguishable from zero, with individual boards from -0.484 to +0.467.

It said correctly that there was no signal in those two framings, and it could not say why, because it only compares
against one policy. The perception probe is what localises the failure, and it is cheaper. **Measure whether the input
is being read before measuring whether the decision is good.**

One number from that diagnostic was reported and is now withdrawn as uninformative rather than wrong: the described
framing's top pick landed in the heuristic's best three five times in twelve where chance gives 1.99, an exact tail
probability of 0.031. It was one of four comparisons on the same twelve boards, which Holm correction puts near 0.12,
and its null assumed the model's pick is uniform over placements when the model may simply prefer certain columns or
phrasings whatever the board. Since the perception probe shows the board is not being read, a positional preference is
the likely explanation and there is no reason to pursue it.

## What is worth knowing from this

* **The mechanism did what it claims.** 34 placements scored in one forward pass, 537 to 680 ms depending on framing,
  against 34 separate calls. That holds in every row of the table, including the rows where the agent cannot play.
* **A mechanism that delivers a decision cheaply cannot make the decision good.** In this task the representation was
  worth about 27 pieces of survival and six rows, which is far more than anything in the engine has been worth -- and
  the representation that paid was numbers, not a clearer picture.
* **Fixing perception did not fix the decision.** The image channel reads the board and plays at the floor. Where a
  chain has three links, measuring the first and the last does not locate the break.
* **A floor belongs in every table like this**, and so does a positive control. The floor says whether the agent is
  playing; the control says whether a failure belongs to the mechanism or to the model.
* **Ranking by P(yes) is already ranking by log-odds** here. A reviewer asked for the latter; a Boolean read-out
  softmaxes over exactly the two declared options, so p(yes) + p(no) = 1 and the logit is monotone in p(yes). Same
  ranking.

## What is still not established

* The features framing hands over the heuristic's own five features, so it tests judgement **given that ontology**. A
  cleaner version gives column heights and holes per column -- nearly lossless for this game, and not a feature set
  chosen by the reference.
* No plain generative agent was run: one ordinary prompt per move, all placements listed, the model reasons and names
  one. Now that the image channel is known to read the board, the interesting version of that is generative **from the
  image**: if reasoning tokens turn 84.8% per-question perception into playable judgement, the gap identified above is
  about one-pass scoring rather than about the model.
* The step that is actually missing has not been probed directly. Ask the model, from the image, for the aggregates it
  would need -- how many holes in total, which column is tallest, how bumpy -- and score those against the truth. The
  per-cell questions are answered at 84.8% and the aggregates are what judging needs, so that is where the accuracy
  should be measured next.
* The diagnostic's boards come from the heuristic's play, so they are cleaner than the boards a weak agent meets.
* `readout_outcome` is nominally below random (24.0 against 23.7) and that is variance, not an anti-signal: p = 0.68.

## Files

* `game.py` -- the board, the seven pieces, legal placements, clears. No rendering loop and no scoring curve.
* `agents.py` -- `Random`, `Heuristic`, `ReadOut`, `ReadOutOutcome`, `ReadOutFeatures` (with the score as a positive
  control), the `perception` probe, the `agreement` diagnostic and `regret`.
* `play.py` -- games, `--perception`, `--diagnose`, and an exact rank test against the floor.
* `../../tests/test_tetris.py` -- the rules, checked without a model, because a comparison built on wrong rules is
  measured precisely and means nothing. Includes the simplification everything rests on: a piece falls straight down and
  cannot slide under an overhang.
