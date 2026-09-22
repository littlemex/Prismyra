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
python examples/tetris/play.py --agents readout,readout_features,readout_score --pieces 60 --games 3
python examples/tetris/play.py --perception 14     # is the board being read at all
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
| `random` | -- | 23.7 +/- 3.4 | 0.2 +/- 0.4 | 17.15 | -- |

`readout_features` is the interesting row. The five numbers for every candidate are in the shared context and the score
is not, so the model still has to decide what those numbers are worth -- and it survives 51.7 pieces of 60 and clears
six rows where random survives 23.7 and clears none.

The two rows below it are **not distinguishable from random**: exact two-sided p of 0.16 and 0.68 over all arrangements
of the six games each. That is why `random` is in the table. Without it, "26 pieces placed at 537 ms a move" reads
like a working agent with a latency figure.

## The conclusion this example published first, and why it was wrong

The first version of this file had only the top and bottom of that table -- the heuristic, `readout`, `readout_outcome`
and `random` -- and concluded:

> Showing the resulting board did not help, so the failure is not that the model cannot work out which position a phrase
> produces. It cannot judge the positions.

Both reviewers rejected the inference and they were right. Showing a branch the board its placement would produce
removes the need to **imagine** a placement. It does not remove the need to **read a grid of text**, and those are
different failures. The conclusion needed a measurement it did not have.

Two were missing, and both are now here.

**Is the board being read at all?** `--perception` asks questions about a rendered board whose answers are mechanically
known and depend on no policy: does it have more than *k* holes, is column *a* taller than column *b*, is column *x*
empty. Thresholds sit either side of the true count so that a constant answer does not score well.

> **110 of 154 right, 71.4%, over 14 boards -- against 69.5% for answering "no" to everything.** P(>= 110) = 0.34
> against that baseline, so this is indistinguishable from not reading the board.

**Can the read-out rank anything at all?** `readout_score` is the positive control: the candidate table carries the
weighted score, so the best placement is the largest number in a column and choosing it needs no judgement. It places
57.3 of 60. So the mechanism and the read-out are sound, and nothing about scoring 34 candidates in one pass is at
fault.

Those two together make the corrected conclusion, and it is the opposite of the first one:

> The model can judge Tetris positions well enough to play. It cannot read an ASCII board. The floor scores are a
> perception failure, and giving it the same state as numbers recovers most of the play.

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
* **A mechanism that delivers a decision cheaply cannot make the decision good**, and it cannot make an unreadable input
  readable. In this task the representation was worth about 28 pieces of survival and six rows -- far more than anything
  in the engine.
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
  one. If that plays from the drawn board, then ASCII perception is recoverable with enough tokens spent on it and the
  perception result is about one-pass scoring rather than about the model. It is slow and it is the next thing worth
  doing.
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
