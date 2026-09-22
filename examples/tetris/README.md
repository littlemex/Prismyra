# Tetris: the shape this design is for, on a task it turns out not to do

Every other measurement in this repository asks questions about a document somebody else wrote. This one asks about a
state the previous answer produced, which is the other thing people build with a model that returns a typed decision in
one pass: a loop that looks at a state, picks one of a few dozen moves, and lives with the consequence.

The fit looks ideal. A tetromino has up to 34 legal placements, so one board is one context and every placement is one
row of one batch -- 34 questions per context, which is the case the crossover measurement says the read-out wins
comfortably. Whether it *plays* is a different question, and the answer is no.

```bash
python examples/tetris/play.py --agents heuristic,random --pieces 300 --games 3
python examples/tetris/play.py --agents readout,readout_outcome --pieces 60 --games 2
python examples/tetris/play.py --diagnose 12
```

## What happened

Same seeds, same pieces in the same order, so a difference belongs to the agent. A game ends when a piece will not fit.

| agent | pieces placed | rows cleared | ms per move |
|---|---|---|---|
| `heuristic` -- five features and five weights | **261.0 / 300** | **96.0** | 0.4 |
| `readout` -- placements described | 26.0 / 60 | 0.0 | 555.8 |
| `readout_outcome` -- resulting board shown | 20.5 / 60 | 0.0 | 809.2 |
| `random` -- the floor | 22.7 / 300 | 0.3 | 0.1 |

**The model agents are at the floor.** Twenty-six pieces against random's twenty-three, and no rows cleared by either
framing against the heuristic's ninety-six. Five lines of arithmetic play Tetris and a 35-billion-parameter model
does not.

Read the floor before anything else in that table, because without it "26 pieces placed, 555 ms per move" reads like a
working agent with a latency figure. It is a latency figure for not playing. This is why `random` is in the file: an
example that shows off a mechanism on a task it fails, without saying so, is worse than no example.

## Two framings, so the failure can be attributed

A floor score has two explanations and they need different fixes.

1. The model cannot judge Tetris positions.
2. The model cannot work out which position a phrase like *"L turned once, left edge in column 4"* produces.

`readout` asks the second question and `readout_outcome` removes it -- the placement is applied and the resulting board
is rendered into the question itself, which fits because a branch may carry 512 tokens and a board is about 150. It is
still one context pass with every candidate as a row.

Showing the consequence **did not help**: 20.5 pieces against 26.0, if anything slightly worse. So explanation 1 is the
one left standing, and it is not about prompt wording.

## The game is a blunt instrument, so the ranking was measured directly

A game ends when one bad placement buries a column, so its outcome is a few decisions amplified by many. `--diagnose`
looks at every decision instead: on boards the heuristic itself reached, it compares the model's ranking of the
placements with the heuristic's, by rank correlation and by whether the model's own pick is among the heuristic's best
three.

| framing | mean rank correlation over 12 boards | model's pick in the reference's best three |
|---|---|---|
| placements described | +0.096 +/- 0.070 | 5 / 12, against 1.99 expected by chance |
| resulting board shown | -0.105 +/- 0.083 | 3 / 12, against 1.99 expected by chance |

Neither correlation is distinguishable from zero (t = +1.36 and -1.27). Individual boards run from -0.484 to +0.467,
which is noise around nothing.

One number in that table is not nothing: the described framing's top pick landed in the heuristic's best three five
times out of twelve where chance gives two, and that has an exact tail probability of 0.031. It is reported because it
is there, and it should not be built on: it is one of four comparisons taken from the same twelve boards, which is
enough multiplicity to explain it, and the thing it would have to predict -- the game -- is at the floor. If it is real
it is a trace, not a signal.

## What this example is for

It is a genuine use case measured honestly, and the honest answer is that this model cannot play Tetris by scoring
placements, at either framing, whatever extraction is used.

That leaves the mechanism's part intact and unflattered. One board, 34 placements, one forward pass, 556 ms: the fork
did exactly what it claims, and 34 separate calls would have cost far more. A mechanism that delivers a decision
cheaply cannot make the decision good, and a task where a decision compounds is where that distinction is most
expensive to get wrong -- which is the reason to keep this example rather than delete it.

Two things would make a real attempt, and neither is here:

* **give the model the features instead of the picture.** Holes, heights and bumpiness of each resulting board, as
  numbers, and ask which is best. That tests arithmetic rather than spatial reading, and if it works the interesting
  measurement becomes how few features it needs.
* **fit something on top of the read-out.** The per-question threshold machinery in `prismyra/thresholds.py` exists
  because a raw probability is not a decision; a ranking learned from played games is the same idea one step further,
  and `--diagnose` is already the instrument that would say whether it had learned anything.

## Files

* `game.py` -- the board, the seven pieces, legal placements, clears. No rendering loop and no scoring curve.
* `agents.py` -- `Random`, `Heuristic`, `ReadOut`, `ReadOutOutcome`, `Generate`, and the `agreement` diagnostic.
* `play.py` -- runs games or the diagnostic and prints outcomes.
* `../../tests/test_tetris.py` -- the rules, checked without a model, because a comparison built on wrong rules is
  measured precisely and means nothing. Includes the one simplification everything rests on: a piece cannot slide
  under an overhang.
