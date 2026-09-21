# What a probability means

The read-out is the product, so its definition is stated once here and versioned as
`prismyra.SCORING_VERSION`. Store that value beside any answer you keep: if the definition changes, the version does.

## The definition

For each option a question declares, the model's own output embedding scores the **single token that spells that option**,
at that branch's final position. Those scores are softmaxed **over the declared options only**.

Three different things could be called "the probability of an option" and they do not agree:

| | what it would mean | used here? |
|---|---|---|
| first-token logit, softmaxed over the whole vocabulary | how likely the model is to say this word next, among all words | no |
| sequence likelihood of the option's full text | how likely the whole phrase is | no |
| **softmax over the declared options** | which of *your* options the model prefers | **yes** |

So a probability here is a **relative preference among the options you offered**. Consequences worth stating plainly:

- It is **not calibrated**. A 0.91 does not mean the answer is right nine times in ten. There is deliberately no field
  named `confidence`, because naming it that invites reading it as calibration.
- It is **not comparable across different option sets**. Adding a third option changes the other two.
- It sums to one by construction, which says nothing about whether the answer is in your option set at all.

Nothing is trained. The read-out is the model's own output embedding used as a reader instead of a writer, so answers are
zero-shot: there is no head to fit and nothing to fine-tune.

## Two things refused rather than answered wrongly

Both raise at construction, never at inference:

**An option that is more than one token.** The scoring reads one token, so a longer name would be scored on its first
piece and the rest ignored -- answerable and wrong. The check tokenises the option **with a leading space**, because that
is how it appears after `Answer:`; a name that is one token bare and two with a space in front would otherwise slip
through and be scored on the wrong token.

**Two options whose first token is the same.** They cannot be told apart, so they would silently tie.

Practically: prefer short option names. `seller` and `buyer` work; `the seller` and `the buyer` collide on `the`.

## Why the option list is in the prompt, and sorted

The rendered question lists the options, in a canonical sorted order. Both halves are load-bearing and were measured:

- listing them in the **caller's** order made answers depend on that order, by up to 0.165;
- **omitting** the list put three-way questions at chance.

So they are listed, and sorted, always. Two callers who declare the same options in different orders get the same answer.

## What this does not protect against

A context is untrusted input. A typed output constrains the *shape* of what comes back -- it will be one of your options
-- but it does not stop a context from arguing for the wrong one. Treat a context the way you would treat any text a
user supplies; the type system here is not a security boundary.
