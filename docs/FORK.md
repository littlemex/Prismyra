# The fork

Asking Q questions about one context the obvious way reads the context Q times. Prismyra reads it once. The saving is
the whole library, and it rests on a single asymmetry.

## What is shared and what is not

After reading a context, the model holds two kinds of state, and they behave differently:

| | written by a branch? | consequence |
|---|---|---|
| the context's keys and values | no, only read | one write can serve every branch |
| the recurrence and convolution state | yes, each advances its own | must be **copied** per branch |
| a branch's own tokens | yes | private by construction |

`prismyra/cache.py` is built on this: `ForkLayer` takes a one-row write and lands it in every row, then returns only the
rows the caller brought. Writing broadly and reading narrowly is what removes the separate copy step, and each
row then owns storage it can write without aliasing its neighbours.

**The context is physically replicated, not shared.** Being read-only is what makes one write correct for every row; it
is not what makes it cheap. Each of the `group` rows holds its own copy of the context's keys and values, so an open
context costs `group` times the context's key-value cache -- about 3.4 GiB at 5,000 tokens with the default group of 32
on the supported model. Storing the context once and giving each branch only its own tail is the obvious improvement and
is not done here. [PERFORMANCE.md](PERFORMANCE.md) has the formula.

The primitive underneath is ordinary: selecting along the batch dimension, which is what beam search does to reorder
candidates. A constant index gives the fork; a non-constant one gives several contexts in one batch.

## Why branches cannot see each other

Two mechanisms, and only the first is code:

1. Each branch writes its own state, so nothing it computes lands where another branch reads.
2. A branch's tokens come **after** the shared context, and attention looks backwards. A branch is structurally unable
   to reach its neighbours' tokens.

The second is why branches need no mask. Padding rows to a common width is safe for the same reason: a pad position may
attend to anything before it, and no pad position is ever read -- each row is read at its own last real token.

## Why the context cannot be padded

The asymmetry does not extend to the context. A token appended to a branch is masked out of attention and never read; a
token appended to the context passes through the recurrence, where there is nothing to mask it out of. Measured:
appending padding to the context moved answers by up to 1.08e-01, against 2.4e-02 for a batch-shape change alone.

Two alternatives were measured and rejected:

- **Reading the context in fixed-size pieces.** Each piece pays the traversal floor again: 2,048 tokens in pieces of 512
  cost 509 ms against 104 ms whole, and answers moved up to 5.4e-02.
- **Padding on the left instead of the right.** No safer -- 3.8e-02 against 4.7e-02 -- because the convolution's window
  reaches into the padding and the rotary positions shift.

What does work is packing: several contexts laid end to end with their boundaries passed as data, so every kernel knows
where each one stops. Nothing here uses it yet. `ask_many` answers requests one after another and shares nothing between
them, and the attention and convolution kernels already take the boundaries they would need, so this is the largest
piece of work the design has room for rather than a feature to describe as though it shipped.

## Images and video

Nothing above changes for them, and that is the point. A frame's embeddings are written into the context's keys and
values, where every branch reads them and none writes them -- the same asymmetry the text case rests on. One image
encoded once serves every question asked about it.

One thing does change. With media present the model uses a three-axis rotary scheme whose text axis stops counting
tokens: an image occupies one position per grid cell rather than one per token. The model works out the difference while
reading the context and records it, and a branch has to continue from there rather than from the token count. Getting
that wrong raises nothing -- every branch simply reads the context from the wrong place -- so the test for it asks which
way a block travels in a clip, and then asks the same of the same frames reversed.

## Groups

A traversal of the model costs about the same whether it carries one branch or thirty-two, because the cost is reading
the experts' weights rather than the branches' tokens. So questions are answered in groups, and the group size is the
unit of cost:

```
total ~= context + per_group x ceil(questions / group)
```

Questions beyond a group start another traversal. This is why the thirty-third question is expensive and the
thirty-second is nearly free, and why `Prismyra(group=...)` exists at all.

The group is set once, on the engine, and not per call. The cache is preallocated for exactly that many rows and a write
of any other row count is refused, which is the point of preallocating: a per-call group would have to reallocate, which
is allocator work on the request path and invalidates any reference already taken to the old buffers.

It is a memory dial as well. The cache holds `group` copies of the context, so halving the group halves the memory and
adds one traversal per group of questions.
