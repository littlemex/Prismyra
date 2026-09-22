"""What an option's score is worth before the context is read, so it can be subtracted.

Some option tokens win before the question is asked. `yes` is a commoner continuation than `no`, `A` a commoner one
than `D`, and a long option name's first token is rarer than a short one's. None of that is about the context, and all
of it lands in the score. Subtracting it is the cheapest improvement available above the raw read-out: it needs no
labels, no training, and the answer stays the model's own -- only the thumb is taken off the scale.

The prior is measured rather than assumed. The same question is asked against a **content-free context** -- text of
the same shape carrying no information about anything -- and whatever the model prefers there is what it prefers for
reasons other than the context. That is the number subtracted.

Cost, and why it is nearly free in the case this package is for: a prior depends on the question, not on the document.
Ask the same eight questions of a thousand contracts and the priors are computed once. They are cached per question
text and option set, so the second document onwards pays nothing.

Whether this helps is an empirical question and `evals/run.py` answers it. It is not on by default until the numbers
say it should be.
"""

from __future__ import annotations

import torch

#: The content-free context. Not an empty string: an empty context is a different shape of input, and the model's
#: behaviour on it is not the behaviour being corrected for. This says nothing while looking like something.
NULL_CONTEXT = "N/A"

#: Several, because one is a sample of size one. The prior is the mean over these, which keeps a quirk of any single
#: placeholder from becoming the correction.
NULL_CONTEXTS = ("N/A", "None.", "The text is not available.")


#: Two things can be subtracted, and they are not the same thing. `null_context` asks the whole question against a
#: content-free context. What comes back includes the option tokens' own frequency, but also everything the question
#: and its option wording say on their own -- world knowledge, a giveaway in how an option is phrased, a dataset's
#: habits. Subtracting it removes all of that, which is pointwise mutual information between the context and the
#: answer rather than a bias correction, and it removes signal a caller may well want. `options_only` drops the
#: question too and leaves the option list and the place the answer goes. What comes back is close to the option
#: tokens' preference alone, which is the thing that is unambiguously not about the context.
MODES = ("null_context", "options_only")


class Calibration:
    """Priors for questions, computed on demand and kept.

    Holds no model. It is handed the engine when it needs to measure something, so an engine and its calibration can
    be reasoned about separately and a cache can outlive a request.
    """

    def __init__(self, mode: str = "options_only", contexts: tuple[str, ...] = NULL_CONTEXTS):
        if mode not in MODES:
            raise ValueError(f"unknown calibration mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.contexts = contexts
        self._cache: dict[tuple, torch.Tensor] = {}

    def priors(self, engine, questions: list, plans: list) -> list[torch.Tensor]:
        """One vector per question: what each option scored with nothing to go on.

        Keyed by the question's rendered text and its option tokens rather than by its id, because an id is the
        caller's label and two callers may use the same one for different questions. The text and the tokens are what
        the number actually depends on.
        """
        # Deduplicated before measuring. The same question twice in one call would otherwise have its prior added
        # twice and divided only by the number of null contexts, leaving a correction scaled by however many copies
        # arrived.
        missing: dict[tuple, object] = {}
        for plan in plans:
            key = self._key(plan)
            if key not in self._cache and key not in missing:
                missing[key] = plan
        if missing:
            self._measure(engine, list(missing.values()))
        return [self._cache[self._key(plan)] for plan in plans]

    def _key(self, plan) -> tuple:
        return (self.mode, self._text(plan), tuple(plan.token_ids))

    def _text(self, plan) -> str:
        """What is asked when the prior is measured.

        In `options_only` the question is dropped and only the option list and the answer marker are kept, so what is
        measured is the option tokens' own preference. The lines are the rendering's own, so this stays in step with
        `schema.render_question` rather than reconstructing its format.
        """
        if self.mode == "null_context":
            return plan.text
        lines = [line for line in plan.text.split("\n") if not line.startswith("Question:")]
        # Everything between the question and the answer marker belongs to the question, except the option list.
        kept = [line for line in lines if line.startswith("Answer")]
        return "\n".join(kept) if kept else plan.text

    def _measure(self, engine, plans: list) -> None:
        """Ask every uncached question against each content-free context and average what comes back.

        Averaged before any softmax, because that is where the correction is applied. Averaging probabilities instead
        would mix normalisations taken over different scales.
        """
        totals: dict[tuple, torch.Tensor] = {}
        for context in self.contexts:
            for plan, row in zip(plans, self._score_all(engine, plans, context), strict=True):
                key = self._key(plan)
                totals[key] = row if key not in totals else totals[key] + row
        for key, total in totals.items():
            self._cache[key] = total / len(self.contexts)

    def _as_asked(self, plan):
        """The plan as it will be asked when measuring, which in `options_only` carries a shortened text."""
        from .readout import Plan

        text = self._text(plan)
        if text == plan.text:
            return plan
        return Plan(text=text, token_ids=plan.token_ids, trailing_space=plan.trailing_space)

    def _score_all(self, engine, plans: list, context: str) -> list[torch.Tensor]:
        """The raw option scores for these questions against one content-free context.

        Uses the engine's own context pass and branch pass, so the prior is measured through the same code path the
        real answer comes through. A prior measured any other way would correct for something else.
        """
        from .readout import logits_for

        with engine._lock, torch.inference_mode():
            opened = engine.open_context(context)
            try:
                prefill = opened._prefill
                assert prefill is not None
                width = engine._width_for([self._as_asked(p) for p in plans])
                out: list[torch.Tensor] = []
                for lo in range(0, len(plans), engine.group):
                    chunk = plans[lo : lo + engine.group]
                    hidden = engine._branch(prefill, [self._text(p) for p in chunk], engine.group, width)
                    out.extend(
                        row.detach().clone()
                        for row in logits_for(hidden, engine.unembedding, [p.token_ids for p in chunk])
                    )
            finally:
                opened.close()
        return out
