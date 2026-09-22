"""The comparison that matters: the same model, generating the answer instead of scoring one token.

Deliberately not a second serving engine. A second process cannot hold a second copy of these weights on one card, and
more importantly a different engine would change two things at once -- the read-out and the implementation -- leaving
no way to say which moved the accuracy. This uses the backbone Prismyra already loaded and the output embedding it
already holds, so the only difference from Prismyra's read-out is what happens at the final position: greedily emit
tokens and read the text, rather than score the declared options.

That makes the **accuracy** comparison clean. The latency comparison is weaker and in the opposite direction to the
one first written here: this is a plain decode loop with no batching, no paged cache and no prefix sharing, so a speed
advantage measured against it is an **upper bound** on the advantage over generation, not a lower one. A serving
engine can batch the questions and share the article's prefix without using this read-out at all. Read every speed-up
here as "against an unoptimised implementation", because that is what it is.
"""

from __future__ import annotations

import torch

from prismyra.media import encode
from prismyra.schema import render_question

#: Enough to clear a reasoning preamble and then say the answer. Measured rather than guessed: this model opens with a
#: `<think>` block even when it has nothing to think about, and eight tokens cut a third of its answers off mid-
#: preamble, which scored generation at 0% and made the read-out look better than it is. A budget that truncates the
#: baseline is a broken baseline, not a fast one.
MAX_NEW_TOKENS = 24

#: When the model is allowed to reason. Large because it uses it: at twenty-four tokens half its answers are still
#: inside the block. This is what generation costs on a reasoning model, and the cost is the finding.
THINKING_TOKENS = 512

#: Emptied and closed immediately on this model, but it still occupies the position the answer would otherwise be at.
THINK = ("<think>", "</think>")


#: Written into the prompt to close the reasoning block before the model opens it, which is how these models are run
#: when an answer is wanted rather than an argument. Without it this comparison is not the one being made: the read-
#: out scores one position and does no reasoning at all, so the matching baseline is generation that does none either.
NO_REASONING = "<think>\n\n</think>\n\n"


def answer_by_generating(
    engine, context: str, questions: list, reasoning: bool = False, budget: int | None = None
) -> tuple[dict[str, str], float, int]:
    """Greedily generate an answer to each question, reading the context once per question.

    Once per question, not once per context, because that is what generating means here: the branch trick is the thing
    being compared against, so borrowing it would compare Prismyra with itself.

    With `reasoning`, the model is left to think first and given a budget to do it in. That is a different and much
    more expensive method than the read-out, and worth measuring separately rather than instead: half of this model's
    answers do not fit in twenty-four tokens because it is genuinely reasoning, and scoring those as unanswered
    understates generation badly. Both rows belong in the table.

    Returns the answers, the elapsed milliseconds, and how many answers ran to the end of the budget. That last number
    is the audit on this baseline: if it is not zero, an accuracy taken from this run is confounded by the budget
    rather than measured, and saying so is the only way a reader can tell.
    """
    import time

    out: dict[str, str] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize(engine.torch_device)
    started = time.perf_counter()

    hit_budget = 0
    for question in questions:
        prompt = f"{context}\n\n{render_question(question)}"
        if not reasoning:
            prompt += NO_REASONING
        text, truncated = _greedy(engine, prompt, budget or (THINKING_TOKENS if reasoning else MAX_NEW_TOKENS))
        out[question.id] = text
        hit_budget += int(truncated)

    if torch.cuda.is_available():
        torch.cuda.synchronize(engine.torch_device)
    return out, (time.perf_counter() - started) * 1e3, hit_budget


def _greedy(engine, prompt: str, budget: int) -> str:
    """One prompt, a few tokens, argmax each step.

    The cache is the model's own rather than one constructed here. A bare dynamic cache has no layers, and this
    architecture asks its cache which kind each layer is -- so building one by hand fails on the first recurrent layer
    with an index error. Letting the first forward create it is also what a caller who had never heard of this package
    would get, which is the point of the comparison.
    """
    encoded = encode(prompt, None, None, engine.processor, engine.tokenizer, engine.device)
    ids = encoded.input_ids
    cache = None
    produced: list[int] = []
    # The output projection in its stored dtype, once. Casting it per token converts the largest matrix in the model
    # on every step, which inflated this baseline's latency by more than the thing being measured. The small side is
    # cast instead, so the arithmetic is unchanged and the copy is gone.
    head = engine.unembedding

    with torch.inference_mode():
        for _ in range(budget):
            out = engine.backbone(input_ids=ids, use_cache=True, past_key_values=cache)
            cache = out.past_key_values if hasattr(out, "past_key_values") else None
            if cache is None:
                raise RuntimeError("the backbone returned no cache, so generation would re-read the prompt each step")
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            logits = hidden[0, -1].to(head.dtype) @ head.t()
            token = int(logits.argmax().item())
            if token in _stop_tokens(engine.tokenizer):
                break
            produced.append(token)
            ids = torch.tensor([[token]], device=engine.device)

    return engine.tokenizer.decode(produced, skip_special_tokens=True).strip(), len(produced) >= budget


#: Worked out once per tokenizer and kept, because it costs a tokenise per lookup otherwise and this sits in a decode
#: loop. A dict rather than a rebound module variable, so nothing is reassigned at import scope.
_STOP: dict[int, set[int]] = {}


def _stop_tokens(tokenizer) -> set[int]:
    key = id(tokenizer)
    if key not in _STOP:
        # Deliberately not the newline. A reasoning block opens with newlines, so stopping at the first one stops
        # before the answer -- which is how this baseline first scored zero.
        candidates = {tokenizer.eos_token_id, tokenizer.pad_token_id}
        ids = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            candidates.add(ids[0])
        _STOP[key] = {token for token in candidates if token is not None}
    return _STOP[key]


def strip_reasoning(text: str) -> str:
    """Drop a reasoning block so the answer after it can be read.

    Not cosmetic. This model opens every answer with `<think>`, so a parser that reads from the first character finds
    a tag rather than an answer and scores a correct answer wrong. Anything after the closing tag is the answer; an
    unclosed block means the budget ran out before the model got to one, which stays unparseable.
    """
    opened, closed = THINK
    if closed in text:
        return text.split(closed, 1)[1]
    if opened in text:
        return ""
    return text


def parse(text: str, question, aliases: dict[str, list[str]] | None = None) -> tuple[object | None, str]:  # noqa: PLR0911
    """Read a generated answer as one of the declared options. Returns the value and how it had to be found.

    Tiers that widen, with which one was used returned rather than hidden. Being strict is not the same as being fair:
    refusing "the answer is B" would penalise generation for writing a sentence, which is not what is being compared.
    Being loose is not fair either, so each widening is counted separately and can be audited.

    **A one-character option is matched differently, and that is what most of this function is for.** A prefix match
    on a single letter reads "because the passage says so" as B, "a few people" as A and "definitely B" as D. All
    three were measured on an earlier version of this function, silently, and all three went against generation. So a
    one-letter option has to be followed by punctuation or nothing at all.

    `aliases` lets an option be answered by another name. A multiple-choice task rendered as letters should accept the
    option's own text as well, because answering "Over 2,000 people" rather than "A" is an answer, and scoring it
    absent is this function failing rather than the model.
    """
    import re

    cleaned = strip_reasoning(text).strip().lower().lstrip("*:. ").strip()
    if not cleaned:
        return None, "empty"

    names: dict[str, str] = {}
    for option in question.options:
        names[option.lower()] = option
        for alias in (aliases or {}).get(option, []):
            names[alias.strip().lower().rstrip(".")] = option

    if cleaned in names:
        return question.value_of(names[cleaned]), "exact"

    # Longest first, so a longer name is not shadowed by a shorter one that prefixes it.
    for name in sorted(names, key=len, reverse=True):
        if len(name) == 1:
            # Punctuation or the end of the answer, never a space. A letter followed by a space begins a word, and "a
            # few people" is not an answer of A -- this is the tier that was letting that through.
            if re.match(rf"{re.escape(name)}\s*(?:[.)\]:,;-]|$)", cleaned):
                return question.value_of(names[name]), "delimited"
        elif cleaned.startswith(name):
            return question.value_of(names[name]), "prefix"

    # The last resort searches the whole answer, and needs two passes rather than one. A single letter appears in
    # ordinary English as a word -- "a few people" -- so accepting one requires punctuation or the end of the answer
    # after it. But an answer naming two options is not an answer to a closed question, and only the loose pass
    # notices that. So: accept on the strict pass, refuse whenever the loose pass sees more than one.
    def matches(pattern: str) -> set:
        return {names[name] for name in names if re.search(pattern.format(re.escape(name)), cleaned)}

    loose = matches(r"(?<![^\W_]){0}(?![^\W_])")
    strict = matches(r"(?<![^\W_]){0}\s*(?:[.)\]:,;-]|$)")
    if len(loose) > 1:
        return None, "ambiguous"
    if len(strict) == 1:
        return question.value_of(strict.pop()), "mentioned"
    return None, "absent"
