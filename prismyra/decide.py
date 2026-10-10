"""A JEV-compatible read-out: ``POST /v1/decide`` takes the request shape autotrust's JEV-27B(-VL) serves
through its own ``serve_decide.py`` -- ``{kind, state, question, options}`` -- and answers it through
Prismyra's own ``ask()`` rather than JEV's lm_head-LoRA decision head. See ``RUN-jev.md`` (internal, not
shipped) for the real-hardware comparison this endpoint is a response to; what follows is the contract,
not the history.

Three closed vocabularies, matching JEV's own:

* ``noul`` -- yes/no. JEV's verbalizer pair is ``["false", "true"]``; Prismyra's native `Boolean` reads
  ``["no", "yes"]`` instead (every other test in this project already depends on that pair existing),
  so this module reuses `Boolean` as-is and relabels the two probabilities on the way out. The token
  that is actually scored on the device is Prismyra's own, not JEV's -- the two label strings carry the
  same meaning, not the same bytes.
* ``score`` -- an integer 0-5. Prismyra's `Scale(low=0, high=5)` already renders exactly JEV's six
  verbalizers (``"0"`` through ``"5"``), so this is a direct pass-through with no relabelling at all.
* ``choice`` -- 2 to 256 named options, by JEV's own ceiling (``serve_decide.py``'s ``MAX_OPTIONS = 256``,
  reached by continuing its trained A-P labels with Q-Z and then two-letter labels). Prismyra's own
  ceiling is lower: `schema.MAX_OPTIONS` is 255, a property of one `Question` carrying at most that many
  options, independent of this endpoint. A `choice` item asking for 256 options is refused with that
  number, not silently truncated to 255 -- see `DecideError` below.

**Why every `choice` question is relabelled, not only the ones that need it.** A `choice` option's own
text is not, in general, one token that shares no first token with any other declared option -- the one
constraint `Prismyra.validate` enforces (`schema.py`'s account of the two tokenizer-dependent refusals,
checked in `prismyra.readout.plan`). Trying the caller's own wording first and relabelling only on
refusal would work, but it buys an inconsistency for nothing: the probe that decides whether relabelling
was needed is itself a call into `plan()`, so skipping it when the wording happens to pass saves one
cheap CPU call and nothing else, while leaving two code paths -- "scored on your own words" and "scored
on a label" -- that read back differently under `heads.py`-style debugging and that this file would have
to keep in step with each other forever. One rule reads every `choice` question the same way JEV's own
prompt already does: list ``A) option0``, ``B) option1``, ... in the question text, and score the single
label token. `choice_labels` below is what finds, once per tokenizer, how many such labels this model's
vocabulary can support.

**What a probability means here is Prismyra's own answer, not a reproduction of JEV's one.** See
`schema.SCORING`. JEV applies a trained bias and a per-kind temperature before its softmax
(`decision_head.json`'s own note: "decision logits = lm_head-LoRA logprobs of verbalizer ids + bias, then
per-kind temperature"). Prismyra applies the analogous correction only when the engine was built with
`calibrate=True` (`prismyra.calibration.Calibration`), and returns the plain softmax over the declared
options otherwise -- nothing here invents a bias or a temperature to make the two numbers mean the same
thing, because that would be answering a question nobody measured.

**Why a batch of decisions about the same state costs one `ask()`, not several.** JEV's own protocol is
one call in, one decision out (`DecideRequest` has no field for a second question). A caller that wants
several decisions about the same `state` -- several `kind="choice"` fields on one record, say -- pays for
that state's context exactly as many times as it asks about it, because each call is its own forward
pass. Prismyra's `ask(context, questions)` was built for exactly this shape: one context, many typed
questions, one pass over the context and one shared branch read. `decide_many` below groups a request's
items by their own (normalised) `state` and answers every group in one `ask()` call -- the same context
read once, however many `kind`s of decision were asked about it. Each item in the response carries
`merged_with`, the other items answered alongside it, so a caller can see the saving rather than take it
on faith.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import Boolean, Choice, PrismyraError, Question, QuestionError, Scale

#: JEV's own ceiling (`serve_decide.py`'s `MAX_OPTIONS`). Prismyra's own, lower ceiling
#: (`schema.MAX_OPTIONS`, 255) is what a `choice` item is actually checked against -- this constant is
#: kept only to phrase the refusal in JEV's own terms ("JEV allows up to 256; this build answers up to
#: N") rather than silently rejecting the overlap as the same thing.
JEV_MAX_OPTIONS = 256

NOUL_OPTIONS = ("false", "true")
SCORE_OPTIONS = tuple(str(n) for n in range(6))

#: One protocol id, so a caller logging it alongside JEV's `"jev27-bare-v1"` can tell which server
#: answered. Not a claim of wire-level interchangeability -- the two protocols' response shapes differ in
#: ways documented below (`options`/`probabilities` are keyed/ordered differently, and `value` has no JEV
#: equivalent) -- only an identifier.
PROTOCOL = "prismyra-decide-v1"

#: Candidate single-token labels, generated the same way JEV's own `setup()` does: the alphabet, then
#: every two-letter combination, in that order (not sorted -- see `choice_labels`'s own note on why the
#: order here does not need to match the order a rendered prompt lists its options in).
import string as _string  # noqa: E402 - kept local to the constant it builds, not a module-wide import

_LABEL_CANDIDATES = list(_string.ascii_uppercase) + [
    a + b for a in _string.ascii_uppercase for b in _string.ascii_uppercase
]


class DecideError(PrismyraError):
    """A `/v1/decide` item cannot be answered as written -- a bad `kind`, a `choice` outside this build's
    supported option count, an empty `state`, or a `state` shaped as JEV's multimodal parts list (not yet
    supported by this endpoint). Raised before anything reaches the device, the same discipline
    `QuestionError` already keeps for `/ask`.
    """


@dataclass(frozen=True)
class DecideItem:
    """One caller-supplied decision, parsed and validated, with everything needed to answer it and to
    translate the answer back into JEV's own vocabulary.
    """

    external_id: str
    kind: str
    context: str
    question: Question
    #: The options in the order the response reports them: JEV's own fixed order for `noul`/`score`, and
    #: the caller's own order (not Prismyra's internally-sorted rendering order) for `choice`.
    reported_options: tuple[str, ...]
    #: `choice` only: option text -> the single-token label `question` actually scores for it. `None` for
    #: `noul`/`score`, which score Prismyra's own native tokens directly and need no such map.
    symbols: dict[str, str] | None = None


def choice_labels(tokenizer, limit: int = 255) -> list[str]:
    """How many single-token labels this tokenizer can support at once, found the same way JEV's own
    `setup()` finds its A-Z/AA-ZZ pool, but via Prismyra's own check (`readout.plan`) instead of a second,
    hand-rolled tokenisation rule that could silently drift from the one `Choice` is actually scored
    with.

    A label set of size *n* is usable exactly when a `Choice` declaring those *n* strings as its options
    can be planned at all -- `plan()` already refuses a multi-token option and two options sharing a
    first token, which is precisely what a label pool must avoid internally to be usable together. Tried
    at the full candidate count first (cheap when it simply works, which it does on every tokenizer this
    project targets: 702 candidates, Prismyra's own ceiling is 255), falling back to a binary search only
    if that first try fails.
    """
    from .readout import plan

    limit = min(limit, len(_LABEL_CANDIDATES))

    def usable(n: int) -> bool:
        probe = Choice(id="_prismyra_decide_label_probe", prompt="probe", choices=_LABEL_CANDIDATES[:n])
        try:
            plan(probe, tokenizer)
        except QuestionError:
            return False
        return True

    if usable(limit):
        return _LABEL_CANDIDATES[:limit]
    lo, hi, best = 2, limit, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if usable(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return _LABEL_CANDIDATES[:best]


def _state_to_context(state) -> str:
    """JEV's own `state` is a string, a JSON object, or a list mixing text and image parts
    (`serve_decide.py`'s `_parts`). Prismyra's context is text only: a string is used as-is, an object is
    rendered the same way JEV's own `_parts` renders one (`json.dumps(state, ensure_ascii=False)`, so the
    two servers read the same bytes for the same dict-shaped `state`), and a list is refused -- it exists
    in JEV's protocol to carry images, which neither of this project's two served models reads, and
    pretending to accept the shape while dropping the images it is for would be worse than refusing it by
    name.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False)
    raise DecideError(
        "state was a list, which in JEV's own protocol carries image or video parts; this endpoint reads "
        "text only (prismyra.schema.Request requires a text context) -- send a string or a JSON object"
    )


def parse_item(raw: dict, index: int, labels: list[str]) -> DecideItem:
    """One wire item -> one `DecideItem`, or `DecideError`/`QuestionError` before anything is queued.

    `index` names the item in error messages and is the external id's fallback (`str(index)`, the same
    "fill the id from the position" rule `server.build_questions` already uses for plain `/ask`), so a
    caller that sent a bare list of items without `id`s still gets a response it can index back against
    its own request.
    """
    kind = raw.get("kind")
    if kind not in ("noul", "score", "choice"):
        raise DecideError(f"item {index}: kind must be one of noul, score, choice -- got {kind!r}")
    question_text = raw.get("question")
    if not question_text or not str(question_text).strip():
        raise DecideError(f"item {index}: a decision needs a non-empty question")
    external_id = str(raw.get("id") or index)
    context = _state_to_context(raw.get("state", ""))
    if not context.strip():
        raise DecideError(
            f"item {index}: state is empty. JEV's own protocol allows an empty state (a decision from the "
            f"question alone); Prismyra's engine does not -- `ask()` requires a non-empty context, because "
            f"reading one context well is this library's whole design (see the README's scope note). Send "
            f"the document, record or transcript the decision is about as state."
        )
    internal_id = f"item{index}"

    if kind == "noul":
        return DecideItem(
            external_id=external_id,
            kind=kind,
            context=context,
            question=Boolean(id=internal_id, prompt=str(question_text)),
            reported_options=NOUL_OPTIONS,
        )
    if kind == "score":
        return DecideItem(
            external_id=external_id,
            kind=kind,
            context=context,
            question=Scale(id=internal_id, prompt=str(question_text), low=0, high=5),
            reported_options=SCORE_OPTIONS,
        )
    # kind == "choice"
    options = raw.get("options") or []
    if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
        raise DecideError(f"item {index}: choice needs options as a list of strings")
    n = len(options)
    if n > JEV_MAX_OPTIONS:
        raise DecideError(f"item {index}: choice needs 2-{JEV_MAX_OPTIONS} options (JEV's own ceiling), got {n}")
    if n > len(labels):
        raise DecideError(
            f"item {index}: choice needs 2-{len(labels)} options on this build -- this tokenizer supports "
            f"{len(labels)} simultaneous single-token labels (schema.MAX_OPTIONS caps it at 255; JEV's own "
            f"ceiling is {JEV_MAX_OPTIONS}), got {n}"
        )
    if n < 2:
        raise DecideError(f"item {index}: choice needs at least 2 options, got {n}")
    if len(set(options)) != len(options):
        raise DecideError(f"item {index}: choice repeats an option: {options}")
    symbols = dict(zip(options, labels[:n], strict=True))
    lines = "\n".join(f"{symbols[opt]}) {opt}" for opt in options)
    prompt = f"{question_text}\n[options]\n{lines}"
    return DecideItem(
        external_id=external_id,
        kind=kind,
        context=context,
        question=Choice(id=internal_id, prompt=prompt, choices=tuple(symbols.values())),
        reported_options=tuple(options),
        symbols=symbols,
    )


def _item_response(item: DecideItem, answer, model: str, merged_with: list[str]) -> dict:
    """One item's answer, in JEV's own vocabulary (`options`/`probabilities`/`choice_index`/`choice`),
    plus two fields JEV's protocol has no equivalent for: `value` (Prismyra's typed read: `bool`, `int` or
    the option's own text) and `merged_with` (which other items in this request were answered in the same
    `ask()` pass as this one).
    """
    value: bool | int | str
    if item.kind == "noul":
        probs = [answer.probabilities["no"], answer.probabilities["yes"]]
        chosen_index = 1 if answer.value else 0
        chosen = NOUL_OPTIONS[chosen_index]
        value = bool(answer.value)
    elif item.kind == "score":
        probs = [answer.probabilities[o] for o in SCORE_OPTIONS]
        chosen_index = SCORE_OPTIONS.index(answer.option)
        chosen = answer.option
        value = int(answer.value)
    else:  # choice
        symbols = item.symbols or {}
        probs = [answer.probabilities[symbols[opt]] for opt in item.reported_options]
        chosen_index = probs.index(max(probs))
        chosen = item.reported_options[chosen_index]
        value = chosen  # the caller's own option text; `answer.value` is the symbol `Choice.value_of` saw, not this
    return {
        "id": item.external_id,
        "kind": item.kind,
        "options": list(item.reported_options),
        "probabilities": probs,
        "choice_index": chosen_index,
        "choice": chosen,
        "value": value,
        "adaptation": "native" if item.kind != "choice" else "symbol-labelled",
        "symbols": dict(item.symbols) if item.symbols else None,
        "protocol": PROTOCOL,
        "model": model,
        "merged_with": merged_with,
    }


def decide_many(engine, raw_items: list[dict], ask) -> tuple[list[dict], int]:
    """Parse, group by state, and answer every group in one call to `ask`.

    `ask` is `(context, questions) -> Result` -- the caller's own queue entry point (`Worker.submit` in
    `server.py`, so this never touches the device from the HTTP handler's own thread), not `engine.ask`
    directly. Validation happens before any group is submitted: one bad item refuses the whole request,
    the same all-or-nothing admission `/ask` already gives a malformed question, rather than this
    endpoint inventing a partial-success shape nothing else here has.

    Returns the responses in the caller's own order and the number of `ask` calls actually made --
    equal to the number of distinct states, which is what a caller measuring "did batching several
    decisions about one record actually cost one pass" should compare against the number of items.
    """
    # Computed only when a `choice` item is actually present: `choice_labels` probes the tokenizer, and a
    # request with only `noul`/`score` items (which score Prismyra's own fixed, already-known-safe verbalizers)
    # has no use for it.
    labels = choice_labels(engine.tokenizer) if any(raw.get("kind") == "choice" for raw in raw_items) else []
    items = [parse_item(raw, i, labels) for i, raw in enumerate(raw_items)]

    groups: dict[str, list[DecideItem]] = {}
    order: list[str] = []
    for item in items:
        if item.context not in groups:
            groups[item.context] = []
            order.append(item.context)
        groups[item.context].append(item)

    answers_by_internal_id: dict[str, object] = {}
    for context in order:
        group = groups[context]
        result = ask(context, [i.question for i in group])
        for i in group:
            answers_by_internal_id[i.question.id] = result[i.question.id]

    responses = []
    for item in items:
        group = groups[item.context]
        merged_with = [other.external_id for other in group if other is not item]
        responses.append(
            _item_response(item, answers_by_internal_id[item.question.id], engine.model_name, merged_with)
        )
    return responses, len(order)
