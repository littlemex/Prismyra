"""How one request becomes a tag, or does not. DISTILL-RL-DESIGN-v2.md section 1:

    要求は、task を名指しするか、問いの文面と選択肢が spec と完全に一致したときだけ、そのタグになる。
    どちらでもない要求は、タグなし(学習しない)。

A request is tagged in exactly one of two ways, and anything else is untagged -- untagged means unseen: no log
entry, no feature, no student ever sees it.

1. It names a registered ``task`` directly. This is still required to agree with the entry it names on ``kind``
   and on the option set: naming a task is a shortcut for finding the entry, not a license to log a different
   decision under that entry's name. A request that names a task that is not registered, or that is registered
   but the shapes do not agree, is untagged -- it is not an error, because this path exists for callers who do
   not know they are inside an experiment at all (``/ask`` has no ``task`` field; only ``/v1/decide`` does, and
   only because this project added it there for exactly this purpose).
2. It does not name a task, and its ``(kind, question, options)`` matches exactly one spec entry once the
   question text has been run through that entry's own ``normalize`` rule and the option sets are compared as
   sets, not sequences (section 1: "文書 (context) はタグに入れない。違う文書に対する同じ判断を集める" -- the
   match is on the shape of the question, never on the document it is being asked about).

Nothing here reads ``context``. A tag names a kind of judgment, not an instance of one, by the same design choice
``prismyra.heads`` already made for its own matching rule (see that module's docstring).
"""

from __future__ import annotations

from collections.abc import Sequence

from .spec import LearnSpec, TagSpec


def normalize(text: str, rule: str) -> str:
    """The four rules ``spec.NORMALIZE_RULES`` allows. Reachable only with a rule a loaded spec has already
    validated, so the ``ValueError`` below is a programming error inside this package, never a request-time
    outcome.
    """
    if rule == "none":
        return text
    if rule == "strip":
        return text.strip()
    if rule == "lower":
        return text.lower()
    if rule == "strip+lower":
        return text.strip().lower()
    raise ValueError(f"unknown normalize rule {rule!r}")


def tag_for(
    spec: LearnSpec,
    *,
    task: str | None,
    question: str,
    options: Sequence[str],
    kind: str,
) -> TagSpec | None:
    """The one comparison the design's section 8 budgets for every request once a spec is loaded.

    Returns the matching ``TagSpec``, or ``None`` when the request is untagged. Never raises: an untagged request
    is the common case (most traffic has nothing to do with any registered tag), not a condition worth an
    exception's cost or its control flow.
    """
    want_options = frozenset(options)
    if task:
        found = spec.by_task(task)
        if found is None:
            return None
        if found.kind != kind or frozenset(found.options) != want_options:
            # Named the right tag, but the shape does not match it: logging this under that tag's name would mix
            # a different judgment into its experience, exactly what section 1's last line forbids. Untagged, not
            # an error -- the caller may simply be answering a different question with the same id by accident.
            return None
        return found

    for entry in spec.tags:
        if entry.kind != kind:
            continue
        if frozenset(entry.options) != want_options:
            continue
        if normalize(question, entry.normalize) == normalize(entry.question, entry.normalize):
            return entry
    return None
