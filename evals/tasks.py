"""Public benchmarks with real labels, shaped into one context and the questions asked about it.

Real labels, on purpose. The comparison this package invites -- one traversal against one per question -- says nothing
unless the answers are right, and a wrong answer can be made arbitrarily fast. So every task here comes from a
published dataset with published labels, and every number the harness prints carries an accuracy beside it.

Three tasks, chosen for what they expose rather than for what they flatter:

* **boolq** -- one yes or no question per passage. The fork earns nothing here, and it is the honest case: below four
  questions a general serving engine is faster, so this measures accuracy alone.
* **race** -- an article with several questions, four options each. This is the shape the design is for, and the
  grouping by article is what makes it so.
* **unfair_tos** -- one contract clause and eight independent unfairness types, so eight questions about one
  context. The labels are sparse, which makes accuracy a poor summary and the per-type balance the thing to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prismyra import Boolean, Choice, Question

LETTERS = ("A", "B", "C", "D", "E")

#: The eight unfairness types in the LexGLUE unfair terms-of-service task, in the dataset's own label order.
UNFAIR_TYPES = (
    ("limitation_of_liability", "does this clause limit the provider's liability?"),
    ("unilateral_termination", "does this clause let the provider terminate the agreement at will?"),
    ("unilateral_change", "does this clause let the provider change the terms unilaterally?"),
    ("content_removal", "does this clause let the provider remove the user's content at will?"),
    ("contract_by_using", "does this clause bind the user merely by their using the service?"),
    ("choice_of_law", "does this clause fix which jurisdiction's law applies?"),
    ("jurisdiction", "does this clause fix which court disputes must be brought in?"),
    ("arbitration", "does this clause require arbitration instead of a court?"),
)


@dataclass
class Item:
    """One context and everything asked about it, with the right answers.

    `gold` is keyed by question id, as answers are, so scoring is a lookup rather than a zip -- a zip over two lists
    is how an off-by-one in one of them becomes a plausible accuracy.
    """

    context: str
    questions: list[Question]
    gold: dict[str, Any]
    task: str


def load(task: str, limit: int, split: str = "validation", seed: int = 0) -> list[Item]:
    """A seeded random sample, not the first N rows.

    The first N is a slice of whatever order the dataset happens to be in, and once a budget or a prompt has been
    chosen by looking at it, it has become a development set. A seed makes the sample nameable, so a result can be
    repeated on the same one or checked on a different one.
    """
    from datasets import load_dataset

    if task == "boolq":
        rows = _sample(load_dataset("google/boolq", split=split), limit, seed)
        return [_boolq(row) for row in rows]
    if task == "race":
        return _race(load_dataset("ehovy/race", "middle", split=split), limit, seed)
    if task == "unfair_tos":
        rows = _sample(load_dataset("coastalcph/lex_glue", "unfair_tos", split=split), limit, seed)
        return _unfair_tos(rows)
    raise ValueError(f"unknown task {task!r}; expected boolq, race or unfair_tos")


def _sample(rows, limit: int, seed: int):
    import random

    picked = list(range(len(rows)))
    random.Random(seed).shuffle(picked)
    return rows.select(sorted(picked[:limit]))


def _boolq(row) -> Item:
    return Item(
        context=row["passage"],
        questions=[Boolean(id="answer", prompt=row["question"].strip().rstrip("?") + "?")],
        gold={"answer": bool(row["answer"])},
        task="boolq",
    )


def _race(rows, limit: int, seed: int) -> list[Item]:
    """Grouped by article, because that is the shape this package is for.

    The dataset stores one row per question with the article repeated, so reading it row by row would ask one question
    per context and measure the case the design is worst at. Articles are sampled, not taken in order.
    """
    import random

    by_article: dict[str, list] = {}
    for row in rows:
        by_article.setdefault(row["article"], []).append(row)

    keys = list(by_article)
    random.Random(seed).shuffle(keys)

    items = []
    for article in keys[:limit]:
        group = by_article[article]
        questions, gold = [], {}
        for index, row in enumerate(group):
            options = row["options"]
            letters = list(LETTERS[: len(options)])
            listed = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, options, strict=True))
            questions.append(Choice(id=f"q{index}", prompt=f"{row['question'].strip()}\n{listed}", choices=letters))
            gold[f"q{index}"] = row["answer"]
        items.append(Item(context=article, questions=questions, gold=gold, task="race"))
    return items


def _unfair_tos(rows) -> list[Item]:
    """Eight questions about one clause. Clauses with no label at all are kept: most of them have none, and dropping
    them would measure a balance the task does not have."""
    items = []
    for row in rows:
        text = row["text"].strip()
        if not text:
            continue
        present = set(row["labels"])
        items.append(
            Item(
                context=f"A clause from a terms of service agreement:\n\n{text}",
                questions=[Boolean(id=name, prompt=prompt) for name, prompt in UNFAIR_TYPES],
                gold={name: (index in present) for index, (name, _) in enumerate(UNFAIR_TYPES)},
                task="unfair_tos",
            )
        )
    return items
