#!/usr/bin/env python3
"""fp8spd6 (round7, task2): torch.equal gate for `interleave.read_and_branch_shelf_many` /
`Prismyra._shelf_ask_interleaved_many`. For N in {2, 3}, each with its own distinct question count from
{1, 3, 7}, compares the new fused multi-document path against the existing two-step path (`Shelf.put_many`
then `Shelf.ask`) on fresh `Shelf`s, same documents, same questions, same order.
"""
import os
import sys

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
DOC_TEMPLATES = [
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。",
    "配送は注文確定から二営業日以内に発送します。離島・一部地域では追加で二日ほどかかる場合があります。"
    "配送業者の都合による遅延については当社は責任を負いません。",
    "会員登録は無料です。登録情報に変更があった場合は速やかにマイページから更新してください。"
    "長期間利用がない場合、ポイントは失効することがあります。",
]


def questions_for(n, tag):
    return [Boolean(id=f"{tag}q{i}", prompt=f"条項 {i} はこの文書の主題について述べているか。") for i in range(n)]


def probs_of(result):
    return torch.tensor(
        [v for qid in sorted(result.answers) for v in result.answers[qid].probabilities.values()],
        dtype=torch.float64,
    )


def two_step(engine, contexts, questions_per_doc):
    shelf = engine.open_shelf()
    try:
        handles = shelf.put_many(contexts)
        asked = {h: qs for h, qs in zip(handles, questions_per_doc, strict=True)}
        answers = shelf.ask(asked)
        return [answers[h] for h in handles]
    finally:
        shelf.close()


def fused(engine, contexts, questions_per_doc):
    shelf = engine.open_shelf()
    try:
        triples = engine._shelf_ask_interleaved_many(shelf, contexts, questions_per_doc)
        return [r for r, _h, _s in triples]
    finally:
        shelf.close()


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True, paged=True)

    all_ok = True
    for combo_name, counts in [("N=2 {1,7}", [1, 7]), ("N=2 {3,3}", [3, 3]), ("N=3 {1,3,7}", [1, 3, 7])]:
        n = len(counts)
        contexts = [DOC_TEMPLATES[i % len(DOC_TEMPLATES)] for i in range(n)]
        questions_per_doc = [questions_for(c, f"d{i}_") for i, c in enumerate(counts)]

        baseline = two_step(engine, contexts, questions_per_doc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        new = fused(engine, contexts, questions_per_doc)

        for i, (b, f) in enumerate(zip(baseline, new, strict=True)):
            pb, pf = probs_of(b), probs_of(f)
            eq = torch.equal(pb, pf)
            maxdiff = (pb - pf).abs().max().item()
            all_ok = all_ok and eq
            print(f"[{combo_name}] doc{i} (n={counts[i]}) torch.equal={eq} maxdiff={maxdiff:.6e}", flush=True)

    print(f"ALL_EQUAL={all_ok}")


if __name__ == "__main__":
    main()
