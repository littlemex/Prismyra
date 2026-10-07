#!/usr/bin/env python3
"""fp8spd6 (round8, task1): torch.equal gate for `read_and_branch_shelf_many` when the fused documents have
very different token lengths -- not tested in round7's own gate (all three `DOC_TEMPLATES` were a similar
short length). A long document (~5,000 tokens) and a short one (~50 tokens) answered together, each with its
own question count, compared against the existing two-step path.
"""
import os
import sys

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
SHORT_DOC = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
)
LONG_PARAGRAPH = (
    "配送は注文確定から二営業日以内に発送します。離島・一部地域では追加で二日ほどかかる場合があります。"
    "配送業者の都合による遅延については当社は責任を負いません。"
)


def build_long(pad_to_tokens, tokenizer):
    text = LONG_PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += LONG_PARAGRAPH
    return text


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
    tokenizer = engine.tokenizer
    long_doc = build_long(5000, tokenizer)
    long_tok = len(tokenizer(long_doc)["input_ids"])
    short_tok = len(tokenizer(SHORT_DOC)["input_ids"])
    print(f"long={long_tok} tokens, short={short_tok} tokens", flush=True)

    all_ok = True
    for combo_name, docs_counts in [
        ("short+long {3,5}", [(SHORT_DOC, 3), (long_doc, 5)]),
        ("long+short {5,3}", [(long_doc, 5), (SHORT_DOC, 3)]),
        ("short+short+long {1,1,7}", [(SHORT_DOC, 1), (SHORT_DOC, 1), (long_doc, 7)]),
    ]:
        contexts = [d for d, _ in docs_counts]
        questions_per_doc = [questions_for(c, f"d{i}_") for i, (_, c) in enumerate(docs_counts)]

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
            print(f"[{combo_name}] doc{i} torch.equal={eq} maxdiff={maxdiff:.6e}", flush=True)

    print(f"ALL_EQUAL={all_ok}")


if __name__ == "__main__":
    main()
