"""A learning-free "scratch KV": work the model already understands, built through its own token space.

A frozen model can only read what lands in a distribution it already knows how to read: the KV it would have
written for tokens it actually saw. So the three variants below never write a hidden state or a KV cell
directly. They each add real tokens (or, for Soft Thinking, a probability-weighted mixture of real token
embeddings -- still inside the convex hull of the embedding table, which is Soft Thinking's own argument for
why it stays "in-distribution") to the branch's own row, before the existing closed-form read-out runs.

Three variants, in increasing distance from "just add literal tokens":

1. `evidence_question` -- copy `k` sentences out of the context, verbatim, into the question's own row,
   after the question (and optionally before it again). This is the cheapest and most literal: it does not
   even use a new mechanism, it is simply a different `Question.prompt`.
2. `memo_question` -- same idea, but the inserted text is not copied from the document: it is written by a
   model (frozen, no training) that read the document once. See `scratchkv.LEAK_PATTERNS` for the mechanical
   leak check this needs.
3. `soft_thinking_readout` -- no new discrete tokens at all. `K` extra positions are appended to the row,
   each one the probability-weighted mixture of the *same* model's own next-token distribution and its own
   input embedding table (Zhang et al., arXiv:2505.15778, "Soft Thinking"). Every mixture is still a convex
   combination of embeddings the model has read before, which is the sense in which it stays in-distribution
   that a hidden state written by hand would not.

Nothing here is trained. Nothing here changes the context; only the question's own row (its tokens, or in
(3) its row's extra positions) is touched, so the "the context is read once, every branch is private" property
`docs/FORK.md` describes still holds -- these are just unusually long or unusual branches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from .schema import Choice, Question
from .readout import plan, score


# ----------------------------------------------------------------------------------------------- sentences


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_split(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace. Good enough for the devset's prose and
    dialogue; not a real sentence boundary detector. Empty pieces (consecutive punctuation) are dropped, and
    whitespace is not otherwise touched -- a selected sentence is byte-identical to its slice of `text`."""
    pieces = [p for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return pieces or [text.strip()]


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_topk(sentences: list[str], query: str, k: int) -> list[int]:
    """Indices of the `k` sentences with the highest BM25Okapi score against `query`. Ties keep document
    order (stable sort on `-score, index`), so this is deterministic given the same sentence list and query."""
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize_for_bm25(s) for s in sentences]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize_for_bm25(query))
    order = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    return order[: min(k, len(sentences))]


class AttentionProbe:
    """Captures (query, key) at chosen `FlashAttention` layers during one `_read_one_pass` call, by calling
    the module's own `_project` (same weights, same RoPE) from a forward pre-hook. The real forward is left
    completely alone -- this only reads what `_project` would have computed anyway, a second time, for
    ranking. Because `_read_one_pass` prefills the whole sequence in one call, `hidden_states` at the hook
    already covers every position, so this needs no extra forward pass.
    """

    def __init__(self, backbone, layer_indices_1based: set[int]):
        self.captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = []
        for _name, module in backbone.named_modules():
            if type(module).__name__ != "FlashAttention":
                continue
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None or layer_idx + 1 not in layer_indices_1based:
                continue
            self.handles.append(
                module.register_forward_pre_hook(self._hook(layer_idx + 1, module), with_kwargs=True)
            )

    def _hook(self, li: int, module):
        def hook(mod, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            position_embeddings = args[1] if len(args) > 1 else kwargs["position_embeddings"]
            with torch.no_grad():
                q, k, _v, _gate, _shape = mod._project(hidden_states, position_embeddings)
            self.captured[li] = (q.detach(), k.detach())
            return None

        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def full_attention_layers(engine) -> list[int]:
    """1-based layer numbers whose `layer_types` entry is `full_attention` -- the only layers an eager
    query/key recomputation can give an interpretable per-token attention map for; the gated-delta layers
    are a recurrence/convolution, not an attention matrix."""
    decoder = getattr(engine.config, "text_config", engine.config)
    layer_types = list(getattr(decoder, "layer_types", []) or [])
    if not layer_types:
        return list(range(1, int(getattr(decoder, "num_hidden_layers", 0)) + 1))
    return [i + 1 for i, t in enumerate(layer_types) if t == "full_attention"]


def _expand_kv_heads(k: torch.Tensor, n_query_heads: int) -> torch.Tensor:
    """Grouped-query attention: `k` has fewer heads than `q` (this checkpoint has 2 KV heads against 16
    query heads), so each KV head is shared by `n_query_heads // k.shape[1]` query heads. Repeats along the
    head dimension to match, the same convention `repeat_kv` uses in every GQA implementation."""
    n_kv_heads = k.shape[1]
    if n_kv_heads == n_query_heads:
        return k
    n_rep = n_query_heads // n_kv_heads
    return k.repeat_interleave(n_rep, dim=1)


def rank_sentences_by_attention(
    engine, context: str, question: Question, sentences: list[str], probe_layers: list[int]
) -> list[int] | None:
    """Read `context` + `question` once (the plain, un-augmented question), average the question's own last
    position's attention over `probe_layers`, map each context token's share onto the sentence it falls in
    (by character offset -> token offset via the tokenizer's fast offset mapping), and return every sentence
    index ordered best-first. Returns `None` if `encode()` wrapped the context in media tokens the plain
    tokenizer call does not produce (the mapping would misalign silently otherwise) -- callers skip the
    attention variant for that row and record the skip, rather than guess."""
    from .fork import branch_ids
    from .media import encode

    planned = plan(question, engine.tokenizer)
    encoded = encode(context, None, None, engine.processor, engine.tokenizer, engine.device)
    suffix = branch_ids(planned.text, engine.tokenizer)
    ids = torch.cat([encoded.input_ids, torch.tensor([suffix], device=engine.device)], dim=1)
    ctx_len = encoded.input_ids.shape[1]

    enc = engine.tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    if len(offsets) != ctx_len:
        return None

    sent_spans = []
    pos = 0
    for s in sentences:
        start = context.index(s, pos)
        end = start + len(s)
        sent_spans.append((start, end))
        pos = end

    tok_sentence: list[int | None] = [None] * ctx_len
    si = 0
    for ti in range(ctx_len):
        c0 = offsets[ti][0]
        while si < len(sent_spans) - 1 and c0 >= sent_spans[si][1]:
            si += 1
        if sent_spans[si][0] <= c0 < sent_spans[si][1]:
            tok_sentence[ti] = si

    probe = AttentionProbe(engine.backbone, set(probe_layers))
    try:
        with engine._lock, torch.inference_mode():
            engine._read_one_pass(ids)
        mass = [0.0] * len(sentences)
        for li in probe_layers:
            q_t, k_t = probe.captured[li]
            head_dim = q_t.shape[-1]
            k_t = _expand_kv_heads(k_t, q_t.shape[1])
            q_last = q_t[0, :, -1:, :].float()
            k_ctx = k_t[0, :, :ctx_len, :].float()
            scores = torch.einsum("hqd,hkd->hqk", q_last, k_ctx) * (head_dim**-0.5)
            weights = torch.softmax(scores, dim=-1).mean(dim=0).squeeze(0).tolist()
            for ti in range(ctx_len):
                if tok_sentence[ti] is not None:
                    mass[tok_sentence[ti]] += weights[ti]
    finally:
        probe.close()
    return sorted(range(len(sentences)), key=lambda i: (-mass[i], i))


def attention_topk_sentences(
    engine, context: str, question: Question, sentences: list[str], k: int, probe_layers: list[int]
) -> list[int] | None:
    order = rank_sentences_by_attention(engine, context, question, sentences, probe_layers)
    return None if order is None else order[: min(k, len(sentences))]


# ----------------------------------------------------------------------------------------------- question builders


def _rebuild(question: Question, prompt: str) -> Question:
    if isinstance(question, Choice):
        return Choice(id=question.id, prompt=prompt, choices=question.choices)
    return type(question)(id=question.id, prompt=prompt)


def evidence_question(question: Question, evidence_sentences: list[str], repeat_question: bool) -> Question:
    """`{prompt}\\n\\nEvidence:\\n{sentences}\\n\\n{prompt again, if repeat_question}`. The context itself is
    never touched; only this question's own branch row grows."""
    block = "\n".join(evidence_sentences)
    prompt = f"{question.prompt}\n\nEvidence:\n{block}"
    if repeat_question:
        prompt = f"{prompt}\n\n{question.prompt}"
    return _rebuild(question, prompt)


def prompt_repetition_question(question: Question) -> Question:
    """The question asked twice, back to back, with no evidence at all -- isolates "reread the question" from
    "reread chosen evidence"."""
    return _rebuild(question, f"{question.prompt}\n\n{question.prompt}")


def memo_question(question: Question, memo: str) -> Question:
    """`{memo}\\n\\n{prompt}` -- a note written by a separate, frozen model, placed ahead of the question."""
    return _rebuild(question, f"{memo}\n\n{question.prompt}")


# ----------------------------------------------------------------------------------------------- leak check


LEAK_PATTERNS = [
    re.compile(r"\bthe answer is\b", re.I),
    re.compile(r"\boption\s+[A-Za-z]\b", re.I),
    re.compile(r"^\s*[A-Z][.):]", re.M),
    re.compile(r"\b(yes|no)\b[.!]?\s*$", re.I),
]


def memo_leaks(memo: str, correct_option_text: str) -> bool:
    """A mechanical (not semantic) leak check: does the memo contain the correct option's text verbatim, or
    one of a short list of answer-announcing patterns? Mirrors loop0's oracle leak scan, without the semantic
    (ask-another-model) half -- see RUN-ws2.md 1.2 for why that half was dropped for this cheap probe."""
    if correct_option_text.strip().lower() in memo.lower():
        return True
    return any(p.search(memo) for p in LEAK_PATTERNS)


# ----------------------------------------------------------------------------------------------- soft thinking


@dataclass(frozen=True)
class SoftThinkingResult:
    probabilities: list[float]
    options: list[str]
    steps: int


def _branch_ids_tensor(engine, context: str, question: Question):
    from .fork import branch_ids
    from .media import encode

    planned = plan(question, engine.tokenizer)
    encoded = encode(context, None, None, engine.processor, engine.tokenizer, engine.device)
    suffix = branch_ids(planned.text, engine.tokenizer)
    ids = torch.cat([encoded.input_ids, torch.tensor([suffix], device=engine.device)], dim=1)
    return ids, planned


def soft_thinking_readout(engine, context: str, question: Question, steps: int) -> SoftThinkingResult:
    """Append `steps` continuous "concept token" positions (Zhang et al. 2505.15778, eq. 5: the
    probability-weighted mixture of the model's own input-embedding rows) to the question's row, then read
    out at the new last position with the model's existing closed-form scoring. Every step re-reads the whole
    (growing) sequence with `inputs_embeds` and `use_cache=False` -- the same "no incremental cache" choice
    `loop.loop_one_pass` makes, for the same reason: it needs no custom cache object, at the cost of
    recomputing the gated-delta recurrence from scratch each step (negligible next to `steps <= 16`).
    """
    ids, planned = _branch_ids_tensor(engine, context, question)
    embed = engine.backbone.get_input_embeddings()
    unembed = engine.unembedding
    with engine._lock, torch.inference_mode():
        embeds = embed(ids)  # (1, L, H)
        for _ in range(steps):
            out = engine.backbone(inputs_embeds=embeds, use_cache=False)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            last = hidden[0, -1]  # (H,) -- already final-normed, same as `_read_one_pass`'s return
            logits = (last.float() @ unembed.float().t())
            p = torch.softmax(logits, dim=-1)
            soft = (p.unsqueeze(0) @ embed.weight.float()).to(embeds.dtype)  # (1, H)
            embeds = torch.cat([embeds, soft.unsqueeze(1)], dim=1)
        out = engine.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        final = hidden[0, -1:]
    (p,) = score(final, unembed, [planned.token_ids])
    return SoftThinkingResult(probabilities=p.float().tolist(), options=list(question.options), steps=steps)


def random_token_control_readout(engine, context: str, question: Question, steps: int, rng) -> SoftThinkingResult:
    """Same shape of intervention as `soft_thinking_readout` (append `steps` extra positions before
    read-out), but each extra position is one real, discrete token embedding drawn at random from the
    context's own tokens -- not a continuous mixture. Isolates "a continuous concept-token mixture helps"
    from "any `steps` extra positions, even junk ones in-distribution by construction, help"."""
    ids, planned = _branch_ids_tensor(engine, context, question)
    embed = engine.backbone.get_input_embeddings()
    unembed = engine.unembedding
    ctx_tokens = ids[0].tolist()
    picks = [ctx_tokens[rng.randrange(len(ctx_tokens))] for _ in range(steps)]
    extra = torch.tensor([picks], device=ids.device)
    with engine._lock, torch.inference_mode():
        full_ids = torch.cat([ids, extra], dim=1)
        embeds = embed(full_ids)
        out = engine.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        final = hidden[0, -1:]
    (p,) = score(final, unembed, [planned.token_ids])
    return SoftThinkingResult(probabilities=p.float().tolist(), options=list(question.options), steps=steps)
