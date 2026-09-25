# Signals: probes on the context pass

`Prismyra(model, probes=[path, ...])` applies linear probes to the hidden state one layer produces at the last context
token while the context is read, and returns their values beside the answers as `Result.signals` (and `"signals"` in
the server's response). A probe is one dot product; loading one changed neither the latency (271.8 ms against
272.1 ms for one question, 715.5 ms against 716.0 ms for 64, on an L40S) nor any probability.

This recipe builds the one probe that earned its place: **self_solve**, the probability that this model, used as an
ordinary chat LLM, answers the context correctly -- a routing signal.

```bash
python problems.py problems.json                       # 4,300 checkable problems in six families
vllm serve <base checkpoint> --served-model-name qwen36 --port 8100 &
python label.py http://localhost:8100/v1/chat/completions labels.jsonl 5 problems.json
python collect.py <served checkpoint> problems.json prompt "Could you solve this problem correctly on your first attempt, without help?" features.npz
python export.py problems.json labels.jsonl features.npz 31 0.01 self_solve.json
```

The label is greedy correctness with thinking off and a 1,536-token budget, plus a five-sample rate; an answer cut off
at the budget counts as a failure, because the target is "solved within this budget". Features come from
the checkpoint that serves -- a probe fit on one checkpoint's hidden states means nothing on another's, so rebuild it
after merging an adapter or truncating layers. `export.py` folds the standardisation, the PCA and the logistic layer
into one weight vector and a bias.

## What was measured

On 4,300 problems (MMLU-Pro, GSM8K, MATH-500, ARC-Challenge, NQ-open, BBH), with AUROC computed within each family so
that knowing the family is worth nothing:

| signal | AUROC within family (random 5-fold) |
|---|---|
| probe on the last context token (layers 30-34 chosen) | **0.752** |
| asking "could you solve this?" through the read-out | 0.688 |
| TF-IDF on the problem text | 0.593 |
| family and length | 0.517 |

With a family held out entirely -- the probe fit on the other five -- it was better than asking on five of six:

| held-out family | asking | probe |
|---|---|---|
| ARC-Challenge | 0.786 | **0.846** |
| BBH | 0.491 | **0.599** |
| GSM8K | 0.671 | **0.672** |
| MATH-500 | 0.800 | **0.820** |
| NQ-open | 0.571 | **0.670** |
| MMLU-Pro | **0.759** | 0.706 |

Kept on this model at a 95% accuracy floor, ranking by the probe keeps 65.3% of the problems against 42.6% ranking by
asking (threshold set on the same data, so an upper bound for both). The shipped probe reads layer 31 and scores 0.886
overall under shuffled 5-fold cross-validation.

Two other signals were measured and are not shipped. For PII and for prompt injection, a probe fit on one source did
not transfer to another (PII: AUROC 0.69 at best against 0.94 for asking), although within one document it separated a
positive from its paired negative almost perfectly. For both, asking through the read-out is the better signal.

The one-pass path (`ask` with one question) computes the last context token inside a longer sequence, and a probe's
value there differs from the forked path's by a mean of 0.016 (largest 0.18 over 100 problems).
