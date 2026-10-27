"""xgb-student2's real-machine check: two tags, one new (banking77 intent classification, a short-text
classification task sized the way a production deployment would actually see it -- banking77's own 77
intents narrowed to however many this build's tokenizer can label at once, ~10,000 training utterances, not
the few-hundred-row toy run `tools/collect_and_fit.py` used), one reused (`snake-move-up-v1`'s own structured
board features, `tools/snake_features.py`, carried over unmodified from learn3/learn4 to confirm the admitted
task still admits on a fresh sample and with both student backends).

Not shipped as part of the package; a one-off script for this round's own verification, the same convention
`tools/collect_and_fit.py`/`tools/collect_fresh_eval.py` already use.

Why one collection pass, several fits: a teacher's probability distribution over a fixed (context, question,
options) triple does not depend on which student backend or which text tokenizer will later be trained on it
-- paying for a fresh forward pass per (backend, tokenizer) combination would be answering the same question
of the model 2x/4x over for nothing. This script calls the real engine exactly once per training row (through
`/v1/decide`, tagged and logged, the same path a real deployment takes) and once per eval row (through
`engine.ask` directly, bypassing the log, the same disjoint-by-construction split `collect_and_fit.py` already
uses), then reuses that one logged pool for every (backend, tokenizer) combination's own `fit_student` call --
all of which are CPU-only and cheap once the teacher's own answers are on disk.

Usage (on a box with the real model and a GPU, `HF_HOME` already pointed at a warm cache):

    python3.12 tools/collect_and_fit_xgb2.py \\
      --banking77-train-csv /work/xgb-student2-verify/data/banking77_train.csv \\
      --banking77-test-csv /work/xgb-student2-verify/data/banking77_test.csv \\
      --out-dir /work/xgb-student2-verify/run \\
      --banking77-n-train 9500 --banking77-n-eval 500 \\
      --snake-n-train 4000 --snake-n-eval 400
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import snake_features  # noqa: F401 - importing registers "snake_board_v1" as a side effect (module import time)
from snake_port import best_direction_heuristic, build_request, sample_states_with_up_legal

INTENT_QUESTION = "What is the customer's banking intent in this message?"
SNAKE_QUESTION = "Is moving up the best of the directions listed above?"


def load_banking77_labels() -> list[str]:
    path = Path(__file__).parent / "banking77_labels.json"
    return json.loads(path.read_text())


def load_banking77_csv(path: str) -> list[tuple[str, str]]:
    """``(text, category)`` pairs, in file order. The official PolyAI/banking77 HF dataset card's own loading
    script (`banking77.py`, not runnable under current `datasets` -- "Dataset scripts are no longer
    supported") reads from this exact URL
    (`raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{train,test}.csv`); this
    reads the same bytes from the same canonical source, not a different or unaudited mirror.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)  # header
        for row in reader:
            text, category = row
            rows.append((text, category))
    return rows


def rank_labels_by_frequency(rows: list[tuple[str, str]], all_labels: list[str]) -> list[str]:
    """``all_labels``, most frequent in ``rows`` first (ties broken by ``all_labels``'s own canonical order).
    Not truncated here -- callers slice the prefix they need, so one ranking serves both "how many classes
    can this tokenizer/branch-width actually support" (`max_choice_options_for_prompt`) and "then take that
    many" (its caller in `main`) without computing the same ranking twice.
    """
    from collections import Counter

    counts = Counter(category for _text, category in rows)
    return sorted(all_labels, key=lambda label: (-counts.get(label, 0), label))


def max_choice_options_for_prompt(tokenizer, question: str, ranked_labels: list[str], limit: int) -> int:
    """How many of ``ranked_labels`` (already ordered, most important first) a `kind="choice"` item naming
    ``question`` can actually offer at once on this real server, measured the same way the server itself
    would check it -- not just `prismyra.decide.choice_labels`'s own "does every option still get a distinct
    single-token label" check, but also `prismyra.engine.ShelfEngine.validate`'s width refusal: a rendered
    question (`prismyra.schema.render_question`'s own wrapper, which repeats the full option list inside
    "Answer with one of: ...") has to fit in the widest fixed branch this build's one-pass reader has
    (`prismyra.fork.WIDTHS[-1]`, 512 tokens). Found empirically here because a 77-way, then even a
    tokenizer-narrowed-to-68-way, banking77 choice prompt was measured on this build's real tokenizer to
    render past that limit -- a real constraint this script ran into, not a guess it is now avoiding by
    being cautious in advance.
    """
    from prismyra.decide import _LABEL_CANDIDATES
    from prismyra.fork import WIDTHS, branch_ids
    from prismyra.readout import plan
    from prismyra.schema import Choice

    def fits(k: int) -> bool:
        labels = ranked_labels[:k]
        symbols = dict(zip(labels, _LABEL_CANDIDATES[:k], strict=True))
        lines = "\n".join(f"{symbols[opt]}) {opt}" for opt in labels)
        prompt = f"{question}\n[options]\n{lines}"
        q = Choice(id="_width_probe", prompt=prompt, choices=tuple(symbols.values()))
        chosen = plan(q, tokenizer)
        return len(branch_ids(chosen.text, tokenizer)) <= WIDTHS[-1]

    limit = min(limit, len(ranked_labels))
    lo, hi, best = 2, limit, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def write_spec(spec_path: Path, banking77_labels: list[str]) -> None:
    entries = [
        {
            "task": "banking77-intent-v1",
            "question": INTENT_QUESTION,
            "options": banking77_labels,
            "kind": "choice",
            "normalize": "strip+lower",
            "retain_days": 30,
            "keep_hidden": False,
            "eval_set": "banking77_eval.jsonl",
        },
        {
            "task": "snake-move-up-v1",
            "question": SNAKE_QUESTION,
            "options": ["no", "yes"],
            "kind": "boolean",
            "normalize": "none",
            "retain_days": 30,
            "keep_hidden": False,
            "eval_set": "snake_eval.jsonl",
            "features": "snake_board_v1",
        },
    ]
    spec_path.write_text(json.dumps(entries, indent=2))


def collect_banking77(
    client, engine, rows: list[tuple[str, str]], labels: list[str], n_train: int, n_eval: int, out_dir: Path, seed: int
):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    train_idx = idx[:n_train]

    t0 = time.monotonic()
    print(f"[banking77] logging {len(train_idx)} train rows through /v1/decide (tagged) ...", flush=True)
    for n, i in enumerate(train_idx, 1):
        text, _category = rows[i]
        resp = client.post(
            "/v1/decide",
            json={
                "kind": "choice",
                "task": "banking77-intent-v1",
                "state": text,
                "question": INTENT_QUESTION,
                "options": labels,
            },
        )
        assert resp.status_code == 200, resp.text
        if n % 500 == 0:
            elapsed = time.monotonic() - t0
            print(
                f"  [banking77] {n}/{len(train_idx)} ({elapsed:.1f}s elapsed, {elapsed / n * 1000:.1f} ms/row)",
                flush=True,
            )
    print(f"[banking77] train collection done in {time.monotonic() - t0:.1f}s", flush=True)
    return train_idx


def collect_banking77_eval(
    client, engine, rows, labels: list[str], train_idx: set[int], n_eval: int, out_dir: Path, seed: int
):
    """Eval rows come from a different CSV file (`banking_data/test.csv`) than train (`train.csv`) -- disjoint
    by construction, not by avoiding indices already drawn for train (``train_idx`` is accepted only so a
    caller can assert that at the call site, see ``main``). Answered through `engine.ask` directly, never
    through the server or the log, matching `tools/collect_and_fit.py`'s own eval collection -- but relabelled
    exactly as `prismyra.decide.parse_item` relabels a live `/v1/decide` `choice` item (the same single-token
    symbol substitution), so the "Prismyra accuracy" this produces is the accuracy of the same deployed
    read-out path, not of a differently-scored raw `Choice`.
    """
    from prismyra.decide import choice_labels
    from prismyra.schema import Choice

    symbol_pool = choice_labels(engine.tokenizer, limit=len(labels))
    symbols = dict(zip(labels, symbol_pool, strict=True))
    lines = "\n".join(f"{symbols[opt]}) {opt}" for opt in labels)
    prompt = f"{INTENT_QUESTION}\n[options]\n{lines}"

    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    eval_idx = idx[:n_eval]

    t0 = time.monotonic()
    print(f"[banking77] building {len(eval_idx)} eval rows via engine.ask directly (never logged) ...", flush=True)
    eval_rows = []
    for n, i in enumerate(eval_idx, 1):
        text, category = rows[i]
        q = Choice(id="intent", prompt=prompt, choices=tuple(symbols.values()))
        result = engine.ask(text, [q])
        probs_by_symbol = dict(result["intent"].probabilities)
        probs_by_label = {opt: probs_by_symbol[symbols[opt]] for opt in labels}
        eval_rows.append(
            {
                "context": text,
                "gold": category,
                "options": labels,
                "prismyra_probabilities": probs_by_label,
            }
        )
        if n % 100 == 0:
            print(f"  [banking77 eval] {n}/{len(eval_idx)}", flush=True)
    (out_dir / "banking77_eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
    print(f"[banking77] eval collection done in {time.monotonic() - t0:.1f}s", flush=True)


def collect_snake(client, engine, n_train: int, n_eval: int, out_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    train_states = sample_states_with_up_legal(rng, n_train)
    eval_states = sample_states_with_up_legal(rng, n_eval)  # same rng, continues past train draws: disjoint

    t0 = time.monotonic()
    print(f"[snake] logging {len(train_states)} train states through /ask (tagged) ...", flush=True)
    for n, s in enumerate(train_states, 1):
        context, questions, _ = build_request(s)
        resp = client.post("/ask", json={"context": context, "questions": questions})
        assert resp.status_code == 200, resp.text
        if n % 500 == 0:
            print(f"  [snake] {n}/{len(train_states)}", flush=True)
    print(f"[snake] train collection done in {time.monotonic() - t0:.1f}s", flush=True)

    print(f"[snake] building {len(eval_states)} eval states via engine.ask directly (never logged) ...", flush=True)
    from prismyra.schema import Boolean

    eval_rows = []
    for s in eval_states:
        context, questions, cands = build_request(s)
        up_q = next(q for q in questions if q["prompt"] == SNAKE_QUESTION)
        gold_dir = best_direction_heuristic(cands)
        gold = "yes" if gold_dir == "up" else "no"
        result = engine.ask(context, [Boolean(id=up_q["id"], prompt=up_q["prompt"])])
        eval_rows.append(
            {
                "context": context,
                "gold": gold,
                "options": ["no", "yes"],
                "prismyra_probabilities": dict(result[up_q["id"]].probabilities),
            }
        )
    (out_dir / "snake_eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
    print(f"[snake] eval collection done in {time.monotonic() - t0:.1f}s", flush=True)


def score_combo(
    tag, train, held_out, eval_path: Path, *, backend_name: str, text_tokenizer: str, num_boost_round: int = 100
):
    """One (backend, text_tokenizer) combination's own artifact, fit on an already-collected ``train``/
    ``held_out`` split -- no new engine call. ``tag`` is overridden per call (`dataclasses.replace`) so one
    collection pass can be scored under every combination this round compares, exactly the saving this
    script's own module docstring describes.
    """
    from prismyra.learn.fit import (
        StudentArtifact,
        _score_eval,
        argmax,
        fit_student,
        load_eval_set,
        tau_sweep,
        wilson_lower_bound,
    )

    combo_tag = dataclasses.replace(tag, student=backend_name, text_tokenizer=text_tokenizer)
    t0 = time.monotonic()
    model_text, vectorizer, backend = fit_student(
        combo_tag, train=train, dims=256, num_threads=4, seed=0, num_boost_round=num_boost_round, min_data_in_leaf=5
    )
    fit_s = time.monotonic() - t0
    loaded = backend.load(model_text)

    import numpy as np

    def predict_fn(context: str) -> dict[str, float]:
        vec = vectorizer.transform(context, combo_tag.question)
        (probs,) = loaded.predict_proba(np.asarray([vec], dtype=np.float64))
        return dict(zip(combo_tag.options, (float(p) for p in probs), strict=True))

    eval_items = load_eval_set(eval_path)
    eval_metrics = _score_eval(eval_items, predict_fn)

    agree = 0
    for e in held_out:
        probs = predict_fn(e.context)
        agree += int(argmax(probs) == argmax(e.probabilities))
    lower_bound = wilson_lower_bound(agree, len(held_out)) if held_out else 0.0

    metrics = {
        "train_n": len(train),
        "held_out_n": len(held_out),
        "eval": eval_metrics,
        "held_out_agreement": {
            "n": len(held_out),
            "agree": agree,
            "rate": (agree / len(held_out)) if held_out else 0.0,
            "lower_bound_95": lower_bound,
        },
        "fit_seconds": fit_s,
        "tau_sweep": tau_sweep(held_out, predict_fn),
    }

    required_accuracy = combo_tag.min_accuracy_ratio * eval_metrics.get("prismyra_accuracy", 0.0)
    condition_1 = "prismyra_accuracy" in eval_metrics and eval_metrics["student_accuracy"] >= required_accuracy
    condition_2 = len(held_out) > 0 and lower_bound >= combo_tag.min_match_rate
    reasons = []
    if not condition_1:
        reasons.append(f"eval accuracy {eval_metrics.get('student_accuracy')} below {required_accuracy:.4f}")
    if not condition_2:
        reasons.append(f"held-out agreement lower bound {lower_bound:.4f} below {combo_tag.min_match_rate}")

    artifact = StudentArtifact(
        tag=combo_tag.task,
        spec_version="n/a-offline-combo",
        package_version="n/a",
        backbone="n/a",
        options=combo_tag.options,
        kind=combo_tag.kind,
        dims=256,
        idf=vectorizer.hashing.idf if vectorizer.hashing is not None else (),
        model_text=model_text,
        created_at=time.time(),
        admitted=condition_1 and condition_2,
        reasons=tuple(reasons),
        metrics=metrics,
        features=combo_tag.features,
        feature_names=vectorizer.names,
        backend=backend.name,
        backend_version=backend.library_version(),
        text_tokenizer=vectorizer.text_tokenizer,
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banking77-train-csv", required=True)
    parser.add_argument("--banking77-test-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--banking77-n-train", type=int, default=9500)
    parser.add_argument("--banking77-n-eval", type=int, default=500)
    parser.add_argument("--snake-n-train", type=int, default=4000)
    parser.add_argument("--snake-n-eval", type=int, default=400)
    parser.add_argument("--held-out-fraction", type=float, default=0.1)
    parser.add_argument("--learning-curve-sizes", type=str, default="2000,4000,6000,8500")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()

    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_banking77_csv(args.banking77_train_csv)
    test_rows = load_banking77_csv(args.banking77_test_csv)
    print(f"banking77: {len(train_rows)} train rows (file), {len(test_rows)} test rows (file)", flush=True)

    # A `kind="choice"` item needs one single-token label per option (`prismyra.decide.choice_labels`) *and*
    # has to fit the one-pass reader's widest fixed branch once rendered (`prismyra.engine.ShelfEngine.
    # validate`, `prismyra.fork.WIDTHS[-1]` = 512 tokens) -- both are properties of this tokenizer and this
    # build, not numbers this script gets to pick. banking77's real 77 intents turned out to be more than
    # either allows here (measured, not assumed: this tokenizer found only 68 single-token labels, and even
    # 68's own rendered prompt is 635 tokens, past the 512 limit -- `max_choice_options_for_prompt` finds the
    # largest count that clears both). Narrowed by real training frequency (`rank_labels_by_frequency`), not
    # by an arbitrary alphabetical cut, and both train and eval rows are filtered to the same narrowed set so
    # no row's own true label is ever excluded from the options it is scored against.
    all_labels = load_banking77_labels()
    from transformers import AutoTokenizer

    from prismyra.decide import choice_labels

    print(f"loading {model}'s tokenizer only, to size the choice label pool ...", flush=True)
    probe_tokenizer = AutoTokenizer.from_pretrained(model)
    ranked_labels = rank_labels_by_frequency(train_rows, all_labels)
    single_token_n = len(choice_labels(probe_tokenizer, limit=len(all_labels)))
    final_n = max_choice_options_for_prompt(probe_tokenizer, INTENT_QUESTION, ranked_labels, single_token_n)
    print(
        f"banking77: {len(all_labels)} intents in the real dataset; this tokenizer supports {single_token_n} "
        f"single-token choice labels, and the rendered prompt fits the 512-token branch width for at most "
        f"{final_n} of them",
        flush=True,
    )
    banking77_labels = ranked_labels[:final_n]
    train_rows = [(t, c) for t, c in train_rows if c in banking77_labels]
    test_rows = [(t, c) for t, c in test_rows if c in banking77_labels]
    print(
        f"banking77: narrowed to {len(banking77_labels)} intents -- {len(train_rows)} train rows, "
        f"{len(test_rows)} test rows remain after filtering to them",
        flush=True,
    )

    spec_path = out_dir / "learn.json"
    write_spec(spec_path, banking77_labels)

    from fastapi.testclient import TestClient

    from prismyra.server import create_app

    print(f"loading {model} ...", flush=True)
    app = create_app(model, max_queue=64, learn_spec=str(spec_path), learn_log_dir=str(out_dir / "experience"))
    client = TestClient(app)
    engine = app.state.engine

    collect_banking77(
        client, engine, train_rows, banking77_labels, args.banking77_n_train, args.banking77_n_eval, out_dir, args.seed
    )
    collect_banking77_eval(
        client, engine, test_rows, banking77_labels, set(), args.banking77_n_eval, out_dir, args.seed + 1
    )
    collect_snake(client, engine, args.snake_n_train, args.snake_n_eval, out_dir, args.seed + 2)

    print("learn stats:", json.dumps(client.get("/stats").json().get("learn", {}), indent=2)[:3000], flush=True)

    import prismyra
    from prismyra.learn.experience import LocalExperienceSource
    from prismyra.learn.fit import dedup, split_train_held_out
    from prismyra.learn.spec import load_spec

    spec = load_spec(spec_path)
    results: dict = {}

    # banking77: 4 combinations, one collection pass.
    tag = spec.by_task("banking77-intent-v1")
    assert tag is not None
    all_exp = dedup(list(LocalExperienceSource(out_dir / "experience").iter_experiences(tag.task)))
    train, held_out = split_train_held_out(all_exp, args.held_out_fraction, args.seed)
    print(f"banking77: deduped_n={len(all_exp)} train_n={len(train)} held_out_n={len(held_out)}", flush=True)
    eval_path = out_dir / "banking77_eval.jsonl"
    combos = [(b, t) for b in ("lightgbm", "xgboost") for t in ("words", "char3")]
    for backend_name, text_tokenizer in combos:
        key = f"banking77-intent-v1__{backend_name}__{text_tokenizer}"
        print(f"\n=== fitting {key} (full train_n={len(train)}) ===", flush=True)
        artifact = score_combo(
            tag, train, held_out, eval_path, backend_name=backend_name, text_tokenizer=text_tokenizer
        )
        (out_dir / f"{key}.json").write_text(json.dumps(artifact.to_json(), indent=2))
        results[key] = artifact.metrics | {"admitted": artifact.admitted, "reasons": artifact.reasons}
        print(f"{key}: admitted={artifact.admitted} reasons={artifact.reasons}", flush=True)
        print(f"  eval: {json.dumps(artifact.metrics['eval'])}", flush=True)
        print(f"  held_out_agreement: {json.dumps(artifact.metrics['held_out_agreement'])}", flush=True)

    # Learning curve: one representative combo (lightgbm + words, this project's own existing default),
    # varying only how much of the *same* shuffled train pool is handed to `fit_student` -- `held_out` and
    # the external gold `eval_path` stay fixed across every point, so the only thing that changes between
    # rows is how much training data the student saw.
    curve_sizes = [int(x) for x in args.learning_curve_sizes.split(",")]
    curve = []
    for requested_size in curve_sizes:
        size = min(requested_size, len(train))
        subset = train[:size]
        print(f"\n=== learning curve point: banking77 lightgbm/words, train_n={size} ===", flush=True)
        artifact = score_combo(tag, subset, held_out, eval_path, backend_name="lightgbm", text_tokenizer="words")
        curve.append({"train_n": size, "admitted": artifact.admitted, **artifact.metrics})
        print(
            f"  train_n={size}: eval_acc={artifact.metrics['eval']['student_accuracy']:.4f} "
            f"held_out_agreement={artifact.metrics['held_out_agreement']['rate']:.4f} admitted={artifact.admitted}",
            flush=True,
        )
    (out_dir / "banking77_learning_curve.json").write_text(json.dumps(curve, indent=2))

    # snake: reused structured features, both backends, one collection pass.
    snake_tag = spec.by_task("snake-move-up-v1")
    assert snake_tag is not None
    snake_exp = dedup(list(LocalExperienceSource(out_dir / "experience").iter_experiences(snake_tag.task)))
    snake_train, snake_held_out = split_train_held_out(snake_exp, args.held_out_fraction, args.seed)
    print(
        f"\nsnake: deduped_n={len(snake_exp)} train_n={len(snake_train)} held_out_n={len(snake_held_out)}", flush=True
    )
    snake_eval_path = out_dir / "snake_eval.jsonl"
    for backend_name in ("lightgbm", "xgboost"):
        key = f"snake-move-up-v1__{backend_name}"
        print(f"\n=== fitting {key} ===", flush=True)
        artifact = score_combo(
            snake_tag, snake_train, snake_held_out, snake_eval_path, backend_name=backend_name, text_tokenizer="words"
        )
        (out_dir / f"{key}.json").write_text(json.dumps(artifact.to_json(), indent=2))
        results[key] = artifact.metrics | {"admitted": artifact.admitted, "reasons": artifact.reasons}
        print(f"{key}: admitted={artifact.admitted} reasons={artifact.reasons}", flush=True)
        print(f"  eval: {json.dumps(artifact.metrics['eval'])}", flush=True)
        print(f"  held_out_agreement: {json.dumps(artifact.metrics['held_out_agreement'])}", flush=True)

    (out_dir / "summary.json").write_text(
        json.dumps({"prismyra_version": prismyra.__version__, "model": model, "results": results}, indent=2)
    )
    print("\nALL DONE. Wrote summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
