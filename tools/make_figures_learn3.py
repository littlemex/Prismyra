"""learn3: render the Snake tag's structured-feature learning curve next to learn2's own hashed-TF-IDF
curve, on the same two axes (accuracy vs. experience count, held-out agreement lower bound vs. experience
count) -- the comparison the task asks for ("結果を、learn2 の学習曲線と並べた図にする"). CPU-only, no
model/GPU involved, same as learn2's own `make_figures.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rows(curve_path: Path, config: str) -> list[dict]:
    d = json.loads(curve_path.read_text())
    return sorted(d[config], key=lambda r: r["train_n"])


def main() -> int:
    learn2_curve = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/akazawt/tmp/smr/learn2/sweep/snake_curve.json")
    learn3_curve = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Users/akazawt/tmp/smr/learn3/snake_curve_structured.json")
    simple_rule_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/Users/akazawt/tmp/smr/learn3/snake_simple_rule_baseline.json")
    out_png = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("/Users/akazawt/tmp/smr/learn3/snake_structured_vs_hashing.png")
    admit_threshold = 0.97

    hashing_default = _rows(learn2_curve, "default")
    hashing_cv = _rows(learn2_curve, "cv_selected")
    structured_default = _rows(learn3_curve, "default")
    structured_cv = _rows(learn3_curve, "cv_selected")
    simple_rule = json.loads(simple_rule_path.read_text()) if simple_rule_path.is_file() else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([r["train_n"] for r in hashing_default], [r["student_accuracy"] for r in hashing_default],
             marker="o", linestyle="--", color="#1f77b4", label="hashing TF-IDF (default) -- learn2")
    ax.plot([r["train_n"] for r in hashing_cv], [r["student_accuracy"] for r in hashing_cv],
             marker="s", linestyle="--", color="#1f77b4", alpha=0.5, label="hashing TF-IDF (CV-selected) -- learn2")
    ax.plot([r["train_n"] for r in structured_default], [r["student_accuracy"] for r in structured_default],
             marker="o", color="#d62728", label="snake_board_v1 (default) -- learn3")
    ax.plot([r["train_n"] for r in structured_cv], [r["student_accuracy"] for r in structured_cv],
             marker="s", color="#d62728", alpha=0.7, label="snake_board_v1 (CV-selected) -- learn3")
    prismyra_acc = hashing_default[0]["prismyra_accuracy"]
    ax.axhline(prismyra_acc, color="#2ca02c", linestyle=":", label=f"Prismyra accuracy ({prismyra_acc:.3f})")
    if simple_rule and simple_rule.get("eval_accuracy") is not None:
        ax.axhline(simple_rule["eval_accuracy"], color="#7f7f7f", linestyle=":",
                    label=f"hand rule, no training ({simple_rule['eval_accuracy']:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("train pool size (experiences)")
    ax.set_ylabel("accuracy on fixed eval set (gold)")
    ax.set_title("snake-move-up-v1: accuracy vs. experience count")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([r["train_n"] for r in hashing_default], [r["held_out_lower_bound_95"] for r in hashing_default],
             marker="o", linestyle="--", color="#1f77b4", label="hashing TF-IDF (default) -- learn2")
    ax.plot([r["train_n"] for r in hashing_cv], [r["held_out_lower_bound_95"] for r in hashing_cv],
             marker="s", linestyle="--", color="#1f77b4", alpha=0.5, label="hashing TF-IDF (CV-selected) -- learn2")
    ax.plot([r["train_n"] for r in structured_default], [r["held_out_lower_bound_95"] for r in structured_default],
             marker="o", color="#d62728", label="snake_board_v1 (default) -- learn3")
    ax.plot([r["train_n"] for r in structured_cv], [r["held_out_lower_bound_95"] for r in structured_cv],
             marker="s", color="#d62728", alpha=0.7, label="snake_board_v1 (CV-selected) -- learn3")
    if simple_rule and simple_rule.get("held_out_lower_bound_95") is not None:
        ax.axhline(simple_rule["held_out_lower_bound_95"], color="#7f7f7f", linestyle=":",
                    label=f"hand rule, no training ({simple_rule['held_out_lower_bound_95']:.3f})")
    ax.axhline(admit_threshold, color="#ff7f0e", linestyle=":", label=f"admit threshold ({admit_threshold})")
    ax.set_xscale("log")
    ax.set_xlabel("train pool size (experiences)")
    ax.set_ylabel("student/Prismyra agreement, Wilson 95% LB")
    ax.set_title("snake-move-up-v1: held-out agreement vs. experience count")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
