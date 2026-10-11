"""learn2: render the two tags' learning curves (student accuracy & Prismyra-agreement lower bound vs.
train-pool size, default vs CV-selected config) to PNG. CPU-only, no model/GPU involved -- just
matplotlib over the JSON train_sweep.py already wrote.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_tag(curve_path: Path, out_png: Path, title: str, admit_threshold: float = 0.97) -> None:
    d = json.loads(curve_path.read_text())
    default_rows = sorted(d["default"], key=lambda r: r["train_n"])
    cv_rows = sorted(d["cv_selected"], key=lambda r: r["train_n"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(
        [r["train_n"] for r in default_rows],
        [r["student_accuracy"] for r in default_rows],
        marker="o",
        label="student (default config)",
        color="#1f77b4",
    )
    ax.plot(
        [r["train_n"] for r in cv_rows],
        [r["student_accuracy"] for r in cv_rows],
        marker="s",
        label="student (CV-selected config)",
        color="#ff7f0e",
    )
    prismyra_acc = default_rows[0].get("prismyra_accuracy")
    if prismyra_acc is not None:
        ax.axhline(prismyra_acc, color="#2ca02c", linestyle="--", label=f"Prismyra accuracy ({prismyra_acc:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("train pool size (experiences)")
    ax.set_ylabel("accuracy on fixed eval set (gold)")
    ax.set_title(f"{title}: accuracy vs. experience count")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(
        [r["train_n"] for r in default_rows],
        [r["held_out_lower_bound_95"] for r in default_rows],
        marker="o",
        label="agreement LB (default)",
        color="#1f77b4",
    )
    ax.plot(
        [r["train_n"] for r in cv_rows],
        [r["held_out_lower_bound_95"] for r in cv_rows],
        marker="s",
        label="agreement LB (CV-selected)",
        color="#ff7f0e",
    )
    ax.axhline(admit_threshold, color="#d62728", linestyle="--", label=f"admit threshold ({admit_threshold})")
    ax.set_xscale("log")
    ax.set_xlabel("train pool size (experiences)")
    ax.set_ylabel("student/Prismyra agreement, Wilson 95% LB")
    ax.set_title(f"{title}: held-out agreement vs. experience count")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> int:
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/work/learn2/sweep")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/work/learn2/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_tag(sweep_dir / "snake_curve.json", out_dir / "snake_learning_curve.png", "snake-move-up-v1")
    plot_tag(sweep_dir / "tweet_curve.json", out_dir / "tweet_learning_curve.png", "tweet-sentiment-v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
