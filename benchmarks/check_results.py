#!/usr/bin/env python3
"""Validate every checked-in result file. Runs in CI without a GPU.

The README's tables are generated from these files, so a malformed one would publish a wrong number. This checks shape
and internal consistency -- that the steps add up, that the crossover is where the two curves actually cross -- not the
measurements themselves, which no machine without the GPU can reproduce.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = (
    "schema",
    "measured_on",
    "machine",
    "model",
    "context",
    "method",
    "whole_job",
    "cost_model",
    "context_pass_steps",
)


def check(path: Path) -> list[str]:
    problems: list[str] = []
    data = json.loads(path.read_text())
    for key in REQUIRED:
        if key not in data:
            problems.append(f"missing {key!r}")
    if problems:
        return problems

    # Each step is a paired run, so what is checked is that the pairs are ordered and that the chain does not jump:
    # one step's "after" and the next step's "before" should differ only by run-to-run variation. A large gap would mean
    # the steps came from different configurations and the chain does not describe one sequence of changes.
    steps = data["context_pass_steps"]
    previous_after = steps["baseline_ms"]
    for step in steps["steps"]:
        if step["after_ms"] >= step["before_ms"]:
            problems.append(f"step {step['change']!r} did not improve: {step['before_ms']} to {step['after_ms']}")
        drift = abs(step["before_ms"] - previous_after)
        if drift > 6.0:
            problems.append(
                f"step {step['change']!r} starts at {step['before_ms']} where the previous step ended at "
                f"{previous_after}; a gap of {drift:.1f} ms is too large to be run-to-run variation"
            )
        previous_after = step["after_ms"]
    if abs(previous_after - steps["final_ms"]) > 0.6:
        problems.append(f"the last step ends at {previous_after} but final_ms is {steps['final_ms']}")

    job = data["whole_job"]
    ours = {int(k): v for k, v in job["prismyra_ms"].items()}
    theirs = {int(k): v for k, v in job["vllm_ms"].items()}
    crossover = job["crossover_questions"]
    shared = sorted(set(ours) & set(theirs))
    below = [n for n in shared if n < crossover and theirs[n] < ours[n]]
    above = [n for n in shared if n >= crossover and theirs[n] > ours[n]]
    if not below:
        problems.append(f"crossover is {crossover} but no measured point below it favours the baseline")
    if not above:
        problems.append(f"crossover is {crossover} but no measured point at or above it favours prismyra")
    for n in shared:
        if n < crossover and theirs[n] > ours[n]:
            problems.append(f"{n} questions is below the stated crossover yet prismyra already wins")

    model = data["cost_model"]
    group = model["questions_per_group"]
    for n, measured in sorted(ours.items()):
        predicted = model["context_ms"] + model["per_group_ms"] * -(-n // group)
        if abs(predicted - measured) > max(12.0, measured * 0.06):
            problems.append(f"the cost model predicts {predicted:.1f} ms at {n} questions, measured {measured}")
    return problems


def main() -> int:
    root = Path(__file__).parent / "results"
    files = sorted(root.glob("*.json"))
    if not files:
        print(f"no result files in {root}")
        return 1
    failed = 0
    for path in files:
        problems = check(path)
        if problems:
            failed += 1
            print(f"FAIL {path.name}")
            for p in problems:
                print(f"  {p}")
        else:
            print(f"ok   {path.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
