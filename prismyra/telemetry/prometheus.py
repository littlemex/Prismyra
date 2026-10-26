"""`/metrics` in the Prometheus text exposition format, built directly from `/stats`'s own dict -- no
dependency, no OTel SDK, always available whether or not ``prismyra[otel]`` is installed or ``--otel`` was
ever passed. This is one of the two ways this project lets Prometheus see these numbers (the other is
pointing an OTel collector's own Prometheus receiver at the OTLP stream `prismyra.telemetry`'s OTel backend
emits, when that is enabled) -- see docs/OBSERVABILITY.md's own table for when each applies.

Every numeric leaf of `/stats`'s dict becomes one gauge, named ``prismyra_<dotted_path>`` with ``.``
replaced by ``_`` (Prometheus metric names cannot contain a dot). A boolean is reported as ``0``/``1``, the
Prometheus convention; a string, ``None``, or a list is not a number and is skipped, not coerced into one.
"""

from __future__ import annotations

import re

_SANITIZE = re.compile(r"[^a-zA-Z0-9_]")


def _metric_name(path: str) -> str:
    return "prismyra_" + _SANITIZE.sub("_", path)


def _walk(prefix: str, obj: object, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(f"{prefix}_{k}" if prefix else str(k), v, out)
    elif isinstance(obj, bool):
        out.append(f"{_metric_name(prefix)} {int(obj)}")
    elif isinstance(obj, (int, float)):
        out.append(f"{_metric_name(prefix)} {obj}")
    # str / None / list / anything else: not a number, not reported.


def render_prometheus(stats: dict) -> str:
    """One line per numeric leaf, Prometheus text format (no ``# TYPE``/``# HELP`` lines -- every metric
    here is a snapshot gauge, and Prometheus does not require either to scrape successfully)."""
    lines: list[str] = []
    _walk("", stats, lines)
    return "\n".join(lines) + "\n"
