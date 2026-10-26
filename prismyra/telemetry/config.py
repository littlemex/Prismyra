"""`prismyra.telemetry`'s one piece of configuration data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TelemetryConfig:
    """``otlp_endpoint`` is the one required address: an OTLP/HTTP collector (e.g. a address this project
    never hard-codes -- see docs/OBSERVABILITY.md for why no example here names a real one).
    ``resource_attributes`` rides alongside ``service_name`` on every span, metric and log record this
    produces, the standard OTel `Resource` mechanism for "which deployment produced this" without this
    project inventing its own.
    """

    service_name: str = "prismyra"
    otlp_endpoint: str = "http://localhost:4318"
    resource_attributes: Mapping[str, str] = field(default_factory=dict)
    prometheus_from_stats: bool = True
