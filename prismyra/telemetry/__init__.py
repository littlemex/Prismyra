"""General observability for Prismyra's own serving process -- metrics, traces and (through an operator's
own collector configuration) logs about *the server*, never about a request's own content. This is a
separate concern from `prismyra.learn`, on purpose:

* **This package** answers "is the server healthy, how loud is it, where did the time go" -- request counts,
  latency (p50/p99 come from the exported histogram, computed by whatever reads the OTLP stream, not by this
  package), queue depth, batcher state, and a span per request with ``queue``/``context``/``readout`` events.
  Attributes on every signal here are shape, not content: counts, durations, token/question counts, `kind`,
  version strings -- never a document's text, a question's prompt, an option's label, or a probability.
* **`prismyra.learn`** answers "what did a *registered tag's* request ask, and what did Prismyra (or its
  student) answer" -- and only when `--learn-spec` is on, and only for a request that matched a registered
  tag. `prismyra.learn.otel_sink.OtelExperienceSink` can reuse this package's OTel SDK *bootstrap* (so a
    server exporting both does not configure two separate OTLP connections for no reason), but never its
  *data*: a different OTel `Logger` name (`"prismyra.learn.experience"` vs this package's own
  `"prismyra.telemetry"`), a different enable flag (`--learn-otel-endpoint` vs `--otel`), and no code path
  that could cross from one into the other. Enabling this package alone emits zero `Experience` rows, ever;
  enabling `prismyra.learn` alone emits zero spans or metrics through this package. Both are checked in
  `tests/test_telemetry_isolation.py`.

**Off by default, and the default costs nothing.** `None` is what `prismyra.server.create_app`'s own
``telemetry`` variable stays at unless ``--otel`` (or ``otel=True``) was passed -- the same
"`learn_hook is None` short-circuits every site that would otherwise touch it" discipline
DISTILL-RL-DESIGN-v2.md section 8 already established for `prismyra.learn`, applied here to this package.
Nothing under `prismyra.telemetry.otel_backend` is imported, and the ``otel`` extra
(``pip install "prismyra[otel]"``) is not required, unless an operator actually turns this on.

``build_telemetry`` is the one factory this module exports; `prismyra.telemetry.otel_backend.OtelTelemetry`
is its only real implementation today. `/metrics` (`prismyra.telemetry.prometheus.render_prometheus`) does
not go through any of this -- it is a zero-dependency reformatting of `/stats`, available whether or not
this package's own OTel path is ever turned on, which is the other place docs/OBSERVABILITY.md explains the
two are independent choices.
"""

from __future__ import annotations

from .config import TelemetryConfig


def build_telemetry(cfg: TelemetryConfig):
    """The one place `prismyra.telemetry.otel_backend` is imported."""
    from .otel_backend import OtelTelemetry

    return OtelTelemetry(cfg)


__all__ = ["TelemetryConfig", "build_telemetry"]
