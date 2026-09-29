"""Prismyra: read one context once, answer many typed questions about it.

    from prismyra import Prismyra, Boolean, Choice

    engine = Prismyra("Qwen/Qwen3.6-35B-A3B-FP8")
    result = engine.ask(
        context=open("policy.txt").read(),
        questions=[
            Boolean(id="returnable", prompt="Can an opened item be returned?"),
            Choice(id="who_pays", prompt="Who pays return shipping?", choices=["seller", "buyer"]),
        ],
    )
    result["returnable"].value          # False
    result["who_pays"].option           # "buyer"
    result.timing.total_ms

Prefill only: nothing here generates text. The scope, and the point below which a general serving engine is faster,
are stated in the README.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

from .engine import Batch, Context, Prismyra, Shelf
from .schedule import Batcher
from .schema import (
    SCORING,
    SCORING_VERSION,
    Answer,
    Boolean,
    Choice,
    PrismyraError,
    Question,
    QuestionError,
    Request,
    Result,
    Scale,
    Timing,
)
from .temperature import Temperature
from .thresholds import Thresholds

try:
    __version__ = _installed_version("prismyra")
except PackageNotFoundError:
    # Running from a source checkout with no distribution installed, editable or otherwise -- there is no metadata
    # to read. pyproject.toml is the one place a version number is written; inventing a second one here that could
    # fall out of step with it is the failure this replaced.
    __version__ = "0+unknown"

__all__ = [
    "SCORING",
    "SCORING_VERSION",
    "Answer",
    "Batch",
    "Batcher",
    "Boolean",
    "Choice",
    "Context",
    "Prismyra",
    "PrismyraError",
    "Question",
    "QuestionError",
    "Request",
    "Result",
    "Scale",
    "Shelf",
    "Temperature",
    "Thresholds",
    "Timing",
    "__version__",
]
