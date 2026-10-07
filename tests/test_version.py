"""The version number has exactly one place it is written: `pyproject.toml`. Everything else reads it from there
through the installed package's own metadata, rather than repeating the number somewhere it can go stale."""

from __future__ import annotations

import re
from importlib.metadata import version as installed_version
from pathlib import Path

import prismyra


def _pyproject_version() -> str:
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match, "no version line found in pyproject.toml"
    return match.group(1)


def test_the_version_comes_from_installed_metadata_not_a_second_hardcoded_string():
    """`prismyra.__version__` used to be written by hand and once went stale after a release bumped
    `pyproject.toml` without it. Reading it from the installed distribution's own metadata means there is nothing
    left to forget: this checks the two never disagree, for whatever version is current, rather than pinning today's
    number."""
    assert prismyra.__version__ == installed_version("prismyra") == _pyproject_version()


def test_the_current_release_is_0_4_0():
    """The number this release shipped, checked directly rather than only through agreement with itself."""
    assert prismyra.__version__ == "0.4.0"
