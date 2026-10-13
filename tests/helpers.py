"""Test helpers. Importing this module makes ``src/`` importable without installing."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from stataudit.models import CallRecord, CallSite  # noqa: E402
from stataudit.trace import build_record  # noqa: E402


def write_script(directory: str | os.PathLike[str], body: str, name: str = "analysis.py") -> Path:
    """Write a dedented script to ``directory/name`` and return its path."""
    path = Path(directory) / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def make_record(func: str, *args, result=None, code: str = "", **kwargs) -> CallRecord:
    """Build a CallRecord by actually calling the scipy function (unless ``result`` is given)."""
    from stataudit._scipy import orig

    if result is None:
        result = orig(func)(*args, **kwargs)
    site = CallSite(filename="test_script.py", lineno=1, code=code) if code else None
    return build_record(func, args, kwargs, result, site=site)


class TempDirMixin:
    """unittest mixin providing ``self.tmp`` (a fresh temporary directory)."""

    def setUp(self) -> None:  # noqa: D401
        self._tmp = tempfile.TemporaryDirectory(prefix="stataudit-test-")
        self.tmp = self._tmp.name
        super().setUp()  # type: ignore[misc]

    def tearDown(self) -> None:
        super().tearDown()  # type: ignore[misc]
        self._tmp.cleanup()


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)
