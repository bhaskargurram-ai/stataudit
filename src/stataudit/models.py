"""Core data structures shared by the tracer, the auditors and the reporters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "Severity",
    "Status",
    "Role",
    "Finding",
    "Alternative",
    "CallSite",
    "CallRecord",
    "AuditResult",
    "Report",
    "STATUS_RANK",
    "SEVERITY_RANK",
]


class Severity(str, Enum):
    """How serious a finding is."""

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


SEVERITY_RANK = {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}


class Status(str, Enum):
    """Whether the reported conclusion survives the audit."""

    HOLDS = "holds"
    FRAGILE = "fragile"
    FAILS = "does_not_hold"
    NOT_AUDITED = "not_audited"

    @property
    def label(self) -> str:
        return {
            Status.HOLDS: "holds",
            Status.FRAGILE: "fragile",
            Status.FAILS: "does not hold",
            Status.NOT_AUDITED: "not audited",
        }[self]


STATUS_RANK = {Status.NOT_AUDITED: 0, Status.HOLDS: 1, Status.FRAGILE: 2, Status.FAILS: 3}


class Role(str, Enum):
    """Why an alternative analysis was computed.

    * ``RECOMMENDED`` - an assumption check failed and this analysis is the
      appropriate replacement. If every recommended alternative disagrees with
      the reported conclusion, the conclusion does not hold.
    * ``SENSITIVITY`` - a robustness probe (e.g. leave-one-out). Disagreement
      makes the conclusion fragile.
    * ``MULTIPLICITY`` - multiple-comparison adjusted p-value (Holm).
    """

    RECOMMENDED = "recommended"
    SENSITIVITY = "sensitivity"
    MULTIPLICITY = "multiplicity"


def _clean(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass
class Finding:
    """One observation about the data or the analysis."""

    code: str
    severity: Severity
    title: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    #: A short clause used in the one-line verdict, e.g. "treated has 3 outliers".
    short: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "title": self.title,
            "short": self.short,
            "detail": self.detail,
            "evidence": _clean(self.evidence),
        }


@dataclass
class Alternative:
    """An alternative analysis actually computed on the same data."""

    name: str
    pvalue: float | None
    role: Role
    reason: str = ""
    statistic: float | None = None
    significant: bool | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pvalue": _clean(self.pvalue),
            "statistic": _clean(self.statistic),
            "significant": self.significant,
            "role": self.role.value,
            "reason": self.reason,
            "note": self.note,
            "extra": _clean(self.extra),
        }


@dataclass
class CallSite:
    """Where in the user's code a statistical function was called."""

    filename: str
    lineno: int
    function: str = ""
    code: str = ""

    @property
    def short(self) -> str:
        import os

        base = os.path.basename(self.filename) or self.filename
        return f"{base}:{self.lineno}" if self.lineno > 0 else base

    def to_dict(self) -> dict[str, Any]:
        return {"filename": self.filename, "lineno": self.lineno, "function": self.function, "code": self.code}


@dataclass
class CallRecord:
    """A single traced call to a statistical function.

    ``inputs`` maps parameter names to numpy arrays (or lists of arrays for
    var-positional parameters such as ``*samples``); ``params`` holds scalar
    keyword arguments; ``result`` holds numeric fields of the returned object.
    """

    index: int
    func: str
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    site: CallSite | None = None

    def to_dict(self) -> dict[str, Any]:
        sizes: dict[str, Any] = {}
        for k, v in self.inputs.items():
            if isinstance(v, list):
                sizes[k] = [list(np.shape(a)) for a in v]
            else:
                sizes[k] = list(np.shape(v))
        return {
            "index": self.index,
            "func": self.func,
            "input_shapes": sizes,
            "params": _clean(self.params),
            "result": _clean(self.result),
            "site": self.site.to_dict() if self.site else None,
        }


@dataclass
class AuditResult:
    """The audit of one reported p-value."""

    call: CallRecord
    test_label: str
    alpha: float
    reported_p: float | None
    reported_stat: float | None = None
    target: str = ""
    findings: list[Finding] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    effect_sizes: dict[str, float] = field(default_factory=dict)
    sample_sizes: dict[str, int] = field(default_factory=dict)
    status: Status = Status.NOT_AUDITED
    verdict: str = ""
    in_family: bool = True
    data_key: str = ""

    @property
    def significant(self) -> bool | None:
        if self.reported_p is None or not math.isfinite(self.reported_p):
            return None
        return bool(self.reported_p < self.alpha)

    @property
    def location(self) -> str:
        return self.call.site.short if self.call.site else "<interactive>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call.index,
            "func": self.call.func,
            "test": self.test_label,
            "target": self.target,
            "location": self.location,
            "code": self.call.site.code if self.call.site else "",
            "alpha": self.alpha,
            "reported_p": _clean(self.reported_p),
            "reported_statistic": _clean(self.reported_stat),
            "reported_significant": self.significant,
            "status": self.status.value,
            "verdict": self.verdict,
            "findings": [f.to_dict() for f in self.findings],
            "alternatives": [a.to_dict() for a in self.alternatives],
            "effect_sizes": _clean(self.effect_sizes),
            "sample_sizes": _clean(self.sample_sizes),
        }


@dataclass
class Report:
    """A full audit: per-test results plus script-level findings."""

    results: list[AuditResult]
    alpha: float
    status: Status
    script_findings: list[Finding] = field(default_factory=list)
    source: str | None = None
    script_error: str | None = None
    multiplicity: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from . import __version__

        return {
            "tool": "stataudit",
            "version": __version__,
            "schema": 1,
            "source": self.source,
            "alpha": self.alpha,
            "status": self.status.value,
            "counts": self.counts(),
            "script_error": self.script_error,
            "script_findings": [f.to_dict() for f in self.script_findings],
            "multiplicity": _clean(self.multiplicity),
            "results": [r.to_dict() for r in self.results],
            "meta": _clean(self.meta),
        }

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Status}
        for r in self.results:
            out[r.status.value] += 1
        return out

    # Rendering conveniences -------------------------------------------------
    def to_text(self, color: bool = False, ascii_only: bool | None = None, verbose: bool = False) -> str:
        from .reports.text import render_text

        return render_text(self, color=color, ascii_only=ascii_only, verbose=verbose)

    def to_markdown(self) -> str:
        from .reports.markdown import render_markdown

        return render_markdown(self)

    def to_html(self) -> str:
        from .reports.html import render_html

        return render_html(self)

    def to_json(self, indent: int = 2) -> str:
        from .reports.jsonout import render_json

        return render_json(self, indent=indent)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.to_text()

    def _repr_markdown_(self) -> str:  # pragma: no cover - notebook hook
        return self.to_markdown()
