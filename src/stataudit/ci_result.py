"""The ``result.json`` document written by ``stataudit ci`` and its strict validator.

``result.json`` travels from the unprivileged job (where pull request code
runs) to the privileged comment job as an artifact, so the comment job treats
it as hostile: :func:`validate_result` checks every field's type, enum and
size and returns a sanitized copy built only from known fields. The card is
re-rendered from that copy; the ``card.md`` in the artifact is never posted.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "MAX_RESULT_BYTES",
    "OVERALL",
    "ResultValidationError",
    "validate_result",
    "new_result",
    "overall_status",
]

SCHEMA_ID = "stataudit.ci"
SCHEMA_VERSION = 1
MAX_RESULT_BYTES = 8_000_000
MAX_REPORT_BYTES = 400_000
OVERALL = ("pass", "warn", "fail")
LINT_STATUS = ("clean", "info", "warn", "error")
RUN_STATUS = ("holds", "fragile", "does_not_hold", "not_audited", "error")
RESULT_STATUS = ("holds", "fragile", "does_not_hold", "not_audited")
REPORT_STATUS = ("consistent", "inconsistent", "gross", "impossible", "error")
CHECK_STATUS = ("consistent", "inconsistent", "gross", "impossible")
FIX_STATUS = ("verified", "rejected", "no_submission", "nothing_to_fix", "proposed", "error")
AGENT_STATUS = ("verified", "rejected", "no_submission", "nothing_to_fix", "error", "skipped")
FAIL_ON = ("never", "warn", "fail")

_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RULE_RE = re.compile(r"^SA\d{3}$")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ResultValidationError(ValueError):
    pass


def _err(path: str, msg: str) -> ResultValidationError:
    return ResultValidationError(f"{path}: {msg}")


def _dict(v: Any, path: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise _err(path, "expected an object")
    return v


def _str(v: Any, path: str, limit: int, optional: bool = False) -> str | None:
    if v is None and optional:
        return None
    if not isinstance(v, str):
        raise _err(path, "expected a string")
    if len(v) > limit:
        raise _err(path, f"longer than {limit} characters")
    return _CTRL_RE.sub(" ", v)


def _int(v: Any, path: str, optional: bool = False, minimum: int = 0, maximum: int = 10**9) -> int | None:
    if v is None and optional:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or not minimum <= v <= maximum:
        raise _err(path, f"expected an integer in [{minimum}, {maximum}]")
    return v


def _num(v: Any, path: str) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise _err(path, "expected a finite number or null")
    return float(v)


def _enum(v: Any, allowed: tuple[str, ...], path: str, optional: bool = False) -> str | None:
    if v is None and optional:
        return None
    if v not in allowed:
        raise _err(path, f"expected one of {', '.join(allowed)}")
    return v


def _list(v: Any, path: str, item: Callable[[Any, str], Any], limit: int) -> list[Any]:
    if v is None:
        return []
    if not isinstance(v, list):
        raise _err(path, "expected a list")
    if len(v) > limit:
        raise _err(path, f"more than {limit} items")
    return [item(x, f"{path}[{i}]") for i, x in enumerate(v)]


def _sha(v: Any, path: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str) or not _SHA_RE.match(v):
        raise _err(path, "expected a hex commit SHA")
    return v


def _path(v: Any, path: str) -> str:
    s = _str(v, path, 300)
    assert s is not None
    if not s or s.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", s) or "\\" in s:
        raise _err(path, "expected a relative POSIX path")
    if any(part in ("..", "") for part in s.split("/")) and s != ".":
        raise _err(path, "path must not contain '..' or empty segments")
    return s


def _counts(v: Any, path: str, keys: tuple[str, ...]) -> dict[str, int]:
    d = _dict(v or {}, path)
    return {k: _int(d.get(k, 0), f"{path}.{k}") or 0 for k in keys}


def _lint_finding(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    rule = _str(d.get("rule"), f"{p}.rule", 10)
    if not rule or not _RULE_RE.match(rule):
        raise _err(f"{p}.rule", "expected a rule id like SA101")
    return {"rule": rule, "line": _int(d.get("line"), f"{p}.line"), "cell": _int(d.get("cell"), f"{p}.cell", optional=True),
            "severity": _enum(d.get("severity"), ("info", "warn"), f"{p}.severity"),
            "message": _str(d.get("message"), f"{p}.message", 500)}


def _lint(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    return {"path": _path(d.get("path"), f"{p}.path"), "status": _enum(d.get("status"), LINT_STATUS, f"{p}.status"),
            "error": _str(d.get("error"), f"{p}.error", 500, optional=True),
            "tests_found": _int(d.get("tests_found", 0), f"{p}.tests_found"),
            "findings": _list(d.get("findings"), f"{p}.findings", _lint_finding, 50)}


def _run_result(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    return {"location": _str(d.get("location"), f"{p}.location", 200), "test": _str(d.get("test"), f"{p}.test", 100),
            "target": _str(d.get("target", ""), f"{p}.target", 200),
            "reported_p": _num(d.get("reported_p"), f"{p}.reported_p"),
            "status": _enum(d.get("status"), RESULT_STATUS, f"{p}.status"),
            "verdict": _str(d.get("verdict", ""), f"{p}.verdict", 1500)}


def _opaque_report(v: Any, p: str) -> dict[str, Any] | None:
    if v is None:
        return None
    d = _dict(v, p)
    try:
        size = len(json.dumps(d, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise _err(p, f"not serializable: {exc}") from exc
    if size > MAX_REPORT_BYTES:
        raise _err(p, "report too large")
    return d


def _run(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    return {"path": _path(d.get("path"), f"{p}.path"), "status": _enum(d.get("status"), RUN_STATUS, f"{p}.status"),
            "error": _str(d.get("error"), f"{p}.error", 2000, optional=True),
            "counts": _counts(d.get("counts"), f"{p}.counts", RESULT_STATUS),
            "results": _list(d.get("results"), f"{p}.results", _run_result, 200),
            "report": _opaque_report(d.get("report"), f"{p}.report")}


def _check_result(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    return {"location": _str(d.get("location"), f"{p}.location", 60), "label": _str(d.get("label"), f"{p}.label", 200),
            "computed_p": _num(d.get("computed_p"), f"{p}.computed_p"),
            "status": _enum(d.get("status"), CHECK_STATUS, f"{p}.status"),
            "message": _str(d.get("message", ""), f"{p}.message", 500)}


def _report(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    return {"path": _path(d.get("path"), f"{p}.path"), "status": _enum(d.get("status"), REPORT_STATUS, f"{p}.status"),
            "error": _str(d.get("error"), f"{p}.error", 500, optional=True),
            "counts": _counts(d.get("counts"), f"{p}.counts", CHECK_STATUS),
            "results": _list(d.get("results"), f"{p}.results", _check_result, 200)}


def _fix_row(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    out: dict[str, Any] = {}
    for k in ("location", "before_test", "before_status", "after_location", "after_test", "after_status", "outcome"):
        out[k] = _str(d.get(k), f"{p}.{k}", 200, optional=True)
    return out


def _fix(v: Any, p: str) -> dict[str, Any]:
    d = _dict(v, p)
    return {"path": _path(d.get("path"), f"{p}.path"), "status": _enum(d.get("status"), FIX_STATUS, f"{p}.status"),
            "rationale": _str(d.get("rationale", ""), f"{p}.rationale", 2000),
            "diff": _str(d.get("diff", ""), f"{p}.diff", 20000),
            "before_status": _str(d.get("before_status", ""), f"{p}.before_status", 20),
            "after_status": _str(d.get("after_status", ""), f"{p}.after_status", 20),
            "rows": _list(d.get("rows"), f"{p}.rows", _fix_row, 50),
            "reasons": _list(d.get("reasons"), f"{p}.reasons", lambda x, q: _str(x, q, 500), 20),
            "model": _str(d.get("model", ""), f"{p}.model", 100)}


def _agent(v: Any, p: str) -> dict[str, Any] | None:
    if v is None:
        return None
    d = _dict(v, p)
    return {"status": _enum(d.get("status"), AGENT_STATUS, f"{p}.status"),
            "notes": _list(d.get("notes"), f"{p}.notes", lambda x, q: _str(x, q, 500), 20),
            "fixes": _list(d.get("fixes"), f"{p}.fixes", _fix, 10)}


def validate_result(data: Any) -> dict[str, Any]:
    """Validate a result document and return a sanitized copy. Raises :class:`ResultValidationError`."""
    d = _dict(data, "$")
    if d.get("schema") != SCHEMA_ID:
        raise _err("$.schema", f"expected {SCHEMA_ID!r}")
    if d.get("version") != SCHEMA_VERSION:
        raise _err("$.version", f"unsupported version (expected {SCHEMA_VERSION})")
    repo = _str(d.get("repo"), "$.repo", 200, optional=True)
    if repo is not None and not _SLUG_RE.match(repo):
        raise _err("$.repo", "invalid owner/repo")
    alpha = _num(d.get("alpha"), "$.alpha")
    if alpha is None or not 0 < alpha < 1:
        raise _err("$.alpha", "expected a number between 0 and 1")
    return {
        "schema": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "tool_version": _str(d.get("tool_version", ""), "$.tool_version", 40),
        "repo": repo,
        "pr_number": _int(d.get("pr_number"), "$.pr_number", optional=True, minimum=1),
        "head": _sha(d.get("head"), "$.head"),
        "base": _sha(d.get("base"), "$.base"),
        "alpha": alpha,
        "fail_on": _enum(d.get("fail_on", "never"), FAIL_ON, "$.fail_on"),
        "status": _enum(d.get("status"), OVERALL, "$.status"),
        "config_source": _str(d.get("config_source", ""), "$.config_source", 200),
        "lint": _list(d.get("lint"), "$.lint", _lint, 200),
        "runs": _list(d.get("runs"), "$.runs", _run, 100),
        "reports": _list(d.get("reports"), "$.reports", _report, 200),
        "agent": _agent(d.get("agent"), "$.agent"),
        "notes": _list(d.get("notes"), "$.notes", lambda x, q: _str(x, q, 500), 20),
    }


def overall_status(result: dict[str, Any]) -> str:
    """``fail`` if a conclusion does not hold or a reported result flips significance / is impossible;
    ``warn`` for fragile conclusions, inconsistent reported results, lint warnings and errors; else ``pass``."""
    level = 0
    for r in result.get("runs") or []:
        if r.get("status") == "does_not_hold":
            level = max(level, 2)
        elif r.get("status") in ("fragile", "error"):
            level = max(level, 1)
    for r in result.get("reports") or []:
        if r.get("status") in ("gross", "impossible"):
            level = max(level, 2)
        elif r.get("status") in ("inconsistent", "error"):
            level = max(level, 1)
    for r in result.get("lint") or []:
        if r.get("status") in ("warn", "error"):
            level = max(level, 1)
    return OVERALL[level]


def new_result(**fields: Any) -> dict[str, Any]:
    from . import __version__

    doc: dict[str, Any] = {"schema": SCHEMA_ID, "version": SCHEMA_VERSION, "tool_version": __version__}
    doc.update(fields)
    return doc
