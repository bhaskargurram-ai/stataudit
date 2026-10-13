"""MCP server exposing stataudit to MCP-capable assistants.

Tools
-----
``audit_script(path, alpha)``
    Run a local Python analysis script under the tracer (subprocess, timeout,
    no shell) and return the Markdown report.
``audit_test(test_name, data_json, alpha)``
    Audit a test on arrays supplied as JSON, e.g.
    ``{"a": [...], "b": [...]}`` for ``ttest_ind`` or
    ``{"observed": [[...], [...]]}`` for ``chi2_contingency``.

Inputs are treated purely as data. Array sizes are capped. The tool functions
are plain Python functions (``audit_script_tool`` / ``audit_test_tool``) so they
can be tested without an MCP client.

Run with ``stataudit-mcp`` (stdio transport).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .api import audit_call, audit_script
from .trace import DATA_PARAMS

__all__ = [
    "MAX_VALUES",
    "MAX_JSON_BYTES",
    "SCRIPT_TIMEOUT",
    "audit_script_tool",
    "audit_test_tool",
    "build_server",
    "main",
]

MAX_VALUES = 200_000
MAX_JSON_BYTES = 5_000_000
SCRIPT_TIMEOUT = 120.0
MAX_SCRIPT_BYTES = 2_000_000

#: Keys accepted in ``data_json`` per test, in positional order.
TEST_INPUTS: dict[str, tuple[str, ...]] = {
    "ttest_ind": ("a", "b"),
    "ttest_rel": ("a", "b"),
    "ttest_1samp": ("a",),
    "pearsonr": ("x", "y"),
    "spearmanr": ("a", "b"),
    "f_oneway": ("samples",),
    "kruskal": ("samples",),
    "chi2_contingency": ("observed",),
    "fisher_exact": ("table",),
    "mannwhitneyu": ("x", "y"),
    "wilcoxon": ("x", "y"),
    "linregress": ("x", "y"),
    "shapiro": ("x",),
}
_ALLOWED_KWARGS = {"equal_var", "alternative", "popmean", "correction", "nan_policy"}


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0 < alpha < 1) or not math.isfinite(alpha):
        raise ValueError("alpha must be between 0 and 1")
    return alpha


def _numeric_list(value: Any, name: str, depth: int = 1) -> list[Any]:
    """Validate a (nested) list of finite-or-null numbers and count values."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")

    def check(v: Any, d: int) -> int:
        if d == 0:
            if v is None:
                return 1
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must contain only numbers")
            return 1
        if not isinstance(v, list):
            raise ValueError(f"{name} must be a {depth}-level nested array")
        return sum(check(x, d - 1) for x in v)

    n = check(value, depth)
    if n > MAX_VALUES:
        raise ValueError(f"{name} has {n} values; the limit is {MAX_VALUES}")
    return value


def audit_test_tool(test_name: str, data_json: str, alpha: float = 0.05) -> str:
    """Audit ``test_name`` on arrays given as a JSON object; returns a Markdown report."""
    alpha = _validate_alpha(alpha)
    if test_name not in TEST_INPUTS:
        raise ValueError(f"unsupported test {test_name!r}; choose from {', '.join(sorted(TEST_INPUTS))}")
    if not isinstance(data_json, str) or len(data_json.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError(f"data_json must be a JSON string of at most {MAX_JSON_BYTES} bytes")
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"data_json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("data_json must be a JSON object")
    keys = TEST_INPUTS[test_name]
    args: list[Any] = []
    total = 0
    for k in keys:
        if k == "samples":
            groups = data.get("samples") or data.get("groups")
            if not isinstance(groups, list) or len(groups) < 2:
                raise ValueError(f"{test_name} needs 'samples': a list of at least two arrays")
            for i, g in enumerate(groups):
                _numeric_list(g, f"samples[{i}]")
                total += len(g)
            args.extend([[math.nan if v is None else v for v in g] for g in groups])
        elif k in ("observed", "table"):
            if k not in data:
                raise ValueError(f"{test_name} needs '{k}': a 2-D array of counts")
            _numeric_list(data[k], k, depth=2)
            args.append(data[k])
        else:
            if k not in data:
                if test_name == "wilcoxon" and k == "y":
                    continue
                raise ValueError(f"{test_name} needs '{k}'")
            _numeric_list(data[k], k)
            total += len(data[k])
            args.append([math.nan if v is None else v for v in data[k]])
    if total > MAX_VALUES:
        raise ValueError(f"too many values ({total}); the limit is {MAX_VALUES}")
    kwargs: dict[str, Any] = {}
    for k, v in (data.get("options") or {}).items():
        if k not in _ALLOWED_KWARGS:
            raise ValueError(f"option {k!r} not allowed; allowed: {', '.join(sorted(_ALLOWED_KWARGS))}")
        if not isinstance(v, (bool, int, float, str)):
            raise ValueError(f"option {k!r} must be a scalar")
        kwargs[k] = v
    if test_name == "ttest_1samp":
        args.append(float(kwargs.pop("popmean", data.get("popmean", 0.0))))
    report = audit_call(test_name, args, kwargs, alpha=alpha, label="<mcp>")
    return report.to_markdown()


def audit_script_tool(path: str, alpha: float = 0.05) -> str:
    """Run a local ``.py`` analysis script under the tracer and return the Markdown report."""
    alpha = _validate_alpha(alpha)
    p = Path(os.path.expanduser(str(path))).resolve()
    if p.suffix != ".py":
        raise ValueError("path must point to a .py file")
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    if p.stat().st_size > MAX_SCRIPT_BYTES:
        raise ValueError("script is too large")
    report = audit_script(p, alpha=alpha, timeout=SCRIPT_TIMEOUT)
    return report.to_markdown()


def build_server() -> Any:
    """Create the MCP server object (supports mcp>=2 ``MCPServer`` and mcp 1.x ``FastMCP``)."""
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
    except ImportError:  # pragma: no cover - mcp 1.x
        from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

    server = _Server(
        "stataudit",
        instructions=(
            "stataudit audits statistical analyses: it checks test assumptions on the actual data, "
            "recomputes robust alternatives and says whether each conclusion holds."
        ),
    )

    @server.tool()
    def audit_script(path: str, alpha: float = 0.05) -> str:
        """Run a local Python analysis script under stataudit's tracer and return a Markdown audit report.

        The script is executed in a subprocess with a timeout. Every scipy.stats test it performs is
        recorded and audited for assumption violations, outliers, multiplicity and robustness.
        """
        return audit_script_tool(path, alpha)

    @server.tool()
    def audit_test(test_name: str, data_json: str, alpha: float = 0.05) -> str:
        """Audit one statistical test on data supplied as JSON and return a Markdown report.

        test_name: one of ttest_ind, ttest_rel, ttest_1samp, pearsonr, spearmanr, f_oneway, kruskal,
        chi2_contingency, fisher_exact, mannwhitneyu, wilcoxon, linregress, shapiro.
        data_json: e.g. {"a": [..], "b": [..], "options": {"equal_var": true}} for ttest_ind,
        {"x": [..], "y": [..]} for pearsonr/linregress, {"samples": [[..], [..], [..]]} for f_oneway,
        {"observed": [[..], [..]]} for chi2_contingency, {"a": [..], "popmean": 0} for ttest_1samp.
        """
        return audit_test_tool(test_name, data_json, alpha)

    return server


def main() -> None:
    """Entry point for ``stataudit-mcp`` (stdio transport)."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
