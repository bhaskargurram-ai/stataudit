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
``check_report(text, path, alpha)``
    Recompute the p-values of APA-style results (``t(28) = 2.20, p = .036``) in
    the given text, or in a local .md/.txt/.html/.pdf file. Executes nothing.
``lint_file(path)``
    Static checks of a local .py or .ipynb analysis (no execution).

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
    "check_report_tool",
    "lint_file_tool",
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
    if p.suffix.lower() not in (".py", ".ipynb"):
        raise ValueError("path must point to a .py file or an .ipynb notebook")
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    if p.stat().st_size > MAX_SCRIPT_BYTES:
        raise ValueError("script is too large")
    report = audit_script(p, alpha=alpha, timeout=SCRIPT_TIMEOUT)
    return report.to_markdown()


MAX_TEXT_CHARS = 2_000_000
REPORT_SUFFIXES = (".md", ".markdown", ".txt", ".html", ".htm", ".rst", ".pdf")


def _local_file(path: str, suffixes: tuple[str, ...]) -> Path:
    p = Path(os.path.expanduser(str(path))).resolve()
    if p.suffix.lower() not in suffixes:
        raise ValueError(f"path must end with one of {', '.join(suffixes)}")
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    if p.stat().st_size > MAX_SCRIPT_BYTES * 10:
        raise ValueError("file is too large")
    return p


def check_report_tool(text: str = "", path: str = "", alpha: float = 0.05) -> str:
    """Recompute reported p-values in ``text`` or in the file at ``path``; returns a Markdown report."""
    from .reportcheck import check_file, check_text

    alpha = _validate_alpha(alpha)
    if bool(text) == bool(path):
        raise ValueError("give exactly one of text or path")
    if text:
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text must be a string of at most {MAX_TEXT_CHARS} characters")
        return check_text(text, alpha=alpha, source="<text>").to_markdown()
    return check_file(_local_file(path, REPORT_SUFFIXES), alpha=alpha).to_markdown()


def lint_file_tool(path: str) -> str:
    """Statically lint a local .py or .ipynb analysis; returns a Markdown report. Executes nothing."""
    from .lint import lint_file

    return lint_file(_local_file(path, (".py", ".ipynb"))).to_markdown()


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
        """Run a local Python analysis script (or .ipynb notebook) under stataudit's tracer; return a Markdown audit report.

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

    @server.tool()
    def check_report(text: str = "", path: str = "", alpha: float = 0.05) -> str:
        """Recompute the p-values of APA-style results in a report and flag inconsistent ones (statcheck-style).

        Give either `text` (the report text) or `path` (a local .md/.txt/.html/.pdf file). Recognises
        t(df) = x, F(df1, df2) = x, r(df) = x, chi2(df) = x, z = x and Q(df) = x followed by p = / < / > or ns,
        allows for rounding, and flags inconsistencies, gross inconsistencies (significance flips at alpha)
        and impossible values. Nothing is executed.
        """
        return check_report_tool(text, path, alpha)

    @server.tool()
    def lint_file(path: str) -> str:
        """Statically check a local .py or .ipynb analysis for statistical smells (no execution, no data).

        Rules: uncorrected multiple tests, Student's t-test without a variance check, Pearson without a
        plot/outlier check, chi-square with small expected counts, test shopping, p < 0.05 selection across
        many tests, normality pre-testing. Each finding has a rule id, explanation and suggested fix.
        """
        return lint_file_tool(path)

    return server


def main() -> None:
    """Entry point for ``stataudit-mcp`` (stdio transport)."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
