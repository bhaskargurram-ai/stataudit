"""``stataudit lint``: static checks for statistical analysis code (no execution, no data).

The rules look at the code's syntax tree only, so they work on code that
cannot be run (missing data, CI without credentials) and finish in
milliseconds. They are heuristics in the spirit of stat-lint ("The Fault in
our Stats", ASE 2025): each one names a statistical obligation that the code
does not visibly meet. A finding is a prompt to check, not proof of an error;
``stataudit run`` checks the same things on the real data.

==========  =============================  ========================================================
Rule        Name                           Fires when
==========  =============================  ========================================================
SA101       uncorrected-multiple-tests     ≥ 3 test calls, or a test inside a loop/comprehension,
                                           and no multiple-comparison correction anywhere
SA102       student-t-unchecked-variances  ``ttest_ind`` without ``equal_var=False`` (or ``trim``)
                                           and no Levene/Bartlett/Fligner check in the file
SA103       pearson-unchecked              ``pearsonr`` with no scatter plot, outlier check or rank
                                           correlation anywhere in the file
SA104       chi2-small-expected            ``chi2_contingency`` on a literal table whose expected
                                           counts fail Cochran's rule (warn), or on a computed table
                                           whose expected counts are never looked at (info)
SA105       test-shopping                  the same data passed to ≥ 3 different tests
SA106       p-threshold-selection          ``p < 0.05``-style comparisons that select results among
                                           many tests (or inside a loop) without a correction
SA107       normality-pretest              a normality test's p-value decides which test runs
==========  =============================  ========================================================

Suppress a finding with a comment on the reported line:
``# stataudit: ignore`` or ``# stataudit: ignore[SA102, SA105]``.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = ["RULES", "Rule", "LintFinding", "LintReport", "lint_source", "lint_file", "lint_notebook"]


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    severity: str  # "warn" | "info"
    summary: str
    explanation: str
    fix: str


RULES: dict[str, Rule] = {
    r.id: r
    for r in (
        Rule(
            "SA101", "uncorrected-multiple-tests", "warn",
            "Many hypothesis tests and no multiple-comparison correction",
            "Every test run at α = 0.05 has a 5% false-positive rate; with m independent null tests the chance of "
            "at least one 'significant' result is 1 - 0.95^m (40% for 10 tests). Results selected from many tests "
            "need a family-wise (Holm, Bonferroni) or false-discovery-rate (Benjamini-Hochberg) correction.",
            "Collect the p-values and adjust them, e.g. "
            "`statsmodels.stats.multitest.multipletests(pvals, method='holm')` or "
            "`scipy.stats.false_discovery_control(pvals)`, and report the adjusted values.",
        ),
        Rule(
            "SA102", "student-t-unchecked-variances", "warn",
            "Student's t-test (equal variances assumed) without a variance check",
            "`scipy.stats.ttest_ind` pools the two variances by default (Student's t-test). When the variances "
            "differ, especially with unequal group sizes, its p-value is wrong in either direction. Welch's t-test "
            "costs almost nothing when the variances are equal.",
            "Pass `equal_var=False` (Welch's t-test), or check the variances first (e.g. `scipy.stats.levene`) "
            "and report the check.",
        ),
        Rule(
            "SA103", "pearson-unchecked", "warn",
            "Pearson correlation without a plot or outlier check",
            "A Pearson correlation can be created or destroyed by one or two extreme points, and it only measures "
            "linear association. Nothing in this file looks at the scatter plot, checks for outliers or "
            "compares with a rank correlation.",
            "Plot the data (e.g. `plt.scatter(x, y)`), check for influential points, and report Spearman's rho "
            "(`scipy.stats.spearmanr`) alongside, or instead.",
        ),
        Rule(
            "SA104", "chi2-small-expected", "warn",
            "Chi-square test on a table with small expected counts",
            "The chi-square p-value is a large-sample approximation. Cochran's rule: no expected count below 1 and "
            "no more than 20% of cells below 5. Sparse tables give unreliable p-values.",
            "Inspect the expected counts (the 4th value returned by `chi2_contingency`); for 2×2 tables use "
            "`scipy.stats.fisher_exact`, otherwise a permutation/Monte Carlo test or merge sparse categories.",
        ),
        Rule(
            "SA105", "test-shopping", "warn",
            "The same data are tested with several different tests",
            "Running several tests on the same data and reporting the one that 'works' is a form of p-hacking: "
            "the reported p-value no longer means what it says.",
            "Choose the test from the design and the data's properties before looking at results; report every "
            "test you ran, or pre-register the analysis.",
        ),
        Rule(
            "SA106", "p-threshold-selection", "warn",
            "Results are selected by p < threshold among many uncorrected tests",
            "Comparing raw p-values with 0.05 across many tests (or inside a loop) and keeping the 'significant' "
            "ones reports the false positives along with the real effects.",
            "Compare adjusted p-values (Holm or Benjamini-Hochberg) with the threshold, and report effect sizes "
            "with confidence intervals rather than a significant/non-significant split.",
        ),
        Rule(
            "SA107", "normality-pretest", "info",
            "A normality test decides which test is run",
            "Choosing between a parametric and a non-parametric test based on a normality test's p-value "
            "(conditional testing) distorts the error rates of the final test, and normality tests have little "
            "power in small samples and too much in large ones.",
            "Decide on the test from the design and the scale of the data, or run the robust test (Welch, "
            "Mann-Whitney, permutation) directly; use plots rather than p-values to judge normality.",
        ),
    )
}

#: Hypothesis tests whose p-values count towards multiplicity.
TEST_FUNCS = {
    "ttest_ind", "ttest_rel", "ttest_1samp", "ttest_ind_from_stats", "pearsonr", "spearmanr", "kendalltau",
    "pointbiserialr", "f_oneway", "chi2_contingency", "chisquare", "mannwhitneyu", "wilcoxon", "kruskal",
    "linregress", "fisher_exact", "ks_2samp", "kstest", "anderson_ksamp", "binomtest", "binom_test",
    "alexandergovern", "brunnermunzel", "friedmanchisquare", "mood", "median_test", "barnard_exact",
    "boschloo_exact", "permutation_test", "proportions_ztest", "ttost_ind",
}
NORMALITY_FUNCS = {"shapiro", "normaltest", "jarque_bera", "anderson", "lilliefors", "kstest_normal"}
CORRECTION_NAMES = {
    "multipletests", "fdrcorrection", "fdrcorrection_twostage", "false_discovery_control", "bonferroni", "holm",
    "p_adjust", "padjust", "pairwise_tukeyhsd", "tukey_hsd", "multipletest", "MultiComparison", "allpairtest",
    "dunnett", "pairwise_ttests", "pairwise_tests",
}
VARIANCE_CHECKS = {"levene", "bartlett", "fligner"}
SCATTER_OR_ROBUST = {
    "scatter", "scatterplot", "regplot", "lmplot", "jointplot", "pairplot", "relplot", "residplot", "boxplot",
    "zscore", "iqr", "spearmanr", "kendalltau", "get_influence", "cooks_distance", "outlier_test",
    "influence_plot", "plot_regress_exog", "quantile", "percentile", "mad", "median_abs_deviation",
}
_IGNORE_RE = re.compile(r"#\s*stataudit:\s*ignore(?:\[([A-Z0-9,\s]+)\])?", re.I)


@dataclass
class LintFinding:
    rule: str
    line: int
    column: int
    message: str
    cell: int | None = None
    code: str = ""
    #: Overrides the rule's severity (e.g. SA104 on a computed table is only informational).
    level: str | None = None

    @property
    def info(self) -> Rule:
        return RULES[self.rule]

    @property
    def severity(self) -> str:
        return self.level or self.info.severity

    def location(self, filename: str = "") -> str:
        base = Path(filename).name if filename else ""
        if self.cell is not None:
            return f"{base} cell {self.cell}, line {self.line}".strip()
        return f"{base}:{self.line}" if base else f"line {self.line}"

    def to_dict(self) -> dict[str, Any]:
        r = self.info
        return {
            "rule": self.rule,
            "name": r.name,
            "severity": self.severity,
            "line": self.line,
            "column": self.column,
            "cell": self.cell,
            "message": self.message,
            "code": self.code,
            "explanation": r.explanation,
            "fix": r.fix,
        }


@dataclass
class LintReport:
    source: str | None
    findings: list[LintFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tests_found: int = 0

    @property
    def status(self) -> str:
        if any(f.severity == "warn" for f in self.findings):
            return "warn"
        return "info" if self.findings else "clean"

    def counts(self) -> dict[str, int]:
        return {
            "warn": sum(f.severity == "warn" for f in self.findings),
            "info": sum(f.severity == "info" for f in self.findings),
        }

    def to_dict(self) -> dict[str, Any]:
        from . import __version__

        return {
            "tool": "stataudit",
            "kind": "lint",
            "version": __version__,
            "schema": 1,
            "source": self.source,
            "status": self.status,
            "counts": self.counts(),
            "tests_found": self.tests_found,
            "notes": self.notes,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self, verbose: bool = False) -> str:
        c = self.counts()
        lines = [f"stataudit lint — {self.source or '<code>'}", "─" * 72]
        if not self.findings:
            lines.append(f"✅ No findings ({self.tests_found} statistical test call(s) found)")
        else:
            lines.append(f"{'⚠️' if c['warn'] else 'ℹ️'} {c['warn']} warning(s), {c['info']} note(s) "
                         f"({self.tests_found} statistical test call(s) found)")
        lines.append("")
        for f in self.findings:
            lines.append(f"{f.location(self.source or '')}: {f.rule} {f.info.name} [{f.severity}]")
            lines.append(f"    {f.message}")
            if f.code:
                lines.append(f"    > {f.code}")
            if verbose:
                lines.append(f"    why: {f.info.explanation}")
            lines.append(f"    fix: {f.info.fix}")
            lines.append("")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines).rstrip() + "\n"

    def to_markdown(self) -> str:
        out = ["# stataudit lint", ""]
        if self.source:
            out.append(f"**Source:** `{self.source}`  ")
        c = self.counts()
        out.append(f"**Findings:** {c['warn']} warning(s), {c['info']} note(s); "
                   f"{self.tests_found} statistical test call(s) found")
        out.append("")
        if self.findings:
            out.append("| Where | Rule | Finding | Suggested fix |")
            out.append("|---|---|---|---|")
            for f in self.findings:
                out.append(
                    f"| `{f.location(self.source or '')}` | {f.rule} `{f.info.name}` | {_md(f.message)} | {_md(f.info.fix)} |"
                )
            out.append("")
        for n in self.notes:
            out.append(f"- note: {_md(n)}")
        return "\n".join(out) + "\n"


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


# --------------------------------------------------------------------------- analysis


def _callee(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _kw(node: ast.Call, name: str) -> ast.expr | None:
    for k in node.keywords:
        if k.arg == name:
            return k.value
    return None


def _const(node: ast.expr | None) -> Any:
    try:
        return ast.literal_eval(node) if node is not None else None
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NOT_CONST


_NOT_CONST = object()


def _literal_table(node: ast.expr) -> list[list[float]] | None:
    """A 2-D table of numbers written literally (list of lists, possibly wrapped in np.array(...))."""
    if isinstance(node, ast.Call) and _callee(node) in ("array", "asarray") and node.args:
        node = node.args[0]
    val = _const(node)
    if val is _NOT_CONST or not isinstance(val, (list, tuple)) or len(val) < 2:
        return None
    rows = []
    for row in val:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            return None
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in row):
            return None
        rows.append([float(x) for x in row])
    if len({len(r) for r in rows}) != 1:
        return None
    return rows


def _cochran(table: list[list[float]]) -> tuple[bool, float, float]:
    """(violated, share of expected < 5, min expected)."""
    total = sum(sum(r) for r in table)
    if total <= 0:
        return False, 0.0, 0.0
    rs = [sum(r) for r in table]
    cs = [sum(c) for c in zip(*table)]
    exp = [ri * cj / total for ri in rs for cj in cs]
    share = sum(e < 5 for e in exp) / len(exp)
    mn = min(exp)
    return (mn < 1 or share > 0.2), share, mn


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loop_depth = 0
        self.tests: list[tuple[ast.Call, str, bool]] = []  # (call, name, in_loop)
        self.calls: list[ast.Call] = []
        self.names: set[str] = set()
        self.p_compares: list[tuple[ast.Compare, bool]] = []
        self.ifs: list[ast.If] = []
        self.manual_bonferroni = False
        self.defined_funcs: set[str] = set()
        self.subscript_3 = False  # result[3] / expected_freq used

    def _loop(self, node: ast.AST) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_For = visit_AsyncFor = visit_While = _loop
    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _loop

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined_funcs.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee(node)
        self.calls.append(node)
        if name in TEST_FUNCS:
            self.tests.append((node, name, self.loop_depth > 0))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.names.add(node.attr)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        self.names.add(node.name.rsplit(".", 1)[-1])
        if node.asname:
            self.names.add(node.asname)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            self.visit_alias(a)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div):
            left = node.left
            lv = _const(left)
            if (isinstance(lv, (int, float)) and not isinstance(lv, bool) and 0 < lv <= 0.1) or (
                isinstance(left, ast.Name) and "alpha" in left.id.lower()
            ):
                self.manual_bonferroni = True
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _const(node.slice) == 3:
            self.subscript_3 = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            left, right = node.left, node.comparators[0]
            if isinstance(node.ops[0], (ast.Gt, ast.GtE)):
                left, right = right, left
            if _is_pvalue_expr(left) and _is_threshold(right):
                self.p_compares.append((node, self.loop_depth > 0))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.ifs.append(node)
        self.generic_visit(node)


def _is_pvalue_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr in ("pvalue", "p_value", "pval")
    if isinstance(node, ast.Name):
        n = node.id.lower()
        return n in ("p", "pv", "pval", "pvalue", "p_value", "p_val") or n.startswith(("p_", "pval")) or n.endswith(("_p", "_pval", "_pvalue"))
    if isinstance(node, ast.Subscript):
        return _is_pvalue_expr(node.value) if isinstance(node.value, (ast.Name, ast.Attribute)) else False
    return False


def _is_threshold(node: ast.expr) -> bool:
    v = _const(node)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return 0 < v <= 0.1
    return isinstance(node, ast.Name) and node.id.lower() in ("alpha", "sig", "sig_level", "threshold", "level")


def _data_key(node: ast.Call) -> tuple[str, ...] | None:
    args = [a for a in node.args if not isinstance(a, ast.Starred)]
    if not args:
        return None
    parts = []
    for a in args[:2]:
        v = _const(a)
        if v is not _NOT_CONST:  # literal constants are not "the same data"
            return None
        parts.append(ast.unparse(a))
    return tuple(sorted(parts))


def _src_line(lines: list[str], lineno: int) -> str:
    return lines[lineno - 1].strip()[:160] if 0 < lineno <= len(lines) else ""


def analyze(tree: ast.Module, lines: list[str]) -> tuple[list[LintFinding], int]:
    """Run every rule over a parsed module. ``lines`` are the source lines (for snippets/suppressions)."""
    v = _Visitor()
    v.visit(tree)
    tests = [(c, n, loop) for c, n, loop in v.tests if n not in v.defined_funcs]
    findings: list[LintFinding] = []
    corrected = v.manual_bonferroni or bool(v.names & CORRECTION_NAMES) or any(
        k in n.lower() for n in v.names for k in ("bonferroni", "holm", "fdr")
    )

    def add(rule: str, node: ast.AST, msg: str) -> None:
        line = getattr(node, "lineno", 1)
        findings.append(LintFinding(rule, line, getattr(node, "col_offset", 0) + 1, msg, code=_src_line(lines, line)))

    n_tests = len(tests)
    in_loop = [t for t in tests if t[2]]
    many = n_tests >= 3 or bool(in_loop)
    # SA101
    if many and not corrected:
        anchor = in_loop[0][0] if in_loop else tests[2][0]
        how = f"{n_tests} test call(s)" + (f", {len(in_loop)} inside a loop or comprehension" if in_loop else "")
        add("SA101", anchor, f"{how}; no multiple-comparison correction (Holm, Bonferroni, FDR) in this file.")
    # SA102
    if not (v.names & VARIANCE_CHECKS):
        for call, name, _ in tests:
            if name != "ttest_ind":
                continue
            ev = _kw(call, "equal_var")
            trim = _kw(call, "trim")
            if trim is not None and _const(trim) not in (0, None):
                continue
            val = _const(ev) if ev is not None else True
            if val is True:
                add("SA102", call, "`ttest_ind` runs Student's t-test (equal variances assumed) and the variances "
                                   "are never checked; pass equal_var=False for Welch's t-test.")
    # SA103
    if not (v.names & SCATTER_OR_ROBUST) and not any("outlier" in n.lower() for n in v.names):
        for call, name, _ in tests:
            if name == "pearsonr":
                add("SA103", call, "Pearson correlation with no scatter plot, outlier check or rank correlation "
                                   "in this file.")
                break
    # SA104
    assigned: dict[str, list[ast.expr]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigned.setdefault(node.targets[0].id, []).append(node.value)
    for call, name, _ in tests:
        if name != "chi2_contingency" or not call.args:
            continue
        arg = call.args[0]
        if isinstance(arg, ast.Name) and len(assigned.get(arg.id, [])) == 1:
            arg = assigned[arg.id][0]  # a table written literally once and then passed by name
        table = _literal_table(arg)
        if table is not None:
            bad, share, mn = _cochran(table)
            if bad:
                add("SA104", call, f"Expected counts: {share:.0%} of cells below 5, smallest {mn:.2f} "
                                   "(Cochran's rule: none below 1, at most 20% below 5).")
        elif not (v.subscript_3 or "expected_freq" in v.names or "fisher_exact" in v.names or _unpacks_expected(tree, call)):
            findings.append(LintFinding("SA104", call.lineno, call.col_offset + 1,
                                        "chi2_contingency on a computed table; its expected counts are never "
                                        "inspected (check Cochran's rule before trusting the p-value).",
                                        code=_src_line(lines, call.lineno), level="info"))
    # SA105
    by_data: dict[tuple[str, ...], list[tuple[ast.Call, str]]] = {}
    for call, name, _ in tests:
        key = _data_key(call)
        if key:
            by_data.setdefault(key, []).append((call, name))
    for key, calls in by_data.items():
        funcs = sorted({n for _, n in calls})
        if len(funcs) >= 3:
            add("SA105", calls[0][0], f"`{', '.join(key)}` is tested with {len(funcs)} different tests: "
                                      f"{', '.join(funcs)}.")
    # SA106
    if many and not corrected and v.p_compares:
        node, _ = next(((n, l) for n, l in v.p_compares if l), v.p_compares[0])
        add("SA106", node, f"p-values are compared with a fixed threshold ({ast.unparse(node)}) across "
                           f"{n_tests} uncorrected test call(s){' in a loop' if in_loop else ''}.")
    # SA107
    for node in v.ifs:
        test_src = ast.unparse(node.test)
        uses_norm = any(fn in test_src for fn in NORMALITY_FUNCS) or _assigned_from_normality(tree, node.test)
        if not uses_norm:
            continue
        body_tests = {_callee(c) for b in (node.body + node.orelse) for c in ast.walk(b) if isinstance(c, ast.Call)}
        if body_tests & TEST_FUNCS:
            add("SA107", node, "The result of a normality test chooses between "
                               f"{', '.join(sorted(body_tests & TEST_FUNCS))}.")
    return findings, n_tests


def _unpacks_expected(tree: ast.Module, call: ast.Call) -> bool:
    """True when ``chi2, p, dof, expected = chi2_contingency(...)`` and ``expected`` is used later."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.value is call and isinstance(node.targets[0], ast.Tuple):
            elts = node.targets[0].elts
            if len(elts) == 4 and isinstance(elts[3], ast.Name) and elts[3].id != "_":
                name = elts[3].id
                uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
                return bool(uses)
    return False


def _assigned_from_normality(tree: ast.Module, test: ast.expr) -> bool:
    names = {n.id for n in ast.walk(test) if isinstance(n, ast.Name)}
    if not names:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _callee(node.value) in NORMALITY_FUNCS:
            for t in node.targets:
                targets = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
                if targets & names:
                    return True
    return False


def _suppressed(f: LintFinding, lines: list[str]) -> bool:
    if not (0 < f.line <= len(lines)):
        return False
    m = _IGNORE_RE.search(lines[f.line - 1])
    if not m:
        return False
    if not m.group(1):
        return True
    ids = {x.strip().upper() for x in m.group(1).split(",")}
    return f.rule in ids


def _finish(findings: list[LintFinding], lines: list[str], select: Iterable[str] | None,
            ignore: Iterable[str] | None) -> list[LintFinding]:
    sel = {s.upper() for s in select} if select else None
    ign = {s.upper() for s in ignore} if ignore else set()
    out = []
    for f in findings:
        if (sel is not None and f.rule not in sel) or f.rule in ign or _suppressed(f, lines):
            continue
        out.append(f)
    out.sort(key=lambda f: (f.cell or 0, f.line, f.rule))
    return out


def lint_source(source: str, filename: str | None = None, select: Iterable[str] | None = None,
                ignore: Iterable[str] | None = None) -> LintReport:
    """Lint Python source text."""
    rep = LintReport(source=filename)
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=filename or "<code>")
    except SyntaxError as exc:
        rep.notes.append(f"could not parse: {exc.msg} (line {exc.lineno})")
        return rep
    findings, rep.tests_found = analyze(tree, lines)
    rep.findings = _finish(findings, lines, select, ignore)
    return rep


def lint_notebook(path: str | Path, select: Iterable[str] | None = None,
                  ignore: Iterable[str] | None = None) -> LintReport:
    """Lint a notebook's code cells as one program (imports in early cells count for later ones)."""
    from .notebook import code_cells, load_notebook, strip_magics

    rep = LintReport(source=str(path))
    nb = load_notebook(path)
    body: list[ast.stmt] = []
    line_map: list[tuple[int, int, str]] = []  # global line -> (cell, line, text)
    offset = 0
    for c in code_cells(nb):
        code, _notes = strip_magics(c.source, c.index)
        cell_lines = code.split("\n")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            rep.notes.append(f"cell {c.index}: could not parse ({exc.msg}, line {exc.lineno}); skipped")
            tree = None
        if tree is not None:
            ast.increment_lineno(tree, offset)
            body.extend(tree.body)
        for i, ln in enumerate(cell_lines, 1):
            line_map.append((c.index, i, ln))
        offset += len(cell_lines)
    module = ast.Module(body=body, type_ignores=[])
    all_lines = [t for _, _, t in line_map]
    findings, rep.tests_found = analyze(module, all_lines)
    findings = _finish(findings, all_lines, select, ignore)
    for f in findings:
        if 0 < f.line <= len(line_map):
            cell, line, _ = line_map[f.line - 1]
            f.cell, f.line = cell, line
    rep.findings = findings
    return rep


def lint_file(path: str | Path, select: Iterable[str] | None = None, ignore: Iterable[str] | None = None) -> LintReport:
    """Lint a ``.py`` file or an ``.ipynb`` notebook."""
    p = Path(path)
    if p.suffix.lower() == ".ipynb":
        return lint_notebook(p, select, ignore)
    return lint_source(p.read_text(encoding="utf-8", errors="replace"), filename=str(p), select=select, ignore=ignore)
