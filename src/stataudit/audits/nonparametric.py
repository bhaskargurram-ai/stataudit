"""Info-level auditors for rank-based tests and normality tests."""

from __future__ import annotations

import numpy as np

from ..models import AuditResult, CallRecord, Finding, Severity
from .anova import groups_from_record
from .common import Unsupported, arg_labels, as_1d, data_key, new_result, small_sample_check

__all__ = ["audit_mannwhitneyu", "audit_wilcoxon", "audit_kruskal", "audit_spearmanr", "audit_shapiro"]


def _tie_fraction(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    _, counts = np.unique(x, return_counts=True)
    return float(counts[counts > 1].sum() / x.size)


def _ties_finding(frac: float, label: str) -> list[Finding]:
    if frac >= 0.2:
        return [
            Finding(
                "ties",
                Severity.INFO,
                f"{frac:.0%} of {label} are tied values",
                "Heavy ties make exact p-values unavailable and the normal approximation (with tie correction) is used; "
                "the data may be ordinal or coarsely rounded.",
                {"tie_fraction": frac},
            )
        ]
    return []


def audit_mannwhitneyu(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    x, _ = as_1d(rec.inputs.get("x"), "x")
    y, _ = as_1d(rec.inputs.get("y"), "y")
    lx, ly = arg_labels(rec, 2, ["x", "y"])
    res = new_result(rec, "Mann-Whitney U", alpha, target=f"{lx} vs {ly}")
    res.data_key = data_key([x, y])
    res.sample_sizes = {lx: int(x.size), ly: int(y.size)}
    if x.size == 0 or y.size == 0:
        raise Unsupported("empty sample")
    res.findings += small_sample_check(int(x.size), lx) + small_sample_check(int(y.size), ly)
    res.findings += _ties_finding(_tie_fraction(np.concatenate([x, y])), "the pooled values")
    u = rec.result.get("statistic")
    if isinstance(u, (int, float)):
        res.effect_sizes = {"rank_biserial": float(2 * u / (x.size * y.size) - 1)}
    vx, vy = np.var(x, ddof=1) if x.size > 1 else 0, np.var(y, ddof=1) if y.size > 1 else 0
    if min(vx, vy) > 0 and max(vx, vy) / min(vx, vy) > 4:
        res.findings.append(
            Finding(
                "shape_difference",
                Severity.INFO,
                "The groups have very different spreads",
                "Mann-Whitney then tests stochastic dominance, not a shift in medians; interpret it that way.",
            )
        )
    if x.size + y.size <= 16 and str(rec.params.get("method", "auto")) == "asymptotic":
        res.findings.append(
            Finding("asymptotic_small", Severity.INFO, "Asymptotic p-value with very small samples",
                    "Use method='exact' for samples this small.")
        )
    return [res]


def audit_wilcoxon(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    x = rec.inputs.get("x")
    y = rec.inputs.get("y")
    if x is None:
        raise Unsupported("no data")
    x = np.asarray(x, dtype=float).ravel()
    d = x - np.asarray(y, dtype=float).ravel() if y is not None else x
    d = d[np.isfinite(d)]
    res = new_result(rec, "Wilcoxon signed-rank", alpha, target="paired differences" if y is not None else "sample")
    res.data_key = data_key([x] + ([np.asarray(y, dtype=float)] if y is not None else []))
    res.sample_sizes = {"pairs": int(d.size)}
    zeros = int(np.sum(d == 0))
    res.findings += small_sample_check(int(d.size), "the paired sample")
    if zeros:
        res.findings.append(
            Finding("zero_differences", Severity.INFO, f"{zeros} zero difference(s)",
                    f"zero_method='{rec.params.get('zero_method', 'wilcox')}' decides how they are handled; "
                    "the default 'wilcox' discards them.", {"zeros": zeros})
        )
    res.findings += _ties_finding(_tie_fraction(np.abs(d[d != 0])), "the absolute differences")
    if d.size < 6 and res.reported_p is not None and res.reported_p >= alpha:
        res.findings.append(
            Finding("cannot_reach_alpha", Severity.INFO,
                    f"With n = {d.size} the smallest attainable two-sided p-value is {2 / 2 ** d.size:.3f}",
                    "A non-significant result here says nothing about the absence of an effect.")
        )
    return [res]


def audit_kruskal(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    groups, labels = groups_from_record(rec)
    res = new_result(rec, "Kruskal-Wallis H", alpha, target=f"{len(groups)} groups")
    res.data_key = data_key(groups)
    res.sample_sizes = {lab: int(g.size) for lab, g in zip(labels, groups)}
    for g, lab in zip(groups, labels):
        if g.size < 5:
            res.findings.append(
                Finding("chi2_approx_small", Severity.INFO, f"{lab} has fewer than 5 observations",
                        "The chi-square approximation for H is rough below 5 per group; a permutation test is safer.")
            )
    res.findings += _ties_finding(_tie_fraction(np.concatenate(groups)), "the pooled values")
    return [res]


def audit_spearmanr(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    a = rec.inputs.get("a")
    b = rec.inputs.get("b")
    if a is None or b is None:
        raise Unsupported("spearmanr with a single 2-D argument is not audited")
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise Unsupported("unequal lengths")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    la, lb = arg_labels(rec, 2, ["x", "y"])
    res = new_result(rec, "Spearman correlation", alpha, target=f"{la} and {lb}")
    res.data_key = data_key([a, b])
    res.sample_sizes = {"pairs": int(a.size)}
    res.findings += small_sample_check(int(a.size), "the sample")
    res.findings += _ties_finding(max(_tie_fraction(a), _tie_fraction(b)), "one variable's values")
    if a.size < 10:
        res.findings.append(
            Finding("asymptotic_small", Severity.INFO, "p-value uses a t approximation with n < 10",
                    "For very small samples a permutation test of rho is more accurate.")
        )
    if isinstance(rec.result.get("statistic"), (int, float)):
        res.effect_sizes = {"rho": float(rec.result["statistic"])}
    return [res]


def audit_shapiro(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    x, _ = as_1d(rec.inputs.get("x"), "x")
    res = new_result(rec, "Shapiro-Wilk normality test", alpha, target="normality")
    res.in_family = False
    res.data_key = data_key([x])
    res.sample_sizes = {"n": int(x.size)}
    if x.size < 20:
        res.findings.append(
            Finding("normality_low_power", Severity.INFO, f"Normality tests have little power at n = {x.size}",
                    "A non-significant Shapiro-Wilk does not show the data are normal.")
        )
    elif x.size > 1000:
        res.findings.append(
            Finding("normality_overpowered", Severity.INFO, f"With n = {x.size} trivial deviations are significant",
                    "Look at a Q-Q plot and the size of the skew rather than the p-value.")
        )
    res.findings.append(
        Finding("pretest_note", Severity.INFO, "Using a normality test to choose the next test",
                "Choosing between t-test and Mann-Whitney based on a pre-test changes the error rates of the "
                "second test; prefer deciding on robust methods up front.")
    )
    return [res]
