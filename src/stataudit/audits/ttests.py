"""Auditors for Student / Welch / paired / one-sample t-tests."""

from __future__ import annotations

import math

import numpy as np

from .. import stats_extra as sx
from .._scipy import ostats
from ..models import AuditResult, CallRecord, Finding, Role, Severity
from .common import (
    SMALL_N,
    Unsupported,
    as_1d,
    arg_labels,
    data_key,
    fmt_num,
    fmt_p,
    make_alt,
    new_result,
    normality_check,
    outlier_check,
    small_sample_check,
)

__all__ = ["audit_ttest_ind", "audit_ttest_rel", "audit_ttest_1samp"]


def _nan_finding(dropped: int, label: str, nan_policy: str) -> list[Finding]:
    if not dropped:
        return []
    sev = Severity.CRITICAL if nan_policy == "propagate" else Severity.INFO
    return [
        Finding(
            code="missing_values",
            severity=sev,
            title=f"{label} contains {dropped} NaN value(s)",
            detail=(
                "With nan_policy='propagate' the reported statistic is NaN. The audit drops missing values."
                if nan_policy == "propagate"
                else "Missing values were dropped by the test; check they are missing at random."
            ),
            evidence={"n_missing": dropped},
            short=f"{label} has missing values",
        )
    ]


def audit_ttest_ind(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit ``scipy.stats.ttest_ind``.

    Checks per-group normality, outliers and equality of variances, then
    recomputes Welch's t-test, a 20% trimmed (Yuen) t-test and Mann-Whitney U
    where the checks say they are the appropriate analysis.
    """
    if int(rec.params.get("axis", 0) or 0) not in (0, -1) and np.ndim(rec.inputs.get("a")) > 1:
        raise Unsupported("multi-dimensional input with a non-default axis")
    a, na_a = as_1d(rec.inputs.get("a"), "a")
    b, na_b = as_1d(rec.inputs.get("b"), "b")
    la, lb = arg_labels(rec, 2, ["group A", "group B"])
    equal_var = bool(rec.params.get("equal_var", True))
    alternative = str(rec.params.get("alternative", "two-sided"))
    trim = float(rec.params.get("trim", 0) or 0)
    nan_policy = str(rec.params.get("nan_policy", "propagate"))
    label = ("Student's t-test" if equal_var else "Welch's t-test") if trim == 0 else "Yuen's trimmed t-test"
    res = new_result(rec, label, alpha, target=f"{la} vs {lb}")
    res.data_key = data_key([a, b])
    res.sample_sizes = {la: int(a.size), lb: int(b.size)}
    res.findings += _nan_finding(na_a, la, nan_policy) + _nan_finding(na_b, lb, nan_policy)
    if a.size < 2 or b.size < 2:
        raise Unsupported("each group needs at least two observations")

    res.findings += small_sample_check(int(a.size), la) + small_sample_check(int(b.size), lb)

    nonnormal_small = False
    for x, lab in ((a, la), (b, lb)):
        f, nonnormal, _ = normality_check(x, lab)
        res.findings += f
        if nonnormal and x.size < SMALL_N:
            nonnormal_small = True

    any_outliers = False
    for x, lab in ((a, la), (b, lb)):
        f, idx = outlier_check(x, lab)
        res.findings += f
        any_outliers = any_outliers or idx.size > 0

    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    ratio = max(va, vb) / min(va, vb) if min(va, vb) > 0 else math.inf
    lev_p = None
    try:
        lev_p = float(ostats.levene(a, b, center="median").pvalue)
    except Exception:
        pass
    n_ratio = max(a.size, b.size) / min(a.size, b.size)
    unequal = (lev_p is not None and lev_p < 0.05) or ratio >= 4 or (ratio >= 2 and n_ratio >= 1.5)
    if unequal:
        larger = la if va > vb else lb
        sev = Severity.WARN if equal_var and trim == 0 else Severity.INFO
        res.findings.append(
            Finding(
                code="unequal_variances",
                severity=sev,
                title=f"Variances differ {ratio:.1f}× (Levene p = {fmt_p(lev_p)})",
                detail=(
                    f"{larger} is more spread out. Student's t-test pools the variances and assumes they are equal; "
                    "when they are not (especially with unequal group sizes) its p-value is unreliable. "
                    "Welch's t-test does not make this assumption."
                    if equal_var
                    else "You already used Welch's t-test, which does not assume equal variances."
                ),
                evidence={"var_ratio": ratio, "levene_p": lev_p, "var": {la: va, lb: vb}, "n_ratio": n_ratio},
                short=f"variances differ {ratio:.0f}×" if ratio >= 10 else f"variances differ {ratio:.1f}×",
            )
        )

    res.effect_sizes = {
        "mean_difference": float(a.mean() - b.mean()),
        "cohens_d": sx.cohens_d(a, b),
        "hedges_g": sx.hedges_g(a, b),
    }

    if trim == 0:
        if equal_var and unequal:
            w = ostats.ttest_ind(a, b, equal_var=False, alternative=alternative)
            res.alternatives.append(
                make_alt(
                    "Welch's t-test",
                    float(w.pvalue),
                    alpha,
                    Role.RECOMMENDED,
                    reason="variances differ",
                    statistic=float(w.statistic),
                    df=float(getattr(w, "df", np.nan)),
                )
            )
        if any_outliers:
            y = ostats.ttest_ind(a, b, equal_var=False, trim=0.2, alternative=alternative)
            res.alternatives.append(
                make_alt(
                    "Yuen's 20% trimmed t-test",
                    float(y.pvalue),
                    alpha,
                    Role.RECOMMENDED,
                    reason="outliers present; compares trimmed means",
                    statistic=float(y.statistic),
                )
            )
        if nonnormal_small or any_outliers:
            mw = ostats.mannwhitneyu(a, b, alternative=alternative)
            rbc = sx.rank_biserial(float(mw.statistic), a.size, b.size)
            res.effect_sizes["rank_biserial"] = rbc
            res.alternatives.append(
                make_alt(
                    "Mann-Whitney U",
                    float(mw.pvalue),
                    alpha,
                    Role.RECOMMENDED,
                    reason="non-normal small samples" if nonnormal_small else "outliers present",
                    statistic=float(mw.statistic),
                    note=f"rank-biserial r = {rbc:.2f}; tests whether one group tends to have larger values, not means",
                    rank_biserial=rbc,
                )
            )
    return [res]


def _paired_like(
    rec: CallRecord,
    d: np.ndarray,
    alpha: float,
    label: str,
    target: str,
    dlabel: str,
    alternative: str,
    keys: list[np.ndarray],
) -> AuditResult:
    res = new_result(rec, label, alpha, target=target)
    res.data_key = data_key(keys)
    res.sample_sizes = {dlabel: int(d.size)}
    if d.size < 2:
        raise Unsupported("need at least two observations")
    res.findings += small_sample_check(int(d.size), dlabel)
    f, nonnormal, _ = normality_check(d, dlabel)
    res.findings += f
    of, idx = outlier_check(d, dlabel)
    res.findings += of
    sd = float(d.std(ddof=1))
    res.effect_sizes = {"mean": float(d.mean()), "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan")}
    if (nonnormal and d.size < SMALL_N) or idx.size > 0:
        if np.all(d == 0):
            return res
        w = ostats.wilcoxon(d, alternative=alternative)
        res.alternatives.append(
            make_alt(
                "Wilcoxon signed-rank",
                float(w.pvalue),
                alpha,
                Role.RECOMMENDED,
                reason="non-normal small sample" if nonnormal else "outliers present",
                statistic=float(w.statistic),
                note="tests whether the differences are symmetric about 0 (median shift), not the mean",
            )
        )
    return res


def audit_ttest_rel(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit ``scipy.stats.ttest_rel``: normality and outliers of the paired differences."""
    a_raw = np.asarray(rec.inputs.get("a"), dtype=float)
    b_raw = np.asarray(rec.inputs.get("b"), dtype=float)
    if a_raw.ndim != 1 or b_raw.ndim != 1:
        a_raw, b_raw = np.squeeze(a_raw), np.squeeze(b_raw)
        if a_raw.ndim != 1 or b_raw.ndim != 1:
            raise Unsupported("only 1-D paired samples are audited")
    if a_raw.shape != b_raw.shape:
        raise Unsupported("paired samples have different lengths")
    keep = np.isfinite(a_raw) & np.isfinite(b_raw)
    la, lb = arg_labels(rec, 2, ["before", "after"])
    d = a_raw[keep] - b_raw[keep]
    alternative = str(rec.params.get("alternative", "two-sided"))
    res = _paired_like(rec, d, alpha, "Paired t-test", f"{la} vs {lb}", "paired differences", alternative, [a_raw, b_raw])
    dropped = int((~keep).sum())
    res.findings += _nan_finding(dropped, "paired data", str(rec.params.get("nan_policy", "propagate")))
    return [res]


def audit_ttest_1samp(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit ``scipy.stats.ttest_1samp``: normality and outliers, Wilcoxon against ``popmean``."""
    x, dropped = as_1d(rec.inputs.get("a"), "a")
    pm = rec.params.get("popmean", rec.inputs.get("popmean", 0.0))
    try:
        popmean = float(np.asarray(pm, dtype=float).ravel()[0]) if np.size(pm) == 1 else None
    except Exception:
        popmean = None
    if popmean is None:
        raise Unsupported("non-scalar popmean")
    (lx,) = arg_labels(rec, 1, ["sample"])
    alternative = str(rec.params.get("alternative", "two-sided"))
    res = _paired_like(
        rec, x - popmean, alpha, "One-sample t-test", f"mean of {lx} vs {fmt_num(popmean)}", lx, alternative, [x]
    )
    res.findings += _nan_finding(dropped, lx, str(rec.params.get("nan_policy", "propagate")))
    return [res]
