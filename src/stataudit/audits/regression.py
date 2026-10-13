"""Auditors for Pearson correlation, simple linear regression and OLS."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats as _st

from .. import stats_extra as sx
from .._scipy import ostats
from ..models import AuditResult, CallRecord, Finding, Role, Severity
from .common import (
    SMALL_N,
    Unsupported,
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

__all__ = ["audit_pearsonr", "audit_linregress", "audit_ols_record", "loo_pearson_pvalues", "pearson_p"]


def pearson_p(r: np.ndarray | float, n: int, alternative: str = "two-sided") -> np.ndarray:
    """p-value of Pearson's r via the exact t transformation, df = n - 2."""
    r = np.clip(np.asarray(r, dtype=float), -1.0, 1.0)
    df = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / np.maximum(1e-300, 1 - r**2))
    if alternative == "less":
        return _st.t.cdf(t, df)
    if alternative == "greater":
        return _st.t.sf(t, df)
    return np.minimum(1.0, 2 * _st.t.sf(np.abs(t), df))


def loo_pearson_pvalues(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> tuple[np.ndarray, np.ndarray]:
    """Pearson r and p-value with each observation left out in turn (vectorised)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    m = n - 1
    sx_, sy_ = x.sum() - x, y.sum() - y
    sxx, syy, sxy = (x * x).sum() - x * x, (y * y).sum() - y * y, (x * y).sum() - x * y
    cov = sxy - sx_ * sy_ / m
    vx = sxx - sx_**2 / m
    vy = syy - sy_**2 / m
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(vx * vy)
    r = np.where(np.isfinite(r), r, 0.0)
    return r, pearson_p(r, m, alternative)


def _paired_xy(rec: CallRecord, xname: str = "x", yname: str = "y") -> tuple[np.ndarray, np.ndarray, int]:
    x = rec.inputs.get(xname)
    y = rec.inputs.get(yname)
    if x is None or y is None:
        raise Unsupported("x and y are both required")
    x = np.squeeze(np.asarray(x, dtype=float))
    y = np.squeeze(np.asarray(y, dtype=float))
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise Unsupported("x and y must be 1-D with equal length")
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep], int((~keep).sum())


def high_influence_set(h: np.ndarray, d: np.ndarray, k: int) -> np.ndarray:
    """Observations to drop in a group-deletion sensitivity check.

    Union of (a) high-leverage points, h > 3k/n, and (b) points with Cook's
    D > 4/n that also have h > 2k/n or D > 0.5. Single-deletion diagnostics
    miss *clusters* of high-leverage points (each masks the other), so
    leverage alone is used for (a). Returns an empty set when the rule would
    drop more than 20% of the data (then the predictor is simply skewed).
    """
    n = h.size
    cut = 4.0 / n
    hl = h > 3 * k / n
    severe = (d > cut) & ((h > 2 * k / n) | (d > 0.5))
    idx = np.flatnonzero(hl | severe)
    if idx.size > max(1, int(0.2 * n)):
        return np.array([], dtype=int)
    return idx


def _influence(
    x: np.ndarray, y: np.ndarray, res: AuditResult, alpha: float, alternative: str, reported_sig: bool | None
) -> tuple[bool, np.ndarray]:
    """Shared leverage / Cook's distance / leave-one-out analysis for one predictor.

    Returns ``(influence_changes_conclusion_or_estimate, dropped_indices)``.
    """
    n = x.size
    X = sx.add_constant(x)
    fit = sx.ols_fit(X, y)
    h = sx.hat_diagonal(X)
    d = sx.cooks_distance(fit)
    cut = 4.0 / n
    infl = np.flatnonzero(d > cut)
    top = int(np.argmax(d))
    drop = high_influence_set(h, d, 2)
    if infl.size:
        res.findings.append(
            Finding(
                code="influential_points",
                severity=Severity.INFO,
                title=f"{infl.size} point(s) with Cook's D > 4/n = {cut:.3f}",
                detail=(
                    f"Largest: index {top} (x = {fmt_num(float(x[top]))}, y = {fmt_num(float(y[top]))}, "
                    f"Cook's D = {d[top]:.2f}, leverage = {h[top]:.2f})."
                ),
                evidence={
                    "indices": infl.tolist()[:50],
                    "cooks_d": d[infl].tolist()[:50],
                    "max_cooks_d": float(d[top]),
                    "most_influential_index": top,
                },
            )
        )
    driven = False
    if drop.size and n - drop.size >= 4:
        keep = np.setdiff1d(np.arange(n), drop)
        if np.ptp(x[keep]) > 0 and np.ptp(y[keep]) > 0:
            pr = ostats.pearsonr(x[keep], y[keep], alternative=alternative)
            r_all = float(np.corrcoef(x, y)[0, 1])
            r_drop = float(pr.statistic)
            flips = reported_sig is not None and (float(pr.pvalue) < alpha) != reported_sig
            driven = flips or abs(r_all - r_drop) > 0.25
            word = "point" if drop.size == 1 else "points"
            res.findings.append(
                Finding(
                    code="high_leverage",
                    severity=Severity.WARN if driven else Severity.INFO,
                    title=f"{drop.size} high-leverage {word} (indices {', '.join(map(str, drop.tolist()[:8]))})",
                    detail=(
                        f"Leverage up to {h[drop].max():.2f} (mean leverage is {2 / n:.2f}). Without them "
                        f"r = {r_drop:.3f} (p = {fmt_p(float(pr.pvalue))}) instead of r = {r_all:.3f}. "
                        + (
                            "These points alone decide the result."
                            if flips
                            else ("They change the estimate substantially." if driven else "The result does not depend on them.")
                        )
                    ),
                    evidence={
                        "indices": drop.tolist()[:50],
                        "leverage": h[drop].tolist()[:50],
                        "r_without": r_drop,
                        "p_without": float(pr.pvalue),
                    },
                    short=(f"{drop.size} high-leverage {word} drive the result" if driven else ""),
                )
            )
            res.alternatives.append(
                make_alt(
                    f"Pearson without {drop.size} high-leverage {word}",
                    float(pr.pvalue),
                    alpha,
                    Role.SENSITIVITY,
                    reason="group deletion of high-leverage points",
                    statistic=r_drop,
                    note=f"r = {r_drop:.3f}",
                    dropped=drop.tolist(),
                )
            )
    # Leave-one-out sensitivity
    if n >= 5 and reported_sig is not None:
        r_loo, p_loo = loo_pearson_pvalues(x, y, alternative)
        flips_idx = np.flatnonzero((p_loo < alpha) != reported_sig)
        if flips_idx.size:
            j = int(flips_idx[np.argmax(np.abs(np.log(p_loo[flips_idx] + 1e-300) - math.log(alpha)))])
            res.findings.append(
                Finding(
                    code="loo_flip",
                    severity=Severity.WARN,
                    title=f"Removing a single observation flips significance ({flips_idx.size} such point(s))",
                    detail=(
                        f"Dropping index {j} (x = {fmt_num(float(x[j]))}, y = {fmt_num(float(y[j]))}) "
                        f"gives r = {r_loo[j]:.3f}, p = {fmt_p(float(p_loo[j]))}."
                    ),
                    evidence={"flip_indices": flips_idx.tolist()[:50], "most_influential_index": j},
                    short="one point decides significance",
                )
            )
            res.alternatives.append(
                make_alt(
                    f"Pearson without index {j}",
                    float(p_loo[j]),
                    alpha,
                    Role.SENSITIVITY,
                    reason="leave-one-out",
                    statistic=float(r_loo[j]),
                )
            )
    return driven, drop


def audit_pearsonr(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit ``scipy.stats.pearsonr``: outliers, influence, normality, leave-one-out."""
    x, y, dropped = _paired_xy(rec)
    lx, ly = arg_labels(rec, 2, ["x", "y"])
    alternative = str(rec.params.get("alternative", "two-sided"))
    res = new_result(rec, "Pearson correlation", alpha, target=f"{lx} and {ly}")
    res.data_key = data_key([x, y])
    res.sample_sizes = {"pairs": int(x.size)}
    if x.size < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise Unsupported("need at least 4 non-constant pairs")
    if dropped:
        res.findings.append(
            Finding("missing_values", Severity.INFO, f"{dropped} pair(s) with missing values ignored by the audit")
        )
    res.findings += small_sample_check(int(x.size), "the sample")
    nonnormal = False
    any_out = False
    for v, lab in ((x, lx), (y, ly)):
        f, nn, _ = normality_check(v, lab)
        res.findings += f
        nonnormal = nonnormal or nn
        of, idx = outlier_check(v, lab)
        res.findings += of
        any_out = any_out or idx.size > 0
    severe, _ = _influence(x, y, res, alpha, alternative, res.significant)
    r = float(np.corrcoef(x, y)[0, 1])
    res.effect_sizes = {"r": r, "r_squared": r * r}
    if nonnormal or any_out or severe:
        sp = ostats.spearmanr(x, y, alternative=alternative)
        res.effect_sizes["spearman_rho"] = float(sp.statistic)
        why = []
        if severe:
            why.append("influential points")
        if any_out:
            why.append("outliers")
        if nonnormal:
            why.append("non-normal margins")
        res.alternatives.append(
            make_alt(
                "Spearman rank correlation",
                float(sp.pvalue),
                alpha,
                Role.RECOMMENDED,
                reason=", ".join(why),
                statistic=float(sp.statistic),
                note=f"rho = {float(sp.statistic):.3f}",
            )
        )
    return [res]


def _ols_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    alpha: float,
    rec: CallRecord,
    label: str,
    reported_p: Sequence[float] | None = None,
    robust_already: bool = False,
    alternative: str = "two-sided",
) -> list[AuditResult]:
    """Audit an OLS fit; one :class:`AuditResult` per non-intercept coefficient."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    keep = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[keep], y[keep]
    n, k = X.shape
    if n <= k + 1:
        raise Unsupported("not enough observations for the number of parameters")
    const_cols = [j for j in range(k) if np.ptp(X[:, j]) == 0]
    fit = sx.ols_fit(X, y)
    common: list[Finding] = []
    common += small_sample_check(n, "the regression")
    if n < 10 * max(1, k - len(const_cols)):
        common.append(
            Finding(
                "few_obs_per_predictor",
                Severity.INFO,
                f"Only {n} observations for {k - len(const_cols)} predictor(s)",
                "A common rule of thumb is at least 10-20 observations per predictor.",
                {"n": n, "k": k},
            )
        )
    # residual normality
    f, nonnormal, _ = normality_check(fit.resid, "residuals")
    common += f
    # heteroskedasticity
    hetero = False
    bp_p = None
    if const_cols and k >= 2:
        aux = X if const_cols else sx.add_constant(X)
        try:
            bp, bp_p, bp_df = sx.breusch_pagan(fit.resid, aux, robust=True)
            if bp_p < 0.05:
                hetero = True
                common.append(
                    Finding(
                        "heteroskedasticity",
                        Severity.INFO if robust_already else Severity.WARN,
                        f"Residual variance is not constant (Breusch-Pagan LM = {bp:.2f}, p = {fmt_p(bp_p)})",
                        (
                            "Classical OLS standard errors assume constant variance. HC3 robust standard errors do not."
                            if not robust_already
                            else "You already used robust standard errors."
                        ),
                        {"bp_lm": bp, "bp_p": bp_p, "df": bp_df},
                        short="residual variance is not constant",
                    )
                )
        except Exception:
            pass
    # influence
    h = sx.hat_diagonal(X)
    d = sx.cooks_distance(fit)
    cut = 4.0 / n
    infl = np.flatnonzero(d > cut)
    severe = high_influence_set(h, d, k)
    if infl.size:
        top = int(np.argmax(d))
        common.append(
            Finding(
                "influential_points",
                Severity.INFO,
                f"{infl.size} observation(s) with Cook's D > 4/n = {cut:.3f}",
                f"Most influential: index {top} (Cook's D = {d[top]:.2f}, leverage = {h[top]:.2f}).",
                {"indices": infl.tolist()[:50], "max_cooks_d": float(d[top]), "most_influential_index": top,
                 "severe_indices": severe.tolist()[:50]},
            )
        )
    hc3_se, hc3_t, hc3_p = sx.hc3_tests(fit)
    fit_drop = None
    if severe.size and n - severe.size > k + 1:
        keep2 = np.setdiff1d(np.arange(n), severe)
        try:
            fit_drop = sx.ols_fit(X[keep2], y[keep2])
        except Exception:
            fit_drop = None

    results: list[AuditResult] = []
    for j in range(k):
        if j in const_cols:
            continue
        name = names[j] if j < len(names) else f"x{j}"
        r = new_result(rec, label, alpha, target=f"slope of {name}")
        if reported_p is not None and j < len(reported_p):
            r.reported_p = float(reported_p[j]) if math.isfinite(reported_p[j]) else None
        elif r.reported_p is None:
            r.reported_p = float(fit.pvalues[j])
        r.reported_stat = float(fit.tvalues[j])
        r.data_key = data_key([X[:, j], y])
        r.sample_sizes = {"n": n}
        r.findings = list(common)
        r.effect_sizes = {"coef": float(fit.beta[j]), "se": float(fit.se[j]), "hc3_se": float(hc3_se[j])}
        if hetero and not robust_already:
            r.alternatives.append(
                make_alt(
                    "OLS with HC3 robust SE",
                    float(hc3_p[j]),
                    alpha,
                    Role.RECOMMENDED,
                    reason="heteroskedasticity",
                    statistic=float(hc3_t[j]),
                    note=f"SE {fmt_num(float(fit.se[j]))} → {fmt_num(float(hc3_se[j]))}",
                )
            )
        if fit_drop is not None:
            sig_all = bool(r.reported_p is not None and r.reported_p < alpha)
            sig_drop = bool(fit_drop.pvalues[j] < alpha)
            if sig_all != sig_drop:
                r.findings.append(
                    Finding(
                        "high_leverage",
                        Severity.WARN,
                        f"Dropping {severe.size} high-influence point(s) changes the significance of {name}",
                        f"Without indices {', '.join(map(str, severe.tolist()[:8]))} the p-value is "
                        f"{fmt_p(float(fit_drop.pvalues[j]))}.",
                        {"indices": severe.tolist()[:50]},
                        short=f"{severe.size} high-influence point(s) drive the {name} slope",
                    )
                )
            r.alternatives.append(
                make_alt(
                    f"OLS without {severe.size} high-influence point(s)",
                    float(fit_drop.pvalues[j]),
                    alpha,
                    Role.SENSITIVITY,
                    reason="influence",
                    statistic=float(fit_drop.tvalues[j]),
                    note=f"coef {fmt_num(float(fit.beta[j]))} → {fmt_num(float(fit_drop.beta[j]))}",
                )
            )
        results.append(r)
    return results


def audit_linregress(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit ``scipy.stats.linregress``: residual normality, Breusch-Pagan → HC3, influence."""
    if rec.inputs.get("y") is None:
        raise Unsupported("linregress with a single 2-D argument is not audited")
    x, y, dropped = _paired_xy(rec)
    lx, ly = arg_labels(rec, 2, ["x", "y"])
    alternative = str(rec.params.get("alternative", "two-sided"))
    if x.size < 4 or np.ptp(x) == 0:
        raise Unsupported("need at least 4 points with varying x")
    X = sx.add_constant(x)
    results = _ols_diagnostics(X, y, ["const", lx], alpha, rec, "Linear regression", alternative=alternative)
    res = results[0]
    res.target = f"slope of {ly} on {lx}"
    rp = rec.result.get("pvalue")
    res.reported_p = float(rp) if isinstance(rp, (int, float)) and math.isfinite(rp) else res.reported_p
    sl = rec.result.get("slope")
    if isinstance(sl, (int, float)):
        res.effect_sizes["coef"] = float(sl)
    # leave-one-out for a single slope is the same as for Pearson's r
    if x.size >= 5 and res.significant is not None:
        r_loo, p_loo = loo_pearson_pvalues(x, y, alternative)
        flips = np.flatnonzero((p_loo < alpha) != res.significant)
        if flips.size:
            j = int(flips[np.argmax(np.abs(np.log(p_loo[flips] + 1e-300) - math.log(alpha)))])
            res.findings.append(
                Finding(
                    "loo_flip",
                    Severity.WARN,
                    f"Removing a single observation flips significance ({flips.size} such point(s))",
                    f"Dropping index {j} gives slope p = {fmt_p(float(p_loo[j]))}.",
                    {"flip_indices": flips.tolist()[:50], "most_influential_index": j},
                    short="one point decides significance",
                )
            )
            res.alternatives.append(
                make_alt(f"Slope without index {j}", float(p_loo[j]), alpha, Role.SENSITIVITY, reason="leave-one-out")
            )
    if dropped:
        res.findings.append(Finding("missing_values", Severity.INFO, f"{dropped} pair(s) with missing values ignored"))
    return [res]


def audit_ols_record(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit an OLS fit recorded as ``X``/``y`` inputs (``audit_ols`` or statsmodels)."""
    X = np.asarray(rec.inputs.get("X"), dtype=float)
    y = np.asarray(rec.inputs.get("y"), dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    names: list[Any] = list(rec.params.get("exog_names") or [])
    if not names:
        names = [f"x{j}" for j in range(X.shape[1])]
    reported = rec.result.get("pvalues")
    robust = str(rec.params.get("cov_type", "nonrobust")).lower() not in ("nonrobust", "none", "")
    label = "OLS regression"
    return _ols_diagnostics(X, y, [str(n) for n in names], alpha, rec, label, reported_p=reported, robust_already=robust)
