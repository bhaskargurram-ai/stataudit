"""Audits for Poisson and logistic regression (statsmodels ``GLM``, ``Logit``, ``Poisson``).

* **Poisson**: overdispersion (Pearson chi-square / df). With overdispersed
  counts the model-based standard errors are too small; quasi-Poisson
  (dispersion-scaled) standard errors are the recommended alternative and a
  robust sandwich the sensitivity check.
* **Logistic**: few events per variable, and (quasi-)complete separation,
  where Wald tests break down (the Hauck-Donner effect: a huge coefficient
  with a huge standard error and p close to 1). Likelihood-ratio tests are the
  recommended alternative under separation.
* **Both**: leave-one-out refits (n <= 300) that flip a coefficient's
  significance, and robust (sandwich) standard errors.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..glmfit import GLMFit, glm_fit, lr_tests, quasi_tests, sandwich_tests
from ..models import AuditResult, CallRecord, Finding, Role, Severity
from .common import Unsupported, data_key, fmt_num, fmt_p, make_alt, new_result, small_sample_check

__all__ = ["audit_glm_record", "LOO_MAX_N"]

#: Leave-one-out refits are done up to this many observations.
LOO_MAX_N = 300

_LABELS = {"poisson": "Poisson regression", "binomial": "Logistic regression"}


def _separation(fit: GLMFit) -> tuple[bool, int]:
    extreme = int(np.sum((fit.mu < 1e-8) | (fit.mu > 1 - 1e-8)))
    sd = fit.X.std(axis=0)
    big = bool(np.any((sd > 0) & (np.abs(fit.beta) * sd > 15)))
    return (not fit.converged) or extreme > 0 or big, extreme


def _loo_flips(fit: GLMFit, alpha: float, cols: Sequence[int]) -> dict[int, tuple[int, float]]:
    """For each column, an observation whose removal flips significance (index, p without it)."""
    n = fit.y.size
    out: dict[int, tuple[int, float]] = {}
    sig = fit.pvalues < alpha
    worst: dict[int, tuple[int, float, float]] = {}
    for i in range(n):
        keep = np.arange(n) != i
        try:
            f = glm_fit(fit.X[keep], fit.y[keep], fit.family, fit.offset[keep])
        except (np.linalg.LinAlgError, ValueError):
            continue
        for j in cols:
            pj = float(f.pvalues[j])
            if not math.isfinite(pj) or (pj < alpha) == sig[j]:
                continue
            dist = abs(math.log(max(pj, 1e-300)) - math.log(alpha))
            if j not in worst or dist > worst[j][2]:
                worst[j] = (i, pj, dist)
    for j, (i, pj, _) in worst.items():
        out[j] = (i, pj)
    return out


def audit_glm_record(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit a Poisson or logistic fit recorded as ``X``/``y`` (and optional ``offset``)."""
    family = str(rec.params.get("family", "")).lower()
    if family not in _LABELS:
        raise Unsupported(f"GLM family {family or 'unknown'} with this link is not audited "
                          "(Poisson with log link and logistic with logit link are)")
    if rec.params.get("weighted"):
        raise Unsupported("weighted GLM fits are not audited")
    X = np.asarray(rec.inputs.get("X"), dtype=float)
    y = np.asarray(rec.inputs.get("y"), dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    off = rec.inputs.get("offset")
    offset = None if off is None else np.asarray(off, dtype=float).ravel()
    keep = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if offset is not None:
        keep &= np.isfinite(offset)
        offset = offset[keep]
    X, y = X[keep], y[keep]
    n, k = X.shape
    if n <= k + 1:
        raise Unsupported("not enough observations for the number of parameters")
    if family == "binomial" and not np.all((y == 0) | (y == 1)):
        raise Unsupported("logistic regression on a non-0/1 response (proportions) is not audited")
    try:
        fit = glm_fit(X, y, family, offset)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise Unsupported(f"could not refit the model: {exc}") from exc
    names: list[Any] = list(rec.params.get("exog_names") or []) or [f"x{j}" for j in range(k)]
    reported: Sequence[float] | None = rec.result.get("pvalues")
    robust_already = str(rec.params.get("cov_type", "nonrobust")).lower() not in ("nonrobust", "none", "")
    label = _LABELS[family]
    const_cols = [j for j in range(k) if np.ptp(X[:, j]) == 0]
    cols = [j for j in range(k) if j not in const_cols]

    common: list[Finding] = list(small_sample_check(n, f"the {label.lower()}"))
    rob_se, rob_z, rob_p = sandwich_tests(fit)
    q_se = q_z = q_p = None
    lr_stat = lr_p = None
    overdispersed = separated = few_events = False

    if family == "poisson":
        phi = fit.dispersion
        if math.isfinite(phi) and phi > 1.5 and fit.df_resid >= 10:
            overdispersed = True
            q_se, q_z, q_p = quasi_tests(fit)
            common.append(Finding(
                "overdispersion", Severity.INFO if robust_already else Severity.WARN,
                f"Counts are overdispersed: Pearson chi-square / df = {phi:.2f}",
                "Poisson regression assumes the variance equals the mean. With more variance than that, its "
                "standard errors are too small and its p-values too optimistic. Quasi-Poisson (standard errors "
                f"scaled by sqrt({phi:.2f})) or a negative binomial model accounts for it."
                + (" You already used robust standard errors." if robust_already else ""),
                {"dispersion": phi, "df_resid": fit.df_resid}, short=f"counts are overdispersed (×{phi:.1f})"))
    else:
        events = int(min(y.sum(), n - y.sum()))
        epv = events / max(1, len(cols))
        if epv < 10:
            few_events = True
            common.append(Finding(
                "few_events_per_variable", Severity.WARN,
                f"Only {events} events for {len(cols)} predictor(s) ({epv:.1f} per predictor)",
                "Logistic regression coefficients are biased away from zero and Wald tests are unreliable with "
                "fewer than about 10 events (the rarer outcome) per predictor.",
                {"events": events, "predictors": len(cols), "events_per_variable": epv},
                short=f"{epv:.1f} events per predictor"))
        sep, extreme = _separation(fit)
        if sep:
            separated = True
            common.append(Finding(
                "separation", Severity.WARN,
                "The outcome is (almost) perfectly separated by the predictors",
                f"{extreme} observation(s) have a fitted probability of 0 or 1"
                + ("" if fit.converged else " and the fit did not converge")
                + ". The maximum-likelihood coefficient is infinite, so the Wald test (estimate / standard error) "
                "is meaningless; a likelihood-ratio test or penalized (Firth) regression is needed.",
                {"extreme_fitted": extreme, "converged": fit.converged}, short="the outcome is separated"))
        if separated or few_events:
            lr_stat, lr_p = lr_tests(fit)

    loo: dict[int, tuple[int, float]] = {}
    if n <= LOO_MAX_N and not separated:
        loo = _loo_flips(fit, alpha, cols)

    results: list[AuditResult] = []
    for j in cols:
        name = str(names[j]) if j < len(names) else f"x{j}"
        r = new_result(rec, label, alpha, target=f"coefficient of {name}")
        if reported is not None and j < len(reported) and isinstance(reported[j], (int, float)) \
                and math.isfinite(reported[j]):
            r.reported_p = float(reported[j])
        else:
            r.reported_p = float(fit.pvalues[j])
        r.reported_stat = float(fit.z[j])
        r.data_key = data_key([X[:, j], y])
        r.sample_sizes = {"n": n}
        r.findings = list(common)
        r.effect_sizes = {"coef": float(fit.beta[j]), "se": float(fit.se[j]), "robust_se": float(rob_se[j]),
                          ("rate_ratio" if family == "poisson" else "odds_ratio"):
                              float(math.exp(min(fit.beta[j], 700)))}
        if overdispersed and q_p is not None:
            r.alternatives.append(make_alt(
                "Quasi-Poisson (dispersion-scaled SE)", float(q_p[j]), alpha,
                Role.SENSITIVITY if robust_already else Role.RECOMMENDED, reason="overdispersion",
                statistic=float(q_z[j]), note=f"SE {fmt_num(float(fit.se[j]))} → {fmt_num(float(q_se[j]))}"))
        if lr_p is not None and math.isfinite(lr_p[j]):
            r.alternatives.append(make_alt(
                "Likelihood-ratio test", float(lr_p[j]), alpha,
                Role.RECOMMENDED if separated else Role.SENSITIVITY,
                reason="separation" if separated else "few events", statistic=float(lr_stat[j])))
        if not robust_already and not separated:  # no finite estimate to make robust under separation
            r.alternatives.append(make_alt(
                "Robust (sandwich) standard errors", float(rob_p[j]), alpha, Role.SENSITIVITY,
                reason="model misspecification", statistic=float(rob_z[j]),
                note=f"SE {fmt_num(float(fit.se[j]))} → {fmt_num(float(rob_se[j]))}"))
        if j in loo:
            i, pj = loo[j]
            r.findings.append(Finding(
                "loo_flip", Severity.WARN, f"Removing a single observation flips the significance of {name}",
                f"Without index {i} the p-value is {fmt_p(pj)}.", {"index": i, "pvalue": pj},
                short="one observation decides significance"))
            r.alternatives.append(make_alt(f"Without index {i}", pj, alpha, Role.SENSITIVITY,
                                           reason="leave-one-out"))
        results.append(r)
    return results
