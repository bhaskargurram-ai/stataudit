"""Statistical helpers implemented directly on numpy/scipy.

Nothing here depends on statsmodels. Every function is deliberately small and
written in the textbook form so it can be checked against an independent
calculation (see ``tests/test_stats_extra.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as _st

from ._scipy import orig

__all__ = [
    "OLSFit",
    "ols_fit",
    "hat_diagonal",
    "cooks_distance",
    "hc3_cov",
    "hc3_tests",
    "breusch_pagan",
    "welch_anova",
    "holm",
    "chi2_statistic",
    "expected_counts",
    "chi2_monte_carlo",
    "robust_z",
    "mad_outliers",
    "iqr_outliers",
    "cohens_d",
    "hedges_g",
    "rank_biserial",
    "cramers_v",
    "eta_squared",
    "safe_shapiro",
    "skew_kurt",
]


# ---------------------------------------------------------------------------
# Ordinary least squares and diagnostics
# ---------------------------------------------------------------------------


@dataclass
class OLSFit:
    """Result of an ordinary least squares fit ``y = X @ beta + e``."""

    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    xtx_inv: np.ndarray
    sigma2: float
    se: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    df_resid: int

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def k(self) -> int:
        return int(self.X.shape[1])


def add_constant(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to a 1-D or 2-D design."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(x.shape[0]), x])


def has_constant(X: np.ndarray) -> bool:
    """True if any column of ``X`` is constant and non-zero (an intercept)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return False
    return bool(np.any((np.ptp(X, axis=0) == 0) & (X[0] != 0)))


def ols_fit(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Fit OLS with classical (homoskedastic) standard errors.

    ``X`` must already contain an intercept column if one is wanted.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    df_resid = n - k
    sigma2 = float(resid @ resid / df_resid)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = beta / se
    pvalues = 2 * _st.t.sf(np.abs(tvalues), df_resid)
    return OLSFit(X, y, beta, resid, fitted, xtx_inv, sigma2, se, tvalues, pvalues, df_resid)


def hat_diagonal(X: np.ndarray) -> np.ndarray:
    """Leverages h_ii = diag(X (X'X)^-1 X') computed stably via QR."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    q, _ = np.linalg.qr(X)
    return np.sum(q * q, axis=1)


def cooks_distance(fit: OLSFit) -> np.ndarray:
    """Cook's distance D_i = e_i^2 / (k s^2) * h_i / (1 - h_i)^2."""
    h = hat_diagonal(fit.X)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = fit.resid**2 / (fit.k * fit.sigma2) * h / (1 - h) ** 2
    return np.where(np.isfinite(d), d, np.inf)


def hc3_cov(fit: OLSFit) -> np.ndarray:
    """HC3 heteroskedasticity-consistent covariance (MacKinnon & White 1985).

    ``(X'X)^-1 X' diag(e_i^2 / (1 - h_i)^2) X (X'X)^-1``
    """
    h = hat_diagonal(fit.X)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = fit.resid**2 / (1 - h) ** 2
    omega = np.where(np.isfinite(omega), omega, 0.0)
    meat = (fit.X * omega[:, None]).T @ fit.X
    return fit.xtx_inv @ meat @ fit.xtx_inv


def hc3_tests(fit: OLSFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HC3 standard errors, t statistics and two-sided p-values.

    p-values use a t distribution with ``n - k`` degrees of freedom, which is
    slightly more conservative than the normal reference some packages use.
    """
    se = np.sqrt(np.clip(np.diag(hc3_cov(fit)), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = fit.beta / se
    p = 2 * _st.t.sf(np.abs(t), fit.df_resid)
    return se, t, p


def breusch_pagan(resid: np.ndarray, X: np.ndarray, robust: bool = True) -> tuple[float, float, int]:
    """Breusch-Pagan test for heteroskedasticity.

    ``X`` is the design used for the auxiliary regression and must contain an
    intercept. With ``robust=True`` (default) this is Koenker's studentized
    version, ``LM = n R^2`` from regressing e^2 on X. With ``robust=False`` it
    is the original Breusch-Pagan statistic, ``ESS / 2`` from regressing
    ``e^2 / (SSR/n)`` on X.

    Returns ``(statistic, pvalue, df)`` with ``df = k - 1``.
    """
    e = np.asarray(resid, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    df = k - 1
    if df < 1:
        raise ValueError("auxiliary design needs at least one non-constant column")
    e2 = e**2
    if robust:
        aux = ols_fit(X, e2)
        tss = float(np.sum((e2 - e2.mean()) ** 2))
        if tss == 0:
            return 0.0, 1.0, df
        r2 = 1 - float(aux.resid @ aux.resid) / tss
        lm = n * r2
    else:
        g = e2 / (e2.sum() / n)
        aux = ols_fit(X, g)
        ess = float(np.sum((aux.fitted - g.mean()) ** 2))
        lm = ess / 2
    return float(lm), float(_st.chi2.sf(lm, df)), df


# ---------------------------------------------------------------------------
# ANOVA and multiplicity
# ---------------------------------------------------------------------------


def welch_anova(*groups: Sequence[float]) -> tuple[float, float, float, float]:
    """Welch's heteroskedasticity-robust one-way ANOVA (Welch 1951).

    Returns ``(F, pvalue, df1, df2)``.
    """
    gs = [np.asarray(g, dtype=float) for g in groups]
    k = len(gs)
    if k < 2:
        raise ValueError("need at least two groups")
    n = np.array([g.size for g in gs], dtype=float)
    if np.any(n < 2):
        raise ValueError("each group needs at least two observations")
    m = np.array([g.mean() for g in gs])
    v = np.array([g.var(ddof=1) for g in gs])
    if np.any(v == 0):
        raise ValueError("a group has zero variance")
    w = n / v
    W = w.sum()
    mw = float((w * m).sum() / W)
    a = float((w * (m - mw) ** 2).sum() / (k - 1))
    tmp = float((((1 - w / W) ** 2) / (n - 1)).sum())
    b = 1 + 2 * (k - 2) / (k**2 - 1) * tmp
    f = a / b
    df1 = k - 1.0
    df2 = (k**2 - 1) / (3 * tmp)
    return float(f), float(_st.f.sf(f, df1, df2)), df1, float(df2)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (Holm 1979).

    The adjusted p-value for the i-th smallest p is
    ``max_{j<=i} min(1, (m - j + 1) p_(j))``. Non-finite inputs are passed
    through as NaN and excluded from ``m``.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv, kind="mergesort")
    adj_sorted = np.minimum(1.0, (m - np.arange(m)) * pv[order])
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj = np.empty(m)
    adj[order] = adj_sorted
    out[ok] = adj
    return out


# ---------------------------------------------------------------------------
# Contingency tables
# ---------------------------------------------------------------------------


def expected_counts(table: np.ndarray) -> np.ndarray:
    """Expected counts under independence, ``row_sum * col_sum / total``."""
    t = np.asarray(table, dtype=float)
    total = t.sum()
    if total == 0:
        return np.zeros_like(t)
    return np.outer(t.sum(axis=1), t.sum(axis=0)) / total


def chi2_statistic(table: np.ndarray) -> float:
    """Pearson's chi-square statistic (no continuity correction).

    Cells with zero expected count (empty rows/columns) are ignored.
    """
    t = np.asarray(table, dtype=float)
    e = expected_counts(t)
    mask = e > 0
    return float(np.sum((t[mask] - e[mask]) ** 2 / e[mask]))


def chi2_monte_carlo(table: np.ndarray, n_resamples: int = 10000, seed: int = 12345) -> tuple[float, float]:
    """Monte Carlo permutation p-value for the chi-square test of independence.

    Tables with the observed row and column margins are sampled uniformly by
    permuting column labels against row labels (equivalent to sampling from
    the multivariate hypergeometric null). Returns ``(statistic, pvalue)``
    where ``pvalue = (1 + #{T* >= T}) / (1 + B)``.
    """
    t = np.asarray(table)
    if np.any(t < 0) or not np.allclose(t, np.round(t)):
        raise ValueError("Monte Carlo test needs a table of non-negative integer counts")
    t = np.round(t).astype(int)
    # drop empty rows / columns
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    r, c = t.shape
    observed = chi2_statistic(t)
    if r < 2 or c < 2:
        return observed, 1.0
    rows = np.repeat(np.arange(r), t.sum(axis=1))
    cols = np.repeat(np.arange(c), t.sum(axis=0))
    rng = np.random.default_rng(seed)
    e = expected_counts(t)
    count = 0
    batch = 1000
    done = 0
    tol = 1e-9 * max(1.0, observed)
    while done < n_resamples:
        b = min(batch, n_resamples - done)
        perm = np.argsort(rng.random((b, cols.size)), axis=1)
        pc = cols[perm]
        codes = rows[None, :] * c + pc
        offs = (np.arange(b) * r * c)[:, None]
        tabs = np.bincount((codes + offs).ravel(), minlength=b * r * c).reshape(b, r, c)
        stats = np.sum((tabs - e) ** 2 / e, axis=(1, 2))
        count += int(np.sum(stats >= observed - tol))
        done += b
    return observed, (1 + count) / (1 + n_resamples)


def cramers_v(table: np.ndarray) -> float:
    """Cramér's V effect size for an r x c table."""
    t = np.asarray(table, dtype=float)
    n = t.sum()
    k = min(t.shape) - 1
    if n == 0 or k == 0:
        return float("nan")
    return float(math.sqrt(chi2_statistic(t) / (n * k)))


# ---------------------------------------------------------------------------
# Descriptives, outliers, effect sizes
# ---------------------------------------------------------------------------


def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-scores ``0.6745 (x - median) / MAD`` (Iglewicz & Hoaglin).

    Falls back to the mean absolute deviation (scaled by 1.2533) when the MAD
    is zero, and to zeros when all values are equal.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 0:
        return 0.6745 * (x - med) / mad
    meanad = np.mean(np.abs(x - med))
    if meanad > 0:
        return (x - med) / (1.253314 * meanad)
    return np.zeros_like(x)


def mad_outliers(x: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Indices with robust |z| above ``threshold``."""
    return np.flatnonzero(np.abs(robust_z(x)) > threshold)


def iqr_outliers(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Indices outside ``[Q1 - k IQR, Q3 + k IQR]`` (Tukey's fences)."""
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return np.flatnonzero((x < q1 - k * iqr) | (x > q3 + k * iqr))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using the pooled standard deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(sp2))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g: Cohen's d times the small-sample correction J."""
    d = cohens_d(a, b)
    df = np.size(a) + np.size(b) - 2
    j = 1 - 3 / (4 * df - 1) if df > 0 else float("nan")
    return float(d * j)


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from the Mann-Whitney U of the first sample.

    ``r = 2 U / (n1 n2) - 1``; positive means the first sample tends to be larger.
    """
    return float(2 * u_stat / (n1 * n2) - 1)


def eta_squared(*groups: Sequence[float]) -> float:
    """Eta squared, SS_between / SS_total, for a one-way layout."""
    gs = [np.asarray(g, dtype=float) for g in groups]
    allv = np.concatenate(gs)
    grand = allv.mean()
    ss_tot = float(np.sum((allv - grand) ** 2))
    if ss_tot == 0:
        return float("nan")
    ss_b = float(sum(g.size * (g.mean() - grand) ** 2 for g in gs))
    return ss_b / ss_tot


def safe_shapiro(x: np.ndarray) -> float | None:
    """Shapiro-Wilk p-value for 3 <= n <= 5000 and non-constant data, else None."""
    x = np.asarray(x, dtype=float)
    if not 3 <= x.size <= 5000 or np.ptp(x) == 0:
        return None
    try:
        return float(orig("shapiro")(x).pvalue)
    except Exception:
        return None


def skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """Sample skewness and excess kurtosis (bias-corrected when n allows)."""
    x = np.asarray(x, dtype=float)
    if x.size < 4 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    return float(_st.skew(x, bias=False)), float(_st.kurtosis(x, bias=False))
