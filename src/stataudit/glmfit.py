"""Generalized linear models (Poisson and logistic) on numpy, for auditing.

Nothing here depends on statsmodels: the fits use iteratively reweighted
least squares with the canonical link, written in the textbook form so the
numbers can be checked against statsmodels in the tests.

* :func:`glm_fit` - Poisson (log link, optional offset) or logistic (logit
  link, 0/1 response) fit with Wald tests, deviance and the Pearson
  dispersion statistic.
* :func:`sandwich_tests` - robust (HC0 sandwich) Wald tests.
* :func:`quasi_tests` - Wald tests with the standard errors scaled by the
  Pearson dispersion (quasi-Poisson).
* :func:`lr_tests` - likelihood-ratio test for each coefficient, which stays
  valid where Wald tests break down (separation, the Hauck-Donner effect).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as _st

__all__ = ["GLMFit", "glm_fit", "sandwich_tests", "quasi_tests", "lr_tests", "FAMILIES"]

FAMILIES = ("poisson", "binomial")


@dataclass
class GLMFit:
    family: str
    X: np.ndarray
    y: np.ndarray
    offset: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    se: np.ndarray
    z: np.ndarray
    pvalues: np.ndarray
    deviance: float
    pearson_chi2: float
    df_resid: int
    converged: bool
    iterations: int

    @property
    def dispersion(self) -> float:
        """Pearson chi-square divided by the residual degrees of freedom."""
        return self.pearson_chi2 / self.df_resid if self.df_resid > 0 else math.nan


def _variance(family: str, mu: np.ndarray) -> np.ndarray:
    return mu if family == "poisson" else mu * (1.0 - mu)


def _mean(family: str, eta: np.ndarray) -> np.ndarray:
    if family == "poisson":
        return np.exp(np.clip(eta, -700, 700))
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))


def _deviance(family: str, y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        if family == "poisson":
            term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
            return float(2.0 * np.sum(term))
        mu = np.clip(mu, 1e-300, 1 - 1e-16)
        return float(-2.0 * np.sum(y * np.log(mu) + (1 - y) * np.log1p(-mu)))


def glm_fit(X: np.ndarray, y: np.ndarray, family: str, offset: np.ndarray | None = None,
            max_iter: int = 100, tol: float = 1e-10) -> GLMFit:
    """Fit a Poisson or logistic GLM with the canonical link by IRLS.

    With the canonical link, ``d eta / d mu = 1 / V(mu)``, so the working
    response is ``eta - offset + (y - mu) / V(mu)`` and the working weights
    are ``V(mu)``.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    if family == "poisson":
        if np.any(y < 0):
            raise ValueError("Poisson response must be non-negative")
        mu = y + 0.5
        eta = np.log(mu)
    else:
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("logistic response must be 0/1")
        mu = (y + 0.5) / 2.0
        eta = np.log(mu / (1.0 - mu))
    beta = np.zeros(k)
    dev_old = math.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        w = np.maximum(_variance(family, mu), 1e-300)
        zwork = eta - off + (y - mu) / w
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(X * sw[:, None], zwork * sw, rcond=None)[0]
        eta = X @ beta + off
        mu = _mean(family, eta)
        dev = _deviance(family, y, mu)
        if abs(dev - dev_old) <= tol * (abs(dev) + 0.1):
            converged = True
            break
        dev_old = dev
    w = np.maximum(_variance(family, mu), 1e-300)
    cov = np.linalg.pinv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    p = 2.0 * _st.norm.sf(np.abs(z))
    pearson = float(np.sum((y - mu) ** 2 / w))
    return GLMFit(family, X, y, off, beta, mu, cov, se, z, p, _deviance(family, y, mu), pearson,
                  n - k, converged, it)


def _wald(beta: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return z, 2.0 * _st.norm.sf(np.abs(z))


def sandwich_tests(fit: GLMFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust (HC0 sandwich) standard errors, z and p-values.

    For a canonical link the score of observation ``i`` is ``x_i (y_i - mu_i)``,
    so the sandwich is ``B (sum x_i x_i' (y_i - mu_i)^2) B`` with ``B`` the
    model-based covariance.
    """
    r = fit.y - fit.mu
    meat = fit.X.T @ (fit.X * (r ** 2)[:, None])
    cov = fit.cov @ meat @ fit.cov
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z, p = _wald(fit.beta, se)
    return se, z, p


def quasi_tests(fit: GLMFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wald tests with standard errors scaled by the Pearson dispersion (quasi-likelihood)."""
    phi = fit.dispersion
    se = fit.se * math.sqrt(phi) if math.isfinite(phi) and phi > 0 else fit.se
    z, p = _wald(fit.beta, se)
    return se, z, p


def lr_tests(fit: GLMFit) -> tuple[np.ndarray, np.ndarray]:
    """Likelihood-ratio statistic and p-value for dropping each column of ``X``.

    Columns whose removal leaves a singular design get NaN.
    """
    k = fit.X.shape[1]
    stat = np.full(k, np.nan)
    p = np.full(k, np.nan)
    for j in range(k):
        Xr = np.delete(fit.X, j, axis=1)
        if Xr.shape[1] == 0:
            Xr = np.zeros((fit.X.shape[0], 0))
            mu0 = _mean(fit.family, fit.offset) if fit.family == "poisson" else np.full(fit.y.size, 0.5)
            dev0 = _deviance(fit.family, fit.y, mu0)
        else:
            try:
                dev0 = glm_fit(Xr, fit.y, fit.family, fit.offset).deviance
            except (np.linalg.LinAlgError, ValueError):
                continue
        lr = max(dev0 - fit.deviance, 0.0)
        stat[j] = lr
        p[j] = float(_st.chi2.sf(lr, 1))
    return stat, p
