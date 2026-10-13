"""Auditors for chi-square tests of independence and Fisher's exact test."""

from __future__ import annotations

import numpy as np

from .. import stats_extra as sx
from .._scipy import ostats
from ..models import AuditResult, CallRecord, Finding, Role, Severity
from .common import Unsupported, arg_labels, data_key, fmt_num, make_alt, new_result

__all__ = ["audit_chi2_contingency", "audit_fisher_exact", "MC_RESAMPLES", "MC_SEED"]

MC_RESAMPLES = 20000
MC_SEED = 20240611


def _table(rec: CallRecord, name: str) -> np.ndarray:
    t = rec.inputs.get(name)
    if t is None:
        raise Unsupported("no table supplied")
    t = np.asarray(t, dtype=float)
    if t.ndim != 2:
        raise Unsupported("only two-way tables are audited")
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise Unsupported("table must contain non-negative finite counts")
    return t


def audit_chi2_contingency(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit ``scipy.stats.chi2_contingency``.

    Checks Cochran's rule (no expected count < 1 and at most 20% below 5).
    When it fails, computes Fisher's exact test (2×2) or a Monte Carlo
    permutation p-value with fixed seed (r×c).
    """
    t = _table(rec, "observed")
    (lab,) = arg_labels(rec, 1, ["table"])
    r, c = t.shape
    res = new_result(rec, "Chi-square test of independence", alpha, target=f"{r}×{c} table" + (f" {lab}" if lab != "table" else ""))
    res.data_key = data_key([t])
    n = int(round(t.sum()))
    res.sample_sizes = {"total count": n}
    if not np.allclose(t, np.round(t)):
        res.findings.append(
            Finding(
                "non_integer_counts",
                Severity.CRITICAL,
                "The table contains non-integer values",
                "Chi-square tests need raw counts, not proportions, percentages or means.",
                short="the table is not made of counts",
            )
        )
    e = sx.expected_counts(t)
    cells = e.size
    lt5 = int(np.sum(e < 5))
    lt1 = int(np.sum(e < 1))
    frac = lt5 / cells
    res.effect_sizes = {"cramers_v": sx.cramers_v(t)}
    correction = bool(rec.params.get("correction", True))
    if (r, c) == (2, 2) and correction:
        res.findings.append(
            Finding(
                "yates",
                Severity.INFO,
                "Yates' continuity correction was applied (the default for 2×2)",
                "It makes the test conservative; Fisher's exact test is the usual choice for small 2×2 tables.",
            )
        )
    violated = lt1 > 0 or frac > 0.2
    if violated:
        res.findings.append(
            Finding(
                "low_expected",
                Severity.WARN,
                f"{lt5} of {cells} cells ({frac:.0%}) have expected count < 5; minimum is {fmt_num(float(e.min()))}",
                "Cochran's rule: the chi-square approximation needs no expected count below 1 and at most 20% "
                "below 5. The reported p-value may be far from the true one.",
                {"expected_min": float(e.min()), "cells_lt5": lt5, "cells_lt1": lt1, "cells": cells,
                 "expected": e.tolist()},
                short=f"{frac:.0%} of cells have expected count < 5",
            )
        )
        tt = np.round(t).astype(int)
        if (r, c) == (2, 2):
            fe = ostats.fisher_exact(tt)
            res.alternatives.append(
                make_alt("Fisher's exact test", float(fe.pvalue), alpha, Role.RECOMMENDED,
                         reason="small expected counts", statistic=float(fe.statistic), note="odds ratio")
            )
        else:
            stat, p = sx.chi2_monte_carlo(tt, n_resamples=MC_RESAMPLES, seed=MC_SEED)
            res.alternatives.append(
                make_alt(
                    "Monte Carlo permutation chi-square",
                    p,
                    alpha,
                    Role.RECOMMENDED,
                    reason="small expected counts",
                    statistic=stat,
                    note=f"{MC_RESAMPLES:,} tables with fixed margins, seed {MC_SEED}",
                )
            )
            res.findings.append(
                Finding(
                    "combine_categories",
                    Severity.INFO,
                    "Consider combining sparse categories",
                    "If merging rare levels is scientifically meaningful, it restores the large-sample approximation.",
                )
            )
    if n < 20:
        res.findings.append(
            Finding("very_small_sample", Severity.WARN, f"Only {n} observations in the table", short=f"only {n} observations")
        )
    return [res]


def audit_fisher_exact(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Fisher's exact test is exact; report the table size and effect size only."""
    t = _table(rec, "table")
    res = new_result(rec, "Fisher's exact test", alpha, target=f"{t.shape[0]}×{t.shape[1]} table")
    res.data_key = data_key([t])
    res.sample_sizes = {"total count": int(round(t.sum()))}
    res.effect_sizes = {"odds_ratio": float(rec.result.get("statistic", float("nan")))}
    if t.sum() < 20:
        res.findings.append(
            Finding("very_small_sample", Severity.INFO, f"Only {int(t.sum())} observations",
                    "The exact test is valid but has little power; a non-significant result is weak evidence.")
        )
    if np.any(t == 0):
        res.findings.append(
            Finding("zero_cell", Severity.INFO, "A cell is zero, so the odds ratio is 0 or infinite",
                    "Report a conditional MLE or a confidence interval rather than the raw odds ratio.")
        )
    return [res]
