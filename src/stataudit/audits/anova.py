"""Auditor for one-way ANOVA (``scipy.stats.f_oneway``)."""

from __future__ import annotations

import numpy as np

from .. import stats_extra as sx
from .._scipy import ostats
from ..models import AuditResult, CallRecord, Finding, Role, Severity
from .common import (
    SMALL_N,
    Unsupported,
    arg_labels,
    as_1d,
    data_key,
    fmt_p,
    make_alt,
    new_result,
    normality_check,
    outlier_check,
    small_sample_check,
)

__all__ = ["audit_f_oneway", "groups_from_record"]


def groups_from_record(rec: CallRecord) -> tuple[list[np.ndarray], list[str]]:
    samples = rec.inputs.get("samples")
    if not samples or len(samples) < 2:
        raise Unsupported("need at least two groups")
    groups = [as_1d(s, f"group {i + 1}")[0] for i, s in enumerate(samples)]
    labels = arg_labels(rec, len(groups), [f"group {i + 1}" for i in range(len(groups))])
    return groups, labels


def audit_f_oneway(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Audit one-way ANOVA: per-group normality, outliers, Levene → Welch ANOVA and Kruskal-Wallis."""
    groups, labels = groups_from_record(rec)
    if any(g.size < 2 for g in groups):
        raise Unsupported("every group needs at least two observations")
    equal_var = bool(rec.params.get("equal_var", True))
    res = new_result(rec, "One-way ANOVA" if equal_var else "Welch's ANOVA", alpha, target=f"{len(groups)} groups")
    res.data_key = data_key(groups)
    res.sample_sizes = {lab: int(g.size) for lab, g in zip(labels, groups)}
    nonnormal_small = False
    any_out = False
    for g, lab in zip(groups, labels):
        res.findings += small_sample_check(int(g.size), lab)
        f, nn, _ = normality_check(g, lab)
        res.findings += f
        nonnormal_small = nonnormal_small or (nn and g.size < SMALL_N)
        of, idx = outlier_check(g, lab)
        res.findings += of
        any_out = any_out or idx.size > 0
    variances = np.array([g.var(ddof=1) for g in groups])
    ratio = float(variances.max() / variances.min()) if variances.min() > 0 else float("inf")
    lev_p = None
    try:
        lev_p = float(ostats.levene(*groups, center="median").pvalue)
    except Exception:
        pass
    sizes = np.array([g.size for g in groups])
    unequal = (lev_p is not None and lev_p < 0.05) or ratio >= 4
    if unequal:
        res.findings.append(
            Finding(
                "unequal_variances",
                Severity.WARN if equal_var else Severity.INFO,
                f"Group variances differ up to {ratio:.1f}× (Levene p = {fmt_p(lev_p)})",
                "Classical ANOVA assumes equal variances; with unequal group sizes the F-test can be badly "
                "miscalibrated. Welch's ANOVA does not need this assumption.",
                {"var_ratio": ratio, "levene_p": lev_p, "variances": variances.tolist(), "sizes": sizes.tolist()},
                short=f"group variances differ {ratio:.1f}×",
            )
        )
    res.effect_sizes = {"eta_squared": sx.eta_squared(*groups)}
    if equal_var and unequal:
        try:
            f, p, df1, df2 = sx.welch_anova(*groups)
            res.alternatives.append(
                make_alt("Welch's ANOVA", p, alpha, Role.RECOMMENDED, reason="variances differ", statistic=f,
                         note=f"F({df1:.0f}, {df2:.1f})")
            )
        except ValueError:
            pass
    if nonnormal_small or any_out:
        kw = ostats.kruskal(*groups)
        res.alternatives.append(
            make_alt(
                "Kruskal-Wallis H",
                float(kw.pvalue),
                alpha,
                Role.RECOMMENDED,
                reason="non-normal small groups" if nonnormal_small else "outliers present",
                statistic=float(kw.statistic),
            )
        )
    if res.significant:
        res.findings.append(
            Finding(
                "posthoc_needed",
                Severity.INFO,
                "A significant ANOVA only says that some group means differ",
                "Use a post-hoc procedure (e.g. Games-Howell or Holm-corrected pairwise Welch tests) to say which.",
            )
        )
    return [res]
