"""Per-test auditors and the dispatcher that runs them over traced calls."""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..models import AuditResult, CallRecord, Finding, Report, Severity, Status
from .anova import audit_f_oneway
from .common import Unsupported, new_result
from .contingency import audit_chi2_contingency, audit_fisher_exact
from .nonparametric import audit_kruskal, audit_mannwhitneyu, audit_shapiro, audit_spearmanr, audit_wilcoxon
from .regression import audit_linregress, audit_ols_record, audit_pearsonr
from .ttests import audit_ttest_1samp, audit_ttest_ind, audit_ttest_rel

log = logging.getLogger("stataudit.audits")

Auditor = Callable[[CallRecord, float], list[AuditResult]]

#: Function name → auditor.
AUDITORS: dict[str, Auditor] = {
    "ttest_ind": audit_ttest_ind,
    "ttest_rel": audit_ttest_rel,
    "ttest_1samp": audit_ttest_1samp,
    "pearsonr": audit_pearsonr,
    "spearmanr": audit_spearmanr,
    "f_oneway": audit_f_oneway,
    "chi2_contingency": audit_chi2_contingency,
    "fisher_exact": audit_fisher_exact,
    "mannwhitneyu": audit_mannwhitneyu,
    "wilcoxon": audit_wilcoxon,
    "kruskal": audit_kruskal,
    "linregress": audit_linregress,
    "shapiro": audit_shapiro,
    "ols": audit_ols_record,
    "statsmodels_ols": audit_ols_record,
}

_LABELS = {
    "ttest_ind": "Two-sample t-test",
    "ttest_rel": "Paired t-test",
    "ttest_1samp": "One-sample t-test",
    "pearsonr": "Pearson correlation",
    "spearmanr": "Spearman correlation",
    "f_oneway": "One-way ANOVA",
    "chi2_contingency": "Chi-square test",
    "fisher_exact": "Fisher's exact test",
    "mannwhitneyu": "Mann-Whitney U",
    "wilcoxon": "Wilcoxon signed-rank",
    "kruskal": "Kruskal-Wallis H",
    "linregress": "Linear regression",
    "shapiro": "Shapiro-Wilk normality test",
    "ols": "OLS regression",
    "statsmodels_ols": "OLS regression",
}

__all__ = ["AUDITORS", "audit_record", "audit_records", "Unsupported"]


def audit_record(rec: CallRecord, alpha: float = 0.05) -> list[AuditResult]:
    """Run the matching auditor. Never raises: problems become a result with a finding."""
    auditor = AUDITORS.get(rec.func)
    label = _LABELS.get(rec.func, rec.func)
    if auditor is None:
        res = new_result(rec, label, alpha)
        res.findings.append(Finding("unsupported", Severity.INFO, f"No auditor for {rec.func}"))
        return [res]
    try:
        out = auditor(rec, alpha)
    except Unsupported as exc:
        res = new_result(rec, label, alpha)
        res.in_family = rec.func != "shapiro"
        res.status = Status.NOT_AUDITED
        res.findings.append(Finding("unsupported", Severity.INFO, "Call shape not audited", str(exc)))
        return [res]
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("auditor for %s failed", rec.func, exc_info=True)
        res = new_result(rec, label, alpha)
        res.findings.append(Finding("audit_error", Severity.INFO, "Internal audit error", repr(exc)))
        return [res]
    return out


def audit_records(
    records: list[CallRecord],
    alpha: float = 0.05,
    source: str | None = None,
    script_error: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Report:
    """Audit a list of traced calls and return the finished :class:`Report`."""
    from ..verdict import finalize

    results: list[AuditResult] = []
    for rec in records:
        results.extend(audit_record(rec, alpha))
    return finalize(results, alpha, source=source, script_error=script_error, meta=meta)
