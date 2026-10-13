"""stataudit — a second-opinion statistician for AI-generated analyses.

Trace the statistical tests an analysis actually runs, check their
assumptions on the actual data, recompute robust alternatives and say whether
the conclusion still holds.

Quick start::

    import stataudit
    rep = stataudit.audit_script("analysis.py")
    print(rep.to_text())

    with stataudit.recording() as calls:
        scipy.stats.ttest_ind(a, b)
    print(stataudit.report(calls).to_text())
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import (
    audit_1samp_ttest,
    audit_anova,
    audit_call,
    audit_chi2,
    audit_data,
    audit_linregress,
    audit_ols,
    audit_paired_ttest,
    audit_pearson,
    audit_script,
    audit_ttest,
    report,
)
from .audits import audit_records
from .models import AuditResult, CallRecord, Finding, Report, Severity, Status
from .trace import recording, trace_script

__all__ = [
    "__version__",
    "audit_call",
    "audit_script",
    "audit_records",
    "audit_ttest",
    "audit_paired_ttest",
    "audit_1samp_ttest",
    "audit_pearson",
    "audit_anova",
    "audit_chi2",
    "audit_linregress",
    "audit_ols",
    "audit_data",
    "report",
    "recording",
    "trace_script",
    "Report",
    "AuditResult",
    "CallRecord",
    "Finding",
    "Severity",
    "Status",
]
