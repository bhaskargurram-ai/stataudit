"""stataudit — check AI-written statistics with evidence instead of trust.

Trace the statistical tests an analysis actually runs, check their
assumptions on the actual data, recompute robust alternatives and say whether
the conclusion still holds; recompute the p-values reported in text
(statcheck-style); lint analysis code statically.

Quick start::

    import stataudit
    rep = stataudit.audit_script("analysis.py")
    print(rep.to_text())

    with stataudit.recording() as calls:
        scipy.stats.ttest_ind(a, b)
    print(stataudit.report(calls).to_text())

    print(stataudit.check_report_text("t(28) = 2.20, p = .36").to_text())
    print(stataudit.lint_code(open("analysis.py").read()).to_text())
"""

from __future__ import annotations

__version__ = "1.0.1"

from .api import (
    audit_1samp_ttest,
    audit_anova,
    audit_call,
    audit_chi2,
    audit_data,
    audit_linregress,
    audit_notebook,
    audit_ols,
    audit_paired_ttest,
    audit_pearson,
    audit_script,
    audit_ttest,
    report,
)
from .audits import audit_records
from .lint import lint_file
from .lint import lint_source as lint_code
from .reportcheck import check_file as check_report_file
from .reportcheck import check_text as check_report_text
from .models import AuditResult, CallRecord, Finding, Report, Severity, Status
from .trace import recording, trace_script

__all__ = [
    "__version__",
    "audit_call",
    "audit_script",
    "audit_notebook",
    "audit_records",
    "check_report_text",
    "check_report_file",
    "lint_code",
    "lint_file",
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
