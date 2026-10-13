"""Markdown output."""

from __future__ import annotations

from ..audits.common import fmt_num, fmt_p
from ..models import Report, Role, Severity, Status
from ..verdict import STATUS_ICON

_HEADLINE = {
    Status.HOLDS: "Conclusions hold",
    Status.FRAGILE: "Some conclusions are fragile",
    Status.FAILS: "At least one conclusion does NOT hold",
    Status.NOT_AUDITED: "Nothing to audit",
}
_SEV = {Severity.INFO: "info", Severity.WARN: "**warn**", Severity.CRITICAL: "**CRITICAL**"}


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Report) -> str:
    """Render the report as GitHub-flavoured Markdown."""
    out: list[str] = []
    out.append("# stataudit report")
    out.append("")
    if report.source:
        out.append(f"**Source:** `{report.source}`  ")
    counts = report.counts()
    out.append(f"**Overall:** {STATUS_ICON[report.status]} {_HEADLINE[report.status]}  ")
    out.append(
        f"**Tests audited:** {len(report.results)} ({counts['holds']} hold, {counts['fragile']} fragile, "
        f"{counts['does_not_hold']} do not hold) at α = {report.alpha:g}"
    )
    out.append("")
    if report.results:
        out.append("| | Location | Test | Reported p | Verdict |")
        out.append("|---|---|---|---|---|")
        for r in report.results:
            tgt = f" ({_esc(r.target)})" if r.target else ""
            out.append(
                f"| {STATUS_ICON[r.status]} | `{r.location}` | {_esc(r.test_label)}{tgt} | {fmt_p(r.reported_p)} | "
                f"{r.status.label} |"
            )
        out.append("")
    for i, r in enumerate(report.results, 1):
        out.append(f"## {STATUS_ICON[r.status]} {i}. {r.test_label} — `{r.location}`")
        out.append("")
        if r.call.site and r.call.site.code:
            out.append("```python")
            out.append(r.call.site.code)
            out.append("```")
            out.append("")
        out.append(f"> {r.verdict}")
        out.append("")
        if r.findings:
            out.append("**Findings**")
            out.append("")
            for f in r.findings:
                line = f"- {_SEV[f.severity]}: {f.title}"
                if f.detail:
                    line += f" — {f.detail}"
                out.append(line)
            out.append("")
        alts = [a for a in r.alternatives if a.pvalue is not None]
        if alts:
            out.append("**Alternative analyses on the same data**")
            out.append("")
            out.append("| Analysis | Why | p-value | Significant? | Notes |")
            out.append("|---|---|---|---|---|")
            for a in alts:
                why = a.reason if a.role is Role.RECOMMENDED else f"{a.role.value}: {a.reason}"
                out.append(
                    f"| {_esc(a.name)} | {_esc(why)} | {fmt_p(a.pvalue)} | {'yes' if a.significant else 'no'} | "
                    f"{_esc(a.note)} |"
                )
            out.append("")
        if r.effect_sizes:
            es = ", ".join(f"{k} = {fmt_num(v)}" for k, v in r.effect_sizes.items())
            out.append(f"**Effect sizes:** {es}  ")
        if r.sample_sizes:
            ss = ", ".join(f"{k}: {v}" for k, v in r.sample_sizes.items())
            out.append(f"**Sample sizes:** {ss}")
        out.append("")
    if report.script_findings:
        out.append("## Script-level checks")
        out.append("")
        for f in report.script_findings:
            out.append(f"- {_SEV[f.severity]}: **{f.title}**" + (f" — {f.detail}" if f.detail and f.code != "script_error" else ""))
            if f.code == "script_error":
                out.append("")
                out.append("  ```text")
                for ln in str(f.evidence.get("error", "")).strip().splitlines()[-6:]:
                    out.append(f"  {ln}")
                out.append("  ```")
        out.append("")
    if report.multiplicity:
        out.append("## Multiple comparisons (Holm-Bonferroni)")
        out.append("")
        out.append("| Location | Test | p | Holm-adjusted p | Survives? |")
        out.append("|---|---|---|---|---|")
        for row in sorted(report.multiplicity, key=lambda d: d["pvalue"]):
            surv = "yes" if row["survives"] else ("**no**" if row["significant"] else "—")
            out.append(
                f"| `{row['location']}` | {_esc(row['test'])} {_esc(row.get('target') or '')} | {fmt_p(row['pvalue'])} | "
                f"{fmt_p(row['holm_pvalue'])} | {surv} |"
            )
        out.append("")
    out.append("---")
    out.append(
        "*stataudit checks assumptions and robustness of the tests it sees. It cannot know your study design, "
        "and it is not a substitute for a statistician.*"
    )
    return "\n".join(out) + "\n"
