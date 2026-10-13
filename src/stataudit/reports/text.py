"""Terminal output: plain formatted text with optional ANSI colour."""

from __future__ import annotations

import os
import sys
import textwrap

from ..audits.common import fmt_num, fmt_p
from ..models import Report, Role, Severity, Status
from ..verdict import STATUS_ASCII, STATUS_ICON

_SEV_ICON = {Severity.INFO: "·", Severity.WARN: "!", Severity.CRITICAL: "✗"}
_SEV_ASCII = {Severity.INFO: ".", Severity.WARN: "!", Severity.CRITICAL: "x"}
_COLORS = {
    Status.HOLDS: "\033[32m",
    Status.FRAGILE: "\033[33m",
    Status.FAILS: "\033[31m",
    Status.NOT_AUDITED: "\033[90m",
}
_BOLD, _DIM, _RESET = "\033[1m", "\033[2m", "\033[0m"

_HEADLINE = {
    Status.HOLDS: "Conclusions hold",
    Status.FRAGILE: "Some conclusions are fragile",
    Status.FAILS: "At least one conclusion does NOT hold",
    Status.NOT_AUDITED: "Nothing to audit",
}


def _can_encode(s: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        s.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def render_text(
    report: Report,
    color: bool = False,
    ascii_only: bool | None = None,
    width: int | None = None,
    verbose: bool = False,
) -> str:
    """Render the report for a terminal.

    Clean results (status "holds" with no warnings) are shown as a single
    verdict line unless ``verbose`` is true.
    """
    if ascii_only is None:
        ascii_only = not _can_encode("✅⚠️❌·✗—×")
    if width is None:
        try:
            width = min(100, max(60, os.get_terminal_size().columns))
        except OSError:
            width = 88
    icons = STATUS_ASCII if ascii_only else STATUS_ICON
    sev_icons = _SEV_ASCII if ascii_only else _SEV_ICON

    def c(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    def wrap(text: str, indent: str) -> list[str]:
        if ascii_only:
            text = text.replace("—", "-").replace("×", "x").replace("→", "->").replace("α", "alpha")
        return textwrap.wrap(
            text, width=width, initial_indent=indent, subsequent_indent=indent, break_on_hyphens=False
        ) or [indent]

    lines: list[str] = []
    rule = "-" * width if ascii_only else "─" * width
    title = f"stataudit report — {report.source}" if report.source else "stataudit report"
    lines += wrap(title, "")
    lines.append(rule)
    counts = report.counts()
    summary = (
        f"{icons[report.status]} {_HEADLINE[report.status]}  "
        f"({len(report.results)} test(s): {counts['holds']} hold, {counts['fragile']} fragile, "
        f"{counts['does_not_hold']} do not hold; alpha = {report.alpha:g})"
    )
    lines.append(c(summary, _BOLD + _COLORS[report.status]))
    lines.append("")
    for i, r in enumerate(report.results, 1):
        head = f"{icons[r.status]} [{i}] {r.test_label}"
        if r.target:
            head += f" — {r.target}"
        head += f"  ({r.location})"
        for ln in wrap(head, ""):
            lines.append(c(ln, _BOLD + _COLORS[r.status]))
        if r.call.site and r.call.site.code:
            lines += [c(ln, _DIM) for ln in wrap(r.call.site.code, "    > ")]
        lines += wrap(r.verdict, "    ")
        clean = r.status is Status.HOLDS and not any(f.severity is not Severity.INFO for f in r.findings)
        if clean and not verbose:
            lines.append("")
            continue
        shown = [f for f in r.findings if f.severity is not Severity.INFO] + [
            f for f in r.findings if f.severity is Severity.INFO
        ]
        for f in shown[:8]:
            lines += wrap(f"{sev_icons[f.severity]} {f.title}", "      ")
        alts = [a for a in r.alternatives if a.pvalue is not None]
        if alts:
            lines.append("      alternatives on the same data:")
            for a in alts:
                mark = "sig" if a.significant else "n.s."
                extra = f"  [{a.note}]" if a.note else ""
                role = "" if a.role is Role.RECOMMENDED else f" ({a.role.value})"
                lines += wrap(f"- {a.name}{role}: p = {fmt_p(a.pvalue)} ({mark}){extra}", "        ")
        es = {k: v for k, v in r.effect_sizes.items() if k not in ("se", "hc3_se")}
        if es:
            lines += wrap("effect sizes: " + ", ".join(f"{k} = {fmt_num(v)}" for k, v in es.items()), "      ")
        lines.append("")
    if report.script_findings:
        lines.append("Script-level checks")
        lines.append(rule)
        for f in report.script_findings:
            lines += wrap(f"{sev_icons[f.severity]} {f.title}", "  ")
            if f.detail and f.code != "script_error":
                lines += wrap(f.detail, "    ")
            if f.code == "script_error":
                err = str(f.evidence.get("error", "")).strip().splitlines()
                lines += ["    " + e for e in err[-3:]]
        lines.append("")
    if report.multiplicity:
        lines.append("Holm-Bonferroni")
        lines.append(rule)
        for row in sorted(report.multiplicity, key=lambda d: d["pvalue"]):
            mark = "survives" if row["survives"] else ("LOST" if row["significant"] else "")
            lines.append(
                f"  {row['location']:<28} {row['test'][:34]:<34} p = {fmt_p(row['pvalue']):<8} "
                f"Holm p = {fmt_p(row['holm_pvalue']):<8} {mark}".rstrip()
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
