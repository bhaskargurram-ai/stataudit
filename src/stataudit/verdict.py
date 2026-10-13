"""Turn findings and alternative analyses into a status and a plain-English verdict."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from . import stats_extra as sx
from .audits.common import VERY_SMALL_N, fmt_p
from .models import (
    SEVERITY_RANK,
    STATUS_RANK,
    AuditResult,
    Finding,
    Report,
    Role,
    Severity,
    Status,
)

__all__ = [
    "STATUS_ICON",
    "STATUS_ASCII",
    "decide_status",
    "write_verdict",
    "apply_multiplicity",
    "script_findings",
    "overall_status",
    "finalize",
]

STATUS_ICON = {Status.HOLDS: "✅", Status.FRAGILE: "⚠️", Status.FAILS: "❌", Status.NOT_AUDITED: "➖"}
STATUS_ASCII = {Status.HOLDS: "[OK]", Status.FRAGILE: "[??]", Status.FAILS: "[XX]", Status.NOT_AUDITED: "[--]"}

#: Order in which problems are mentioned in the one-line verdict (most decisive first).
_PRIORITY = {
    "non_integer_counts": 0,
    "missing_values": 1,
    "high_leverage": 2,
    "loo_flip": 3,
    "outliers": 4,
    "unequal_variances": 5,
    "low_expected": 6,
    "heteroskedasticity": 7,
    "non_normal": 8,
    "very_small_sample": 9,
}

_ARTICLE_SKIP = ("Student", "Welch", "Yuen", "Pearson", "Spearman", "Fisher", "Mann", "Wilcoxon", "Kruskal", "Shapiro")


def decide_status(res: AuditResult) -> Status:
    """Status of one result.

    * ``FAILS`` - a critical finding, or every *recommended* alternative (and
      reaches the opposite conclusion.
    * ``FRAGILE`` - some but not all recommended alternatives disagree, a
      sensitivity analysis (leave-one-out, dropping influential points) flips
      it, or it does not survive the Holm-Bonferroni correction.
    * ``HOLDS`` - otherwise, even if there are warnings.
    """
    if res.reported_p is None:
        if any(f.severity is Severity.CRITICAL for f in res.findings):
            return Status.FAILS
        return Status.NOT_AUDITED
    sig = res.significant
    if any(f.severity is Severity.CRITICAL for f in res.findings):
        return Status.FAILS
    rec = [a for a in res.alternatives if a.role is Role.RECOMMENDED and a.significant is not None]
    disagree = [a for a in rec if a.significant != sig]
    if rec and len(disagree) == len(rec):
        return Status.FAILS
    if disagree:
        return Status.FRAGILE
    # Losing significance after Holm makes a result fragile rather than wrong:
    # whether the tests form one family is a design question stataudit cannot answer.
    mult = [a for a in res.alternatives if a.role is Role.MULTIPLICITY and a.significant is not None]
    if any(a.significant != sig for a in mult):
        return Status.FRAGILE
    sens = [a for a in res.alternatives if a.role is Role.SENSITIVITY and a.significant is not None]
    if any(a.significant != sig for a in sens):
        return Status.FRAGILE
    return Status.HOLDS


def _article(label: str) -> str:
    return label if label.startswith(_ARTICLE_SKIP) else label[0].lower() + label[1:]


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _cap(s: str) -> str:
    """Capitalise the first letter unless the sentence starts with a code identifier."""
    if not s:
        return s
    first = s.split(" ", 1)[0]
    if len(first) <= 2 or "_" in first or "." in first or "[" in first:
        return s
    return s[:1].upper() + s[1:]


def write_verdict(res: AuditResult) -> str:
    """One plain-English sentence (or three) explaining the status."""
    test = _article(res.test_label)
    loc = res.location
    if res.reported_p is None:
        problems = _join([f.short or f.title for f in res.findings if f.severity is not Severity.INFO][:3])
        if res.status is Status.FAILS:
            return f"Your {test} ({loc}) did not produce a usable p-value: {problems}."
        reason = next((f.detail or f.title for f in res.findings if f.code in ("unsupported", "audit_error")), "")
        return f"Your {test} ({loc}) was recorded but not audited" + (f": {reason}" if reason else ".")
    sig_word = "significant" if res.significant else "not significant"
    target = f" for {res.target}" if res.target and res.test_label.startswith(("OLS", "Linear")) else ""
    head = f"Your {test}{target} ({loc}) reports p = {fmt_p(res.reported_p)} ({sig_word})."
    flagged = [f for f in res.findings if f.severity is not Severity.INFO and f.short]
    flagged.sort(key=lambda f: (-SEVERITY_RANK[f.severity], _PRIORITY.get(f.code, 50)))
    seen: set[str] = set()
    problems = [f.short for f in flagged if not (f.short in seen or seen.add(f.short))][:3]
    alts = [a for a in res.alternatives if a.role is not Role.MULTIPLICITY and a.pvalue is not None]
    mult = [
        a for a in res.alternatives
        if a.role is Role.MULTIPLICITY and a.pvalue is not None and a.significant != res.significant
    ]
    alt_txt = _join([f"{a.name} gives p = {fmt_p(a.pvalue)}" for a in alts[:4]])
    mid_parts = []
    if problems:
        mid_parts.append(_cap(_join(problems)))
    if mult:
        m = mult[0]
        mid_parts.append(
            f"{'a' if mid_parts else 'A'}fter Holm correction for {m.extra.get('family_size', '?')} tests "
            f"the adjusted p = {fmt_p(m.pvalue)}"
        )
    mid = "; ".join(mid_parts)
    if alt_txt:
        mid = f"{mid}; {alt_txt}" if mid else _cap(alt_txt)
    tail = {
        Status.HOLDS: "the conclusion holds.",
        Status.FRAGILE: "the conclusion is fragile.",
        Status.FAILS: "the conclusion does NOT hold.",
        Status.NOT_AUDITED: "not audited.",
    }[res.status]
    if res.status is Status.HOLDS and not mid:
        return f"{head} No assumption problems found — {tail}"
    if res.status is Status.HOLDS and alts:
        return f"{head} {mid} — the alternatives agree, so {tail}"
    return f"{head} {mid} — {tail}"


def apply_multiplicity(results: list[AuditResult], alpha: float) -> tuple[list[dict], Finding | None]:
    """Holm-Bonferroni over every in-family p-value.

    Adds a ``MULTIPLICITY`` alternative to each nominally significant result
    and returns the table plus a script-level finding.
    """
    fam = [r for r in results if r.in_family and r.reported_p is not None and math.isfinite(r.reported_p)]
    m = len(fam)
    nsig = sum(1 for r in fam if r.reported_p < alpha)
    if m < 2 or nsig == 0:
        return [], None
    adj = sx.holm([r.reported_p for r in fam])
    table: list[dict] = []
    survivors = 0
    for r, pa in zip(fam, adj):
        survive = bool(pa < alpha)
        survivors += survive and r.reported_p < alpha
        table.append(
            {
                "location": r.location,
                "test": r.test_label,
                "target": r.target,
                "pvalue": r.reported_p,
                "holm_pvalue": float(pa),
                "significant": bool(r.reported_p < alpha),
                "survives": survive,
            }
        )
        if r.reported_p < alpha:
            r.alternatives.append(
                _mult_alt(float(pa), alpha, m)
            )
    lost = nsig - survivors
    sev = Severity.WARN if lost else Severity.INFO
    finding = Finding(
        "multiple_comparisons",
        sev,
        f"{m} p-values, {nsig} significant at α = {alpha:g}; {survivors} survive Holm-Bonferroni",
        (
            "Running many tests and reporting the significant ones inflates the false-positive rate: with "
            f"{m} independent true-null tests the chance of at least one p < {alpha:g} is "
            f"{1 - (1 - alpha) ** m:.0%}. Holm's step-down procedure controls the family-wise error rate."
        ),
        {"m": m, "significant": nsig, "survive_holm": int(survivors)},
        short=f"{lost} of {nsig} significant results do not survive correction" if lost else "",
    )
    return table, finding


def _mult_alt(p_adj: float, alpha: float, m: int):
    from .audits.common import make_alt

    return make_alt("Holm-adjusted", p_adj, alpha, Role.MULTIPLICITY, reason=f"{m} tests", family_size=m)


def script_findings(results: list[AuditResult]) -> list[Finding]:
    """Script-level smells: repeated testing of the same data, very small samples."""
    out: list[Finding] = []
    by_key: dict[str, list[AuditResult]] = defaultdict(list)
    for r in results:
        if r.data_key:
            by_key[r.data_key].append(r)
    for key, rs in by_key.items():
        funcs = sorted({r.call.func for r in rs if r.call.func != "shapiro"})
        if len(funcs) >= 3:
            locs = ", ".join(sorted({r.location for r in rs}))
            ps = [r.reported_p for r in rs if r.reported_p is not None]
            out.append(
                Finding(
                    "test_shopping",
                    Severity.WARN,
                    f"The same data were tested with {len(funcs)} different tests ({', '.join(funcs)})",
                    "Trying several tests on the same data and reporting the one that \"works\" is a form of "
                    f"p-hacking. Pick the test from the design before looking at results. Calls at {locs}; "
                    f"p-values ranged from {fmt_p(min(ps)) if ps else 'n/a'} to {fmt_p(max(ps)) if ps else 'n/a'}.",
                    {"functions": funcs, "locations": locs},
                )
            )
    small = sorted(
        {r.location for r in results if r.sample_sizes and min(r.sample_sizes.values()) < VERY_SMALL_N}
    )
    if small:
        out.append(
            Finding(
                "small_samples",
                Severity.INFO,
                f"{len(small)} test(s) use a group with fewer than {VERY_SMALL_N} observations",
                "At these sizes assumption checks have little power and effect estimates are noisy. "
                f"Locations: {', '.join(small[:10])}.",
                {"locations": small},
            )
        )
    return out


def overall_status(results: Iterable[AuditResult], script_level: Iterable[Finding] = ()) -> Status:
    worst = Status.NOT_AUDITED
    for r in results:
        if STATUS_RANK[r.status] > STATUS_RANK[worst]:
            worst = r.status
    if any(f.code == "test_shopping" for f in script_level) and STATUS_RANK[worst] < STATUS_RANK[Status.FRAGILE]:
        worst = Status.FRAGILE
    return worst


def finalize(
    results: list[AuditResult],
    alpha: float,
    source: str | None = None,
    script_error: str | None = None,
    meta: dict | None = None,
) -> Report:
    """Apply multiplicity, compute statuses and verdicts, and build the :class:`Report`."""
    table, mfinding = apply_multiplicity(results, alpha)
    for r in results:
        if r.status is not Status.NOT_AUDITED or r.reported_p is not None or r.findings:
            r.status = decide_status(r)
        r.verdict = write_verdict(r)
    sfind = script_findings(results)
    if mfinding is not None:
        sfind.insert(0, mfinding)
    if script_error:
        sfind.insert(
            0,
            Finding(
                "script_error",
                Severity.WARN,
                "The analysis script did not finish",
                "Only the calls made before the error were audited.",
                {"error": script_error[-2000:]},
            ),
        )
    if not results:
        sfind.append(
            Finding(
                "nothing_traced",
                Severity.INFO,
                "No supported statistical calls were recorded",
                "stataudit traces scipy.stats tests (and statsmodels OLS when installed). Calls made through other "
                "libraries are not seen.",
            )
        )
    return Report(
        results=results,
        alpha=alpha,
        status=overall_status(results, sfind),
        script_findings=sfind,
        source=source,
        script_error=script_error,
        multiplicity=table,
        meta=dict(meta or {}),
    )
