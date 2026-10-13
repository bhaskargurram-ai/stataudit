"""Shared building blocks for the per-test auditors."""

from __future__ import annotations

import ast
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats as _st

from .. import stats_extra as sx
from ..models import Alternative, AuditResult, CallRecord, Finding, Role, Severity

#: Below this per-group size, normal-theory tests lean heavily on the normality assumption.
SMALL_N = 30
#: Below this per-group size we always flag the sample as very small.
VERY_SMALL_N = 10

__all__ = [
    "SMALL_N",
    "VERY_SMALL_N",
    "Unsupported",
    "fmt_p",
    "fmt_num",
    "as_1d",
    "pvalue_of",
    "stat_of",
    "new_result",
    "arg_labels",
    "normality_check",
    "outlier_check",
    "small_sample_check",
    "make_alt",
    "data_key",
]


class Unsupported(Exception):
    """Raised when a call's inputs are outside what an auditor understands."""


def fmt_p(p: float | None) -> str:
    """Format a p-value for prose: ``0.031``, ``0.21``, ``3.2e-05``."""
    if p is None or not isinstance(p, (int, float)) or not math.isfinite(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.1e}"
    s = f"{p:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def fmt_num(x: float | None, digits: int = 3) -> str:
    """Format a number compactly."""
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "n/a"
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def as_1d(value: Any, name: str = "data") -> tuple[np.ndarray, int]:
    """Return a 1-D float array with NaNs removed, plus the number of NaNs dropped."""
    if value is None:
        raise Unsupported(f"{name} is missing")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        raise Unsupported(f"{name} is a scalar, not a sample")
    if arr.ndim > 1:
        squeezed = np.squeeze(arr)
        if squeezed.ndim != 1:
            raise Unsupported(f"{name} is {arr.ndim}-dimensional; only 1-D samples are audited")
        arr = squeezed
    mask = np.isfinite(arr)
    return arr[mask], int((~mask).sum())


def pvalue_of(rec: CallRecord) -> float | None:
    p = rec.result.get("pvalue")
    if isinstance(p, (int, float)) and math.isfinite(p):
        return float(p)
    return None


def stat_of(rec: CallRecord) -> float | None:
    s = rec.result.get("statistic")
    if isinstance(s, (int, float)) and math.isfinite(s):
        return float(s)
    return None


def new_result(rec: CallRecord, label: str, alpha: float, target: str = "") -> AuditResult:
    return AuditResult(
        call=rec,
        test_label=label,
        alpha=alpha,
        reported_p=pvalue_of(rec),
        reported_stat=stat_of(rec),
        target=target,
    )


def arg_labels(rec: CallRecord, count: int, default: Sequence[str]) -> list[str]:
    """Best-effort names for positional data arguments, read from the call-site source.

    ``stats.ttest_ind(control, treated)`` yields ``["control", "treated"]``.
    Falls back to ``default`` when the source line cannot be parsed.
    """
    explicit = rec.params.get("_labels")
    if isinstance(explicit, list) and len(explicit) >= count:
        return [str(x) for x in explicit[:count]]
    labels = list(default)[:count]
    code = rec.site.code if rec.site else ""
    if not code:
        return labels
    tree = None
    for candidate in (code, code.rstrip(",;") + ")", code + "\n    pass"):
        try:
            tree = ast.parse(candidate.strip())
            break
        except SyntaxError:
            continue
    if tree is None:
        return labels
    func_name = rec.func
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if name != func_name:
            continue
        exprs: list[str] = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                return labels
            exprs.append(ast.unparse(a))
        out = []
        for i in range(count):
            if i < len(exprs) and len(exprs[i]) <= 40:
                out.append(exprs[i])
            else:
                out.append(labels[i] if i < len(labels) else f"sample {i + 1}")
        return out
    return labels


def normality_check(x: np.ndarray, label: str, alpha_check: float = 0.05) -> tuple[list[Finding], bool, dict[str, float | None]]:
    """Shapiro-Wilk plus skewness / excess kurtosis for one sample.

    Returns ``(findings, looks_non_normal, evidence)``. A sample looks
    non-normal when Shapiro-Wilk rejects at ``alpha_check`` or, for samples
    too large for Shapiro, when |skew| > 1 or |excess kurtosis| > 3.
    """
    n = x.size
    sw = sx.safe_shapiro(x)
    sk, ku = sx.skew_kurt(x)
    ev: dict[str, float | None] = {"n": n, "shapiro_p": sw, "skewness": sk, "excess_kurtosis": ku}
    nonnormal = False
    if sw is not None:
        nonnormal = sw < alpha_check
    elif n > 5000 and math.isfinite(sk):
        nonnormal = abs(sk) > 1 or abs(ku) > 3
    findings: list[Finding] = []
    if nonnormal:
        bits = []
        if sw is not None:
            bits.append(f"Shapiro-Wilk p = {fmt_p(sw)}")
        if math.isfinite(sk):
            bits.append(f"skew = {sk:.2f}")
        if math.isfinite(ku):
            bits.append(f"excess kurtosis = {ku:.2f}")
        small = n < SMALL_N
        sev = Severity.WARN if small else Severity.INFO
        detail = (
            "With small samples the normal-theory p-value can be badly off."
            if small
            else f"With n = {n} the central limit theorem usually protects the test of means, "
            "but heavy tails still cost power."
        )
        findings.append(
            Finding(
                code="non_normal",
                severity=sev,
                title=f"{label} looks non-normal ({', '.join(bits)})",
                detail=detail,
                evidence=ev,
                short=f"{label} is non-normal (n = {n})" if small else "",
            )
        )
    return findings, nonnormal, ev


#: Robust |z| above which a point is reported as an outlier.
OUTLIER_Z = 3.5
#: Robust |z| above which an outlier is considered extreme (warning + robust alternatives).
EXTREME_Z = 5.0


def outlier_check(x: np.ndarray, label: str) -> tuple[list[Finding], np.ndarray]:
    """Flag outliers by robust z (MAD) > 3.5, cross-checked with Tukey's IQR fences.

    Points with robust |z| between 3.5 and 5 occur by chance in moderately
    large normal samples, so they are reported at info level only. If any
    point exceeds |z| = 5 the finding is a warning and the returned indices
    (all points above 3.5) trigger robust alternative analyses. Otherwise
    an empty index array is returned.
    """
    empty = np.array([], dtype=int)
    if x.size < 5:
        return [], empty
    mad_idx = sx.mad_outliers(x, OUTLIER_Z)
    if mad_idx.size == 0:
        return [], empty
    iqr_idx = sx.iqr_outliers(x, 1.5)
    rz = sx.robust_z(x)
    both = np.intersect1d(mad_idx, iqr_idx)
    worst = int(mad_idx[np.argmax(np.abs(rz[mad_idx]))])
    zmax = float(np.max(np.abs(rz[mad_idx])))
    extreme = zmax > EXTREME_Z
    k = mad_idx.size
    word = "outlier" if k == 1 else "outliers"
    adj = "" if extreme else "mild "
    finding = Finding(
        code="outliers",
        severity=Severity.WARN if extreme else Severity.INFO,
        short=f"{label} has {k} {word}" if extreme else "",
        title=f"{label} has {k} {adj}{word} (robust |z| > {OUTLIER_Z:g}, max {zmax:.1f})",
        detail=(
            f"Most extreme: value {fmt_num(float(x[worst]))} at position {worst} "
            f"(robust z = {rz[worst]:.1f}); {both.size} of {k} are also outside Tukey's 1.5×IQR fences. "
            + (
                "Means and standard deviations are sensitive to such points."
                if extreme
                else "Values this far out occur by chance in larger samples; no action needed unless they are errors."
            )
        ),
        evidence={
            "indices": mad_idx.tolist()[:50],
            "values": x[mad_idx].tolist()[:50],
            "robust_z_max": zmax,
            "iqr_outliers": int(iqr_idx.size),
        },
    )
    return [finding], (mad_idx if extreme else empty)


def small_sample_check(n: int, label: str) -> list[Finding]:
    if n < VERY_SMALL_N:
        return [
            Finding(
                code="very_small_sample",
                severity=Severity.WARN,
                short=f"{label} has only n = {n}",
                title=f"{label} has only n = {n}",
                detail="Assumption checks have almost no power at this size and estimates are unstable.",
                evidence={"n": n},
            )
        ]
    return []


def make_alt(
    name: str,
    pvalue: float | None,
    alpha: float,
    role: Role,
    reason: str = "",
    statistic: float | None = None,
    note: str = "",
    **extra: Any,
) -> Alternative:
    sig = None
    if pvalue is not None and math.isfinite(pvalue):
        sig = bool(pvalue < alpha)
        pvalue = float(pvalue)
    else:
        pvalue = None
    return Alternative(
        name=name,
        pvalue=pvalue,
        role=role,
        reason=reason,
        statistic=None if statistic is None or not math.isfinite(float(statistic)) else float(statistic),
        significant=sig,
        note=note,
        extra=extra,
    )


def _hash_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


def data_key(arrays: Iterable[np.ndarray]) -> str:
    """Order-insensitive fingerprint of the data a test was run on."""
    hs = sorted(_hash_array(a) for a in arrays)
    return hashlib.sha1("|".join(hs).encode()).hexdigest()[:16]


def two_sided_from_t(t: float, df: float, alternative: str) -> float:
    """p-value for a t statistic under the given alternative."""
    if alternative == "less":
        return float(_st.t.cdf(t, df))
    if alternative == "greater":
        return float(_st.t.sf(t, df))
    return float(2 * _st.t.sf(abs(t), df))
