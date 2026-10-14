"""``stataudit check-report``: recompute p-values from APA-style results in text.

This is a statcheck-style consistency check (Nuijten et al.): find reported
results such as ``t(28) = 2.20, p = .036`` in a Markdown, text, HTML or PDF
document, recompute the p-value from the test statistic and the degrees of
freedom with scipy, and flag results whose reported p-value cannot follow
from the reported statistic.

Recognised forms (case, spacing, Markdown emphasis and Unicode variants are
normalised first)::

    t(28) = 2.20, p = .036          t(27.4) = -2.1, p < .05       t[28] = 2.2, p = .04
    F(2, 45) = 3.81, p = .030       F(1; 38) = 4.5, MSE = 2.1, p = .04
    r(48) = .32, p = .023           r(48) = −.32, p = .02
    χ2(2, N = 120) = 7.31, p = .026 χ²(1) = 3.9, p < .05          chi2(3) = 8.1, p = .04
    z = 2.31, p = .021              Q(12) = 25.1, p = .014
    t = 2.20, df = 28, p-value = 0.036          (R's t.test output style)
    X-squared = 7.31, df = 2, p-value = 0.026    (R's chisq.test output style)
    F = 3.2, num df = 2, denom df = 26.7, p-value = 0.057

and the p-value forms ``p = .036``, ``p < .001``, ``p > .05``, ``p ≤ .05``,
``ps < .01``, ``p = 2.3 × 10⁻⁵``, ``p = 2.3e-5`` and ``ns`` / ``n.s.``.

Rounding: the reported statistic (and a decimal df) is taken as an interval
of half a unit in its last reported decimal, and so is an exact reported
p-value. A result is *consistent* when some statistic inside its interval
gives a p-value inside the reported p interval. An *inconsistent* result is
a *gross* inconsistency when the reported p-value and the recomputed one
fall on opposite sides of alpha (the significance decision flips). As in
statcheck, a reported ``p = .05`` counts as significant at alpha = .05.

Also flagged as *impossible*: p outside [0, 1] or exactly 0, negative or zero
degrees of freedom, a correlation outside [-1, 1], and negative F or χ².

Known gaps: results in tables split over cells, tests without a df-based
reference distribution (U, W, H, ...), and corrected p-values (these are
recognised by words such as "Bonferroni", "Holm", "adjusted" or "corrected"
near the p-value and only checked to be at least the uncorrected p).
"""

from __future__ import annotations

import bisect
import html
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from scipy import stats as _st

__all__ = [
    "CONSISTENT",
    "INCONSISTENT",
    "GROSS",
    "IMPOSSIBLE",
    "STATUSES",
    "ReportedResult",
    "ReportCheck",
    "extract_results",
    "check_text",
    "check_file",
    "read_document",
    "recompute_p",
]

CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"
GROSS = "gross"
IMPOSSIBLE = "impossible"
STATUSES = (CONSISTENT, INCONSISTENT, GROSS, IMPOSSIBLE)
_STATUS_RANK = {CONSISTENT: 0, INCONSISTENT: 1, GROSS: 2, IMPOSSIBLE: 3}
STATUS_ICON = {CONSISTENT: "✅", INCONSISTENT: "⚠️", GROSS: "❌", IMPOSSIBLE: "❌"}
STATUS_LABEL = {
    CONSISTENT: "consistent",
    INCONSISTENT: "inconsistent",
    GROSS: "gross inconsistency",
    IMPOSSIBLE: "impossible value",
}

#: Largest document we read (characters after extraction).
MAX_DOCUMENT_CHARS = 5_000_000
PDF_TIMEOUT = 60

# --------------------------------------------------------------------------- normalisation

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_MINUS_RE = re.compile("[\u2212\u2013\u2012\ufe63\uff0d]")
_SPACE_RE = re.compile("[\u00a0\u2000-\u200b\u202f\u205f\u3000\t]")
#: Greek/mathematical chi variants and the Latin X some authors use.
_CHI_RE = re.compile("[\u03c7\u03a7\U0001d712\U0001d6d8\U0001d74c\U0001d786\U0001d7c0]")


def normalize(text: str) -> str:
    """Normalise one line for matching. Keeps the length-independent structure (no newlines added/removed)."""
    s = _SPACE_RE.sub(" ", text)
    s = _MINUS_RE.sub("-", s)
    s = _CHI_RE.sub("χ", s)
    s = s.replace("≤", "<=").replace("≥", ">=").replace("&lt;", "<").replace("&gt;", ">")
    s = s.replace("\\chi", "χ").replace("$", "")
    # Markdown / LaTeX emphasis around symbols: *t*(28), _p_ < .05, **F**(1, 2), \textit{p}
    s = re.sub(r"\\(?:textit|mathit|emph|mathrm|text)\{([^{}]{1,3})\}", r"\1", s)
    s = s.replace("*", "").replace("_", "").replace("{", "").replace("}", "")
    s = s.translate(_SUPERSCRIPTS)
    return s


# --------------------------------------------------------------------------- numbers

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_SCI = rf"(?:{_NUM})(?:\s*(?:[eE]\s*[-+]?\d+|[x×·*]\s*10\s*\^?\s*\(?[-+]?\s*\d+\)?))?"
_OP = r"(<=|>=|=<|=>|<|>|=)"


@dataclass(frozen=True)
class Num:
    """A reported number with the precision it was reported at."""

    text: str
    value: float
    #: Half-width of the rounding interval (0 for numbers we treat as exact).
    half: float

    @property
    def low(self) -> float:
        return self.value - self.half

    @property
    def high(self) -> float:
        return self.value + self.half


def parse_num(text: str) -> Num:
    """Parse ``.036``, ``2.20``, ``-3``, ``2.3e-5``, ``2.3 × 10^-5`` (and its normalised superscripts)."""
    t = re.sub(r"\s+", "", text)
    m = re.fullmatch(rf"({_NUM})(?:[eE]([-+]?\d+)|[x×·*]10\^?\(?([-+]?\d+)\)?)?", t)
    if not m:
        raise ValueError(f"not a number: {text!r}")
    mant, e1, e2 = m.group(1), m.group(2), m.group(3)
    exp = int(e1 or e2 or 0)
    decimals = len(mant.split(".", 1)[1]) if "." in mant else 0
    value = float(mant) * 10.0**exp
    half = 0.5 * 10.0 ** (-decimals) * 10.0**exp
    return Num(text.strip(), value, half)


# --------------------------------------------------------------------------- patterns

_SEP = r"\s*[,;/]\s*"
_DF = r"(\d+(?:\.\d+)?|-\s*\d+(?:\.\d+)?)"
_OPEN, _CLOSE = r"\s*[(\[]\s*", r"\s*[)\]]"

_STAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # R output styles first (they contain "df =")
    ("t", re.compile(rf"(?<![A-Za-z])t\s*=\s*(?P<stat>{_NUM})\s*,\s*df\s*=\s*(?P<df1>{_DF})", re.I)),
    ("chi2", re.compile(rf"(?:(?<![A-Za-z])X|χ|chi)\s*-?\s*squared?\s*=\s*(?P<stat>{_NUM})\s*,\s*df\s*=\s*(?P<df1>{_DF})", re.I)),
    ("F", re.compile(rf"(?<![A-Za-z])F\s*=\s*(?P<stat>{_NUM})\s*,\s*num\s*df\s*=\s*(?P<df1>{_DF})\s*,\s*denom\s*df\s*=\s*(?P<df2>{_DF})")),
    ("F", re.compile(rf"(?<![A-Za-z])F\s*=\s*(?P<stat>{_NUM})\s*,\s*df\s*=\s*(?P<df1>{_DF})\s*(?:,|and)\s*(?P<df2>{_DF})")),
    # APA styles
    ("t", re.compile(rf"(?<![A-Za-z])t{_OPEN}(?P<df1>{_DF}){_CLOSE}\s*{_OP}\s*(?P<stat>{_NUM})")),
    ("F", re.compile(rf"(?<![A-Za-z])F{_OPEN}(?P<df1>{_DF}){_SEP}(?P<df2>{_DF}){_CLOSE}\s*{_OP}\s*(?P<stat>{_NUM})")),
    ("r", re.compile(rf"(?<![A-Za-z])r{_OPEN}(?P<df1>{_DF}){_CLOSE}\s*{_OP}\s*(?P<stat>{_NUM})")),
    ("chi2", re.compile(
        rf"(?:(?:χ|(?<![A-Za-z])chi)\s*(?:\^?\s*2|-?\s*squared?|-?\s*square)?|(?<![A-Za-z])X\s*(?:\^?\s*2|-?\s*squared?))"
        rf"{_OPEN}(?P<df1>{_DF})(?:{_SEP}N\s*=\s*(?P<n>\d[\d,]*))?{_CLOSE}\s*{_OP}\s*(?P<stat>{_NUM})", re.I)),
    ("Q", re.compile(rf"(?<![A-Za-z])Q\s*(?:w|b|within|between)?{_OPEN}(?P<df1>{_DF}){_CLOSE}\s*{_OP}\s*(?P<stat>{_NUM})")),
    ("z", re.compile(rf"(?<![A-Za-z])[zZ]\s*{_OP}\s*(?P<stat>{_NUM})(?![\d.])")),
]
# The op group is the only unnamed group in the APA patterns.
_P_RE = re.compile(rf"(?<![A-Za-z])p\s*s?\s*(?:-?\s*values?)?\s*{_OP}\s*(?P<p>{_SCI})", re.I)
_NS_RE = re.compile(r"(?<![A-Za-z])(?:n\.\s?s\.?|ns|n\.s|not\s+significant|non-?significant)(?![A-Za-z])", re.I)
_ONE_TAILED_RE = re.compile(r"one[- ]?(?:tailed|sided)|directional\s+(?:test|hypothes)", re.I)
_ADJUSTED_RE = re.compile(
    r"bonferroni|holm|šidák|sidak|tukey|benjamini|fdr|adjusted|corrected|false discovery", re.I
)
#: How far after the statistic a p-value may appear.
P_WINDOW = 90


def _op_of(m: re.Match[str]) -> str:
    for g in m.groups():
        if g in ("=", "<", ">", "<=", ">=", "=<", "=>"):
            return {"=<": "<=", "=>": ">="}.get(g, g)
    return "="


@dataclass
class ReportedResult:
    """One result found in the text, with its recomputed p-value and status."""

    kind: str  # t | F | r | chi2 | z | Q
    text: str
    line: int
    column: int
    df1: Num | None
    df2: Num | None
    stat_op: str
    stat: Num
    p_op: str | None  # "=", "<", ">", "<=", ">=" or "ns"
    p: Num | None
    n: int | None = None
    tails: int = 2
    adjusted: bool = False
    page: int | None = None
    computed_p: float | None = None
    computed_range: tuple[float, float] | None = None
    status: str = CONSISTENT
    message: str = ""
    notes: list[str] = field(default_factory=list)
    reported_significant: bool | None = None
    computed_significant: bool | None = None

    @property
    def label(self) -> str:
        sym = {"chi2": "χ²"}.get(self.kind, self.kind)
        df = ""
        if self.df1 is not None:
            df = f"({self.df1.text}" + (f", {self.df2.text}" if self.df2 is not None else "") + ")"
        stat = f"{sym}{df} {self.stat_op} {self.stat.text}"
        if self.p_op == "ns":
            return f"{stat}, ns"
        if self.p_op and self.p is not None:
            return f"{stat}, p {self.p_op} {self.p.text}"
        return stat

    @property
    def location(self) -> str:
        if self.page:
            return f"page {self.page}, line {self.line}"
        return f"line {self.line}"

    def to_dict(self) -> dict[str, Any]:
        def num(n: Num | None) -> Any:
            return None if n is None else n.value

        return {
            "kind": self.kind,
            "text": self.text,
            "label": self.label,
            "line": self.line,
            "column": self.column,
            "page": self.page,
            "df1": num(self.df1),
            "df2": num(self.df2),
            "statistic_op": self.stat_op,
            "statistic": self.stat.value,
            "statistic_text": self.stat.text,
            "p_op": self.p_op,
            "p": num(self.p),
            "p_text": self.p.text if self.p else None,
            "n": self.n,
            "tails": self.tails,
            "adjusted": self.adjusted,
            "computed_p": _finite(self.computed_p),
            "computed_p_range": [_finite(x) for x in self.computed_range] if self.computed_range else None,
            "reported_significant": self.reported_significant,
            "computed_significant": self.computed_significant,
            "status": self.status,
            "message": self.message,
            "notes": list(self.notes),
        }


def _finite(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(x)


# --------------------------------------------------------------------------- p-values


def recompute_p(kind: str, stat: float, df1: float | None = None, df2: float | None = None, tails: int = 2) -> float:
    """p-value of a reported statistic: two-tailed (or one-tailed) for t, z and r; upper tail for F, χ² and Q."""
    if kind == "t":
        p = float(_st.t.sf(abs(stat), df1))
        return min(1.0, p * tails)
    if kind == "z":
        return min(1.0, float(_st.norm.sf(abs(stat))) * tails)
    if kind == "r":
        r = abs(stat)
        if r >= 1:
            return 0.0
        t = r * math.sqrt(df1 / (1 - r * r))
        return min(1.0, float(_st.t.sf(t, df1)) * tails)
    if kind == "F":
        return float(_st.f.sf(stat, df1, df2))
    if kind in ("chi2", "Q"):
        return float(_st.chi2.sf(stat, df1))
    raise ValueError(f"unknown statistic {kind!r}")


def _stat_interval(res: ReportedResult) -> tuple[float, float]:
    """Interval of statistic values consistent with how it was reported (|stat| for symmetric tests)."""
    lo, hi = res.stat.low, res.stat.high
    if res.stat_op in ("<", "<="):
        lo = -math.inf if res.kind in ("t", "z", "r") else 0.0
        hi = res.stat.value
    elif res.stat_op in (">", ">="):
        lo, hi = res.stat.value, math.inf
    if res.kind in ("t", "z", "r"):
        if lo <= 0 <= hi:
            lo, hi = 0.0, max(abs(lo), abs(hi))
        else:
            lo, hi = sorted((abs(lo), abs(hi)))
        if res.kind == "r":
            hi = min(hi, 1.0)
            lo = min(lo, 1.0)
    else:
        lo = max(lo, 0.0)
    return lo, hi


def _df_values(n: Num | None) -> list[float | None]:
    if n is None:
        return [None]
    if n.half >= 0.5:  # integer df: exact
        return [n.value]
    return [max(n.low, 1e-9), n.high]


def computed_range(res: ReportedResult, tails: int) -> tuple[float, float]:
    lo, hi = _stat_interval(res)
    values: list[float] = []
    for s in (lo, hi):
        if not math.isfinite(s):
            values.append(0.0)
            continue
        for d1 in _df_values(res.df1):
            for d2 in _df_values(res.df2):
                values.append(recompute_p(res.kind, s, d1, d2, tails))
    return min(values), max(values)


def _reported_interval(res: ReportedResult, alpha: float) -> tuple[float, float, bool, bool]:
    """(low, high, low_inclusive, high_inclusive) of p-values the report allows."""
    if res.p_op == "ns":
        return alpha, 1.0, False, True
    assert res.p is not None
    v = res.p.value
    if res.p_op == "=":
        lo = max(0.0, res.p.low)
        return lo, min(1.0, res.p.high), True, True
    if res.p_op == "<":
        return 0.0, v, True, False
    if res.p_op == "<=":
        return 0.0, v, True, True
    if res.p_op == ">":
        return v, 1.0, False, True
    return v, 1.0, True, True  # >=


def _overlaps(c_lo: float, c_hi: float, r: tuple[float, float, bool, bool]) -> bool:
    lo, hi, lo_inc, hi_inc = r
    # computed interval is closed; reported has the given inclusivity
    if c_hi < lo or (c_hi == lo and not lo_inc):
        return False
    if c_lo > hi or (c_lo == hi and not hi_inc):
        return False
    return True


def _reported_sig(res: ReportedResult, alpha: float) -> bool | None:
    if res.p_op == "ns":
        return False
    if res.p is None:
        return None
    v = res.p.value
    if res.p_op in ("=", "<=", "<"):
        if v <= alpha:
            return True
        return False if res.p_op == "=" else None
    # ">" / ">="
    return False if v >= alpha else None


def _impossible(res: ReportedResult) -> str | None:
    if res.p is not None and res.p_op != "ns":
        if res.p.value > 1:
            return f"p = {res.p.text} is greater than 1"
        if res.p.value < 0:
            return f"p = {res.p.text} is negative"
        if res.p_op == "=" and res.p.value == 0 and res.p.half >= 0.05:
            return "p cannot be exactly 0"
        if res.p_op in ("<", "<=") and res.p.value <= 0:
            return f"p {res.p_op} {res.p.text} is impossible"
    for name, d in (("df", res.df1), ("df", res.df2)):
        if d is not None and d.value <= 0:
            return f"degrees of freedom must be positive ({name} = {d.text})"
    if res.kind == "r" and abs(res.stat.value) > 1:
        return f"a correlation must lie between -1 and 1 (r = {res.stat.text})"
    if res.kind in ("F", "chi2", "Q") and res.stat.value < 0:
        sym = {"chi2": "χ²"}.get(res.kind, res.kind)
        return f"{sym} cannot be negative ({res.stat.text})"
    return None


def evaluate(res: ReportedResult, alpha: float = 0.05, one_tailed_in_document: bool = False) -> ReportedResult:
    """Recompute the p-value of ``res`` and set its status and message (in place)."""
    why = _impossible(res)
    if why:
        res.status = IMPOSSIBLE
        res.message = f"Impossible value: {why}."
        return res
    if res.p_op == "=" and res.p is not None and res.p.value == 0:
        res.notes.append(f"p = {res.p.text} should be reported as p < .001")
    rep = _reported_interval(res, alpha)
    res.reported_significant = _reported_sig(res, alpha)
    symmetric = res.kind in ("t", "z", "r")
    tails = res.tails if symmetric else 1
    c_lo, c_hi = computed_range(res, tails if symmetric else 1)
    point = recompute_p(
        res.kind,
        res.stat.value if res.stat_op == "=" else _stat_interval(res)[0 if res.stat_op in (">", ">=") else 1],
        res.df1.value if res.df1 else None,
        res.df2.value if res.df2 else None,
        tails if symmetric else 1,
    ) if math.isfinite(res.stat.value) else None
    res.computed_p, res.computed_range = point, (c_lo, c_hi)
    if c_hi < alpha:
        res.computed_significant = True
    elif c_lo >= alpha:
        res.computed_significant = False
    else:
        res.computed_significant = None if point is None else bool(point < alpha)

    if res.adjusted:
        # A corrected p-value can only be larger than the raw one.
        if res.p_op in ("=", ">", ">=") and res.p is not None and rep[1] < c_lo:
            res.status = INCONSISTENT
            res.message = (
                f"Reported as a corrected p-value, but {res.p_op} {res.p.text} is smaller than the uncorrected "
                f"p = {_fmt(c_lo)} implied by the statistic."
            )
        else:
            res.status = CONSISTENT
            res.message = "Corrected p-value: only checked that it is not smaller than the uncorrected p."
        return res

    if _overlaps(c_lo, c_hi, rep):
        res.status = CONSISTENT
        res.message = f"Recomputed p = {_fmt(point)}."
        return res
    if symmetric and tails == 2 and one_tailed_in_document:
        o_lo, o_hi = computed_range(res, 1)
        if _overlaps(o_lo, o_hi, rep):
            res.status = CONSISTENT
            res.tails = 1
            res.computed_p = recompute_p(res.kind, res.stat.value, res.df1.value if res.df1 else None, None, 1)
            res.computed_range = (o_lo, o_hi)
            res.notes.append("consistent only as a one-tailed test (the document mentions one-tailed tests)")
            res.message = f"Consistent as a one-tailed test: recomputed one-tailed p = {_fmt(res.computed_p)}."
            return res
    reported = "ns" if res.p_op == "ns" else f"p {res.p_op} {res.p.text if res.p else '?'}"
    tail_txt = "one-tailed " if symmetric and tails == 1 else ""
    flip = (
        res.reported_significant is not None
        and ((res.reported_significant and c_lo >= alpha) or (not res.reported_significant and c_hi < alpha))
    )
    if flip:
        res.status = GROSS
        now = "not significant" if res.reported_significant else "significant"
        res.message = (
            f"Reported {reported}, but the statistic gives {tail_txt}p = {_fmt(point)}: "
            f"the result is {now} at α = {alpha:g}."
        )
    else:
        res.status = INCONSISTENT
        res.message = f"Reported {reported}, but the statistic gives {tail_txt}p = {_fmt(point)}."
    return res


def _fmt(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.2e}" if p > 0 else "0"
    return f"{p:.4f}".rstrip("0").rstrip(".") if p < 0.1 else f"{p:.3f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- extraction


def _spans_overlap(a: tuple[int, int], spans: Iterable[tuple[int, int]]) -> bool:
    return any(a[0] < b[1] and b[0] < a[1] for b in spans)


def _parse_df(text: str | None) -> Num | None:
    if text is None:
        return None
    return parse_num(text.replace(" ", ""))


def _sentence_window(norm: str, start: int, end: int) -> str:
    """The text from the previous sentence boundary to the next one, around [start, end)."""
    left = max(norm.rfind(". ", 0, start), norm.rfind("\n", 0, start))
    nxt = [i for i in (norm.find(". ", end), norm.find("\n", end)) if i != -1]
    right = min(nxt) if nxt else len(norm)
    return norm[left + 1 : right]


def _blocks(text: str, page_breaks: bool) -> Iterable[tuple[int, int, str, list[int]]]:
    """Yield ``(page, first_line, normalised_text, line_start_offsets)`` per paragraph.

    Lines of a paragraph are joined with spaces, because results are often
    wrapped across lines (``r(70) = -.28,`` / ``p = .017``). Blank lines and
    Markdown table rows (``| ... |``) end a paragraph; table rows stand alone.
    """
    pages = text.split("\f") if page_breaks else [text]
    for page_no, page_text in enumerate(pages, 1):
        lines = page_text.split("\n")
        buf: list[str] = []
        first = 1

        def flush() -> Iterable[tuple[int, int, str, list[int]]]:
            if buf:
                starts, parts, pos = [], [], 0
                for ln in buf:
                    starts.append(pos)
                    parts.append(ln)
                    pos += len(ln) + 1
                yield page_no, first, " ".join(parts), starts

        for i, raw in enumerate(lines, 1):
            norm = normalize(raw)
            if not norm.strip() or norm.lstrip().startswith("|"):
                yield from flush()
                buf = []
                if norm.strip():
                    yield page_no, i, norm, [0]
                first = i + 1
                continue
            if not buf:
                first = i
            buf.append(norm)
        yield from flush()


def extract_results(text: str, page_breaks: bool = False) -> list[ReportedResult]:
    """Find APA-style results with a p-value in ``text`` (statistics without a p-value are skipped)."""
    out: list[ReportedResult] = []
    for page, first_line, norm, starts_at in _blocks(text, page_breaks):
        taken: list[tuple[int, int]] = []
        found: list[tuple[int, str, re.Match[str]]] = []
        for kind, pat in _STAT_PATTERNS:
            for m in pat.finditer(norm):
                if _spans_overlap(m.span(), taken):
                    continue
                taken.append(m.span())
                found.append((m.start(), kind, m))
        found.sort(key=lambda x: x[0])
        starts = [f[0] for f in found]
        prev_end = 0
        for i, (start, kind, m) in enumerate(found):
            limit = min(starts[i + 1] if i + 1 < len(starts) else len(norm), m.end() + P_WINDOW)
            tail = norm[m.end() : limit]
            pm = _P_RE.search(tail)
            nm = _NS_RE.search(tail)
            p_op: str | None = None
            p: Num | None = None
            end = m.end()
            if pm and (not nm or pm.start() <= nm.start()):
                p_op = {"=<": "<=", "=>": ">="}.get(pm.group(1), pm.group(1))
                try:
                    p = parse_num(pm.group("p"))
                except ValueError:
                    continue
                end = m.end() + pm.end()
                gap = tail[: pm.start()]
            elif nm and nm.start() <= 30:
                p_op = "ns"
                end = m.end() + nm.end()
                gap = tail[: nm.start()]
            else:
                continue
            # Something else between the statistic and the p (another sentence) means they do not belong together.
            if re.search(r"[.!?]\s+[A-Z]", gap):
                continue
            try:
                stat = parse_num(m.group("stat"))
                df1 = _parse_df(m.groupdict().get("df1"))
                df2 = _parse_df(m.groupdict().get("df2"))
            except ValueError:
                continue
            n_txt = m.groupdict().get("n")
            sentence = _sentence_window(norm, start, end)
            boundary = max(prev_end, norm.rfind(". ", 0, start) + 1, start - 40, 0)
            before = norm[boundary:start]
            prev_end = end
            line_idx = bisect.bisect_right(starts_at, start) - 1
            res = ReportedResult(
                kind=kind,
                text=norm[start:end].strip(),
                line=first_line + line_idx,
                column=start - starts_at[line_idx] + 1,
                df1=df1,
                df2=df2,
                stat_op=_op_of(m),
                stat=stat,
                p_op=p_op,
                p=p,
                n=int(n_txt.replace(",", "")) if n_txt else None,
                tails=1 if _ONE_TAILED_RE.search(sentence) else 2,
                adjusted=bool(_ADJUSTED_RE.search(before + norm[m.end():end])),
                page=page if page_breaks else None,
            )
            out.append(res)
    return out


@dataclass
class ReportCheck:
    """Result of checking one document."""

    source: str | None
    alpha: float
    results: list[ReportedResult]
    one_tailed_in_document: bool = False
    note: str = ""

    def counts(self) -> dict[str, int]:
        c = {s: 0 for s in STATUSES}
        for r in self.results:
            c[r.status] += 1
        return c

    @property
    def status(self) -> str:
        worst = CONSISTENT
        for r in self.results:
            if _STATUS_RANK[r.status] > _STATUS_RANK[worst]:
                worst = r.status
        return worst

    @property
    def problems(self) -> list[ReportedResult]:
        return [r for r in self.results if r.status != CONSISTENT]

    def to_dict(self) -> dict[str, Any]:
        from . import __version__

        return {
            "tool": "stataudit",
            "kind": "check-report",
            "version": __version__,
            "schema": 1,
            "source": self.source,
            "alpha": self.alpha,
            "status": self.status,
            "counts": self.counts(),
            "one_tailed_in_document": self.one_tailed_in_document,
            "note": self.note,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    def to_text(self, color: bool = False, verbose: bool = False) -> str:
        c = self.counts()
        colors = {CONSISTENT: "\033[32m", INCONSISTENT: "\033[33m", GROSS: "\033[31m", IMPOSSIBLE: "\033[31m"}
        reset = "\033[0m" if color else ""
        lines = [f"stataudit check-report — {self.source or '<text>'}", "─" * 72]
        head = _headline(self)
        lines.append(f"{STATUS_ICON[self.status]} {head}")
        lines.append(
            f"  {len(self.results)} result(s): {c[CONSISTENT]} consistent, {c[INCONSISTENT]} inconsistent, "
            f"{c[GROSS]} gross, {c[IMPOSSIBLE]} impossible; alpha = {self.alpha:g}"
        )
        if self.note:
            lines.append(f"  note: {self.note}")
        lines.append("")
        for r in self.results:
            if r.status == CONSISTENT and not verbose:
                continue
            col = colors[r.status] if color else ""
            lines.append(f"{col}{STATUS_ICON[r.status]} {STATUS_LABEL[r.status]}{reset}  {r.location}: {r.label}")
            lines.append(f"    {r.message}")
            for n in r.notes:
                lines.append(f"    note: {n}")
        if not verbose and c[CONSISTENT] and self.problems:
            lines.append(f"({c[CONSISTENT]} consistent result(s) not shown; use -v to list them)")
        elif not verbose and c[CONSISTENT] and not self.problems:
            lines.append("Every recognised result is consistent (use -v to list them).")
        return "\n".join(lines).rstrip() + "\n"

    def to_markdown(self) -> str:
        c = self.counts()
        out = ["# stataudit check-report", ""]
        if self.source:
            out.append(f"**Source:** `{self.source}`  ")
        out.append(f"**Overall:** {STATUS_ICON[self.status]} {_headline(self)}  ")
        out.append(
            f"**Results:** {len(self.results)} ({c[CONSISTENT]} consistent, {c[INCONSISTENT]} inconsistent, "
            f"{c[GROSS]} gross, {c[IMPOSSIBLE]} impossible) at α = {self.alpha:g}"
        )
        out.append("")
        if self.results:
            out.append("| | Where | Reported | Recomputed p | Status | Explanation |")
            out.append("|---|---|---|---|---|---|")
            for r in self.results:
                out.append(
                    f"| {STATUS_ICON[r.status]} | {r.location} | `{_md_code(r.label)}` | {_fmt(r.computed_p)} | "
                    f"{STATUS_LABEL[r.status]} | {_md(r.message + (' ' + '; '.join(r.notes) if r.notes else ''))} |"
                )
            out.append("")
        if self.note:
            out.append(f"*{_md(self.note)}*")
            out.append("")
        out.append("---")
        out.append(
            "*Recomputed from the reported statistic and degrees of freedom, allowing for rounding to the reported "
            "decimals. Results in tables, tests without degrees of freedom and corrected p-values are not fully checked.*"
        )
        return "\n".join(out) + "\n"

    def to_html(self) -> str:
        from importlib import resources

        css = resources.files("stataudit.reports").joinpath("templates", "_style.css").read_text(encoding="utf-8")
        cls = {CONSISTENT: "s-holds", INCONSISTENT: "s-fragile", GROSS: "s-does_not_hold", IMPOSSIBLE: "s-does_not_hold"}
        e = html.escape
        c = self.counts()
        rows = []
        for r in self.results:
            notes = "".join(f"<div class='muted'>{e(n)}</div>" for n in r.notes)
            rows.append(
                f"<tr><td><span class='pill {cls[r.status]}'>{e(STATUS_LABEL[r.status])}</span></td>"
                f"<td class='num'>{e(r.location)}</td><td><code>{e(r.label)}</code></td>"
                f"<td class='num'>{e(_fmt(r.computed_p))}</td><td>{e(r.message)}{notes}</td></tr>"
            )
        table = (
            "<div class='table-wrap'><table><thead><tr><th>Status</th><th>Where</th><th>Reported</th>"
            "<th>Recomputed p</th><th>Explanation</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else "<p class='muted'>No APA-style results with a p-value were found.</p>"
        )
        return (
            "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>stataudit check-report</title>\n<style>\n" + css + "\n</style>\n</head>\n<body>\n<main>\n"
            "<h1>stataudit check-report</h1>\n"
            + (f"<div class='muted'>Source: <code>{e(self.source)}</code></div>\n" if self.source else "")
            + f"<div class='banner {cls[self.status]}'><strong>{STATUS_ICON[self.status]} {e(_headline(self))}</strong>"
            f"<div class='muted'>{len(self.results)} result(s): {c[CONSISTENT]} consistent, {c[INCONSISTENT]} "
            f"inconsistent, {c[GROSS]} gross, {c[IMPOSSIBLE]} impossible · α = {self.alpha:g}</div></div>\n"
            f"<div class='card'>{table}</div>\n"
            + (f"<p class='muted'>{e(self.note)}</p>\n" if self.note else "")
            + "<footer>Recomputed from the reported statistic and degrees of freedom, allowing for rounding to the "
            "reported decimals. Results in tables, tests without degrees of freedom and corrected p-values are not "
            "fully checked.</footer>\n</main>\n</body>\n</html>\n"
        )


def _headline(rc: ReportCheck) -> str:
    if not rc.results:
        return "No APA-style results found"
    return {
        CONSISTENT: "All reported p-values are consistent",
        INCONSISTENT: "Some reported p-values are inconsistent",
        GROSS: "A reported p-value flips significance",
        IMPOSSIBLE: "The report contains impossible values",
    }[rc.status]


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _md_code(s: str) -> str:
    return s.replace("`", "'").replace("|", "\\|")


def check_text(text: str, alpha: float = 0.05, source: str | None = None, page_breaks: bool = False) -> ReportCheck:
    """Extract and check every APA-style result in ``text``."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    text = text[:MAX_DOCUMENT_CHARS]
    one_tailed = bool(_ONE_TAILED_RE.search(normalize(text)))
    results = extract_results(text, page_breaks=page_breaks)
    for r in results:
        evaluate(r, alpha, one_tailed_in_document=one_tailed)
    return ReportCheck(source=source, alpha=alpha, results=results, one_tailed_in_document=one_tailed)


# --------------------------------------------------------------------------- documents


class _HTMLText(HTMLParser):
    """Visible text of an HTML document; block elements become line breaks."""

    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article",
              "blockquote", "pre", "ul", "ol", "dd", "dt", "figcaption", "caption", "header", "footer"}
    _SKIP = {"script", "style", "head", "title", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data.replace("\n", " "))

    def text(self) -> str:
        s = "".join(self.parts)
        return re.sub(r"\n[ \t]*\n+", "\n", s)


def html_to_text(source: str) -> str:
    parser = _HTMLText()
    parser.feed(source)
    parser.close()
    return parser.text()


def read_document(path: str | Path) -> tuple[str, bool, str]:
    """Return ``(text, page_breaks, note)`` for a .md/.txt/.rst/.html/.pdf file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        exe = shutil.which("pdftotext")
        if not exe:
            raise ValueError("reading PDF files needs the `pdftotext` program (poppler-utils)")
        proc = subprocess.run([exe, "-enc", "UTF-8", str(p), "-"], capture_output=True, timeout=PDF_TIMEOUT,
                              stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise ValueError(f"pdftotext failed: {proc.stderr.decode('utf-8', 'replace')[:300]}")
        text = proc.stdout.decode("utf-8", "replace")
        # pdftotext breaks lines inside sentences; join hyphen-less wraps within a paragraph.
        return text, True, "PDF text extracted with pdftotext; line numbers are per page."
    raw = p.read_bytes()[: MAX_DOCUMENT_CHARS * 4].decode("utf-8", "replace")
    if suffix in (".html", ".htm", ".xhtml"):
        return html_to_text(raw), False, "HTML converted to text; line numbers refer to the extracted text."
    return raw, False, ""


def check_file(path: str | Path, alpha: float = 0.05) -> ReportCheck:
    """Check one document on disk."""
    text, page_breaks, note = read_document(path)
    rc = check_text(text, alpha=alpha, source=str(path), page_breaks=page_breaks)
    rc.note = note
    return rc
