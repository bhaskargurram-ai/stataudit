"""Convenience functions for auditing without tracing a script."""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import numpy as np

from . import stats_extra as sx
from ._scipy import orig
from .audits import audit_records
from .models import CallRecord, CallSite, Report
from .trace import DATA_PARAMS, build_record, trace_script

__all__ = [
    "audit_call",
    "audit_script",
    "report",
    "audit_ttest",
    "audit_paired_ttest",
    "audit_1samp_ttest",
    "audit_pearson",
    "audit_anova",
    "audit_chi2",
    "audit_linregress",
    "audit_ols",
    "audit_data",
]


def _site(label: str | None) -> CallSite | None:
    return CallSite(filename=label, lineno=0) if label else None


def audit_call(
    name: str,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    result: Any = None,
    alpha: float = 0.05,
    label: str | None = None,
) -> Report:
    """Audit one call described by function name, arguments and (optionally) its result.

    If ``result`` is None the scipy function is called to obtain it::

        stataudit.audit_call("ttest_ind", [a, b])
    """
    if name not in DATA_PARAMS:
        raise ValueError(f"unsupported test {name!r}; choose from {', '.join(DATA_PARAMS)}")
    kwargs = dict(kwargs or {})
    if result is None:
        result = orig(name)(*args, **kwargs)
    rec = build_record(name, tuple(args), kwargs, result, site=_site(label))
    return audit_records([rec], alpha=alpha, source=label)


def report(calls: Iterable[CallRecord], alpha: float = 0.05, source: str | None = None) -> Report:
    """Audit calls captured with :func:`stataudit.recording`."""
    return audit_records(list(calls), alpha=alpha, source=source or "<recording>")


def audit_script(
    path: str | os.PathLike[str],
    args: Sequence[str] = (),
    alpha: float = 0.05,
    timeout: float | None = 300,
) -> Report:
    """Run a script under the tracer and audit every recorded call."""
    tr = trace_script(path, args=list(args), timeout=timeout)
    return audit_records(
        tr.records,
        alpha=alpha,
        source=str(path),
        script_error=tr.error,
        meta={"returncode": tr.returncode, "timed_out": tr.timed_out, "calls": len(tr.records)},
    )


def audit_ttest(a: Any, b: Any, equal_var: bool = True, alpha: float = 0.05, **kwargs: Any) -> Report:
    """Audit ``scipy.stats.ttest_ind(a, b, equal_var=...)``."""
    return audit_call("ttest_ind", [a, b], {"equal_var": equal_var, **kwargs}, alpha=alpha)


def audit_paired_ttest(a: Any, b: Any, alpha: float = 0.05, **kwargs: Any) -> Report:
    """Audit ``scipy.stats.ttest_rel(a, b)``."""
    return audit_call("ttest_rel", [a, b], kwargs, alpha=alpha)


def audit_1samp_ttest(a: Any, popmean: float = 0.0, alpha: float = 0.05, **kwargs: Any) -> Report:
    """Audit ``scipy.stats.ttest_1samp(a, popmean)``."""
    return audit_call("ttest_1samp", [a, popmean], kwargs, alpha=alpha)


def audit_pearson(x: Any, y: Any, alpha: float = 0.05, **kwargs: Any) -> Report:
    """Audit ``scipy.stats.pearsonr(x, y)``."""
    return audit_call("pearsonr", [x, y], kwargs, alpha=alpha)


def audit_anova(*groups: Any, alpha: float = 0.05) -> Report:
    """Audit ``scipy.stats.f_oneway(*groups)``."""
    return audit_call("f_oneway", list(groups), alpha=alpha)


def audit_chi2(table: Any, alpha: float = 0.05, correction: bool = True) -> Report:
    """Audit ``scipy.stats.chi2_contingency(table)``."""
    return audit_call("chi2_contingency", [np.asarray(table)], {"correction": correction}, alpha=alpha)


def audit_linregress(x: Any, y: Any, alpha: float = 0.05) -> Report:
    """Audit ``scipy.stats.linregress(x, y)``."""
    return audit_call("linregress", [x, y], alpha=alpha)


def audit_ols(
    X: Any,
    y: Any,
    names: Sequence[str] | None = None,
    add_constant: bool = True,
    alpha: float = 0.05,
) -> Report:
    """Fit OLS of ``y`` on ``X`` (intercept added by default) and audit every slope.

    ``X`` may be a 1-D array, 2-D array or a pandas DataFrame (column names are used).
    """
    if names is None and hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    names = list(names) if names is not None else [f"x{j + 1}" for j in range(Xa.shape[1])]
    if add_constant and not sx.has_constant(Xa):
        Xa = sx.add_constant(Xa)
        names = ["const", *names]
    ya = np.asarray(y, dtype=float).ravel()
    fit = sx.ols_fit(Xa, ya)
    rec = CallRecord(
        index=0,
        func="ols",
        inputs={"X": Xa, "y": ya},
        params={"exog_names": names, "cov_type": "nonrobust"},
        result={"params": fit.beta.tolist(), "pvalues": fit.pvalues.tolist()},
    )
    return audit_records([rec], alpha=alpha)


#: Tests supported by :func:`audit_data` and their data layout.
DATA_TESTS = {
    "ttest_ind": "two groups",
    "mannwhitneyu": "two groups",
    "f_oneway": "k groups",
    "kruskal": "k groups",
    "ttest_rel": "two paired columns",
    "wilcoxon": "two paired columns",
    "pearsonr": "two numeric columns",
    "spearmanr": "two numeric columns",
    "linregress": "two numeric columns (x, y)",
    "chi2_contingency": "two categorical columns",
    "fisher_exact": "two categorical columns (2x2)",
    "ttest_1samp": "one numeric column",
}


def audit_data(
    df: Any,
    test: str,
    value_col: str | None = None,
    group_col: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    alpha: float = 0.05,
    popmean: float = 0.0,
    equal_var: bool = True,
    source: str | None = None,
) -> Report:
    """Run ``test`` on columns of a pandas DataFrame and audit it.

    * two/k-group tests: ``value_col`` + ``group_col`` (groups in order of appearance)
    * paired / correlation / regression: ``x_col`` + ``y_col``
    * contingency tests: ``x_col`` + ``y_col`` (both categorical)
    * one-sample: ``value_col`` (+ ``popmean``)
    """
    import pandas as pd

    if test not in DATA_TESTS:
        raise ValueError(f"unsupported test {test!r}; choose from {', '.join(DATA_TESTS)}")

    def col(name: str | None, what: str) -> "pd.Series":
        if not name:
            raise ValueError(f"{test} needs {what}")
        if name not in df.columns:
            raise ValueError(f"column {name!r} not found; columns are {', '.join(map(str, df.columns))}")
        return df[name]

    kwargs: dict[str, Any] = {}
    label = source or "<data>"
    if test in ("ttest_ind", "mannwhitneyu", "f_oneway", "kruskal"):
        v = pd.to_numeric(col(value_col, "--value-col"), errors="coerce")
        g = col(group_col, "--group-col")
        sub = pd.DataFrame({"v": v, "g": g}).dropna()
        levels = list(dict.fromkeys(sub["g"].tolist()))
        arrays = [sub.loc[sub["g"] == lv, "v"].to_numpy(dtype=float) for lv in levels]
        if test in ("ttest_ind", "mannwhitneyu") and len(levels) != 2:
            raise ValueError(f"{test} needs exactly 2 groups in {group_col!r}, found {len(levels)}: {levels[:10]}")
        if len(levels) < 2:
            raise ValueError("need at least 2 groups")
        if test == "ttest_ind":
            kwargs["equal_var"] = equal_var
        code = f"{test}({', '.join(f'{group_col}={lv}' for lv in levels)})"  # display only
        args: list[Any] = arrays
    elif test in ("ttest_rel", "wilcoxon", "pearsonr", "spearmanr", "linregress"):
        x = pd.to_numeric(col(x_col, "--x-col"), errors="coerce")
        y = pd.to_numeric(col(y_col, "--y-col"), errors="coerce")
        sub = pd.DataFrame({"x": x, "y": y}).dropna()
        args = [sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float)]
        code = f"{test}({x_col}, {y_col})"
    elif test in ("chi2_contingency", "fisher_exact"):
        tab = pd.crosstab(col(x_col, "--x-col"), col(y_col, "--y-col"))
        if test == "fisher_exact" and tab.shape != (2, 2):
            raise ValueError(f"fisher_exact needs a 2x2 table, got {tab.shape[0]}x{tab.shape[1]}")
        args = [tab.to_numpy()]
        code = f"{test}(crosstab({x_col}, {y_col}))"
    else:  # ttest_1samp
        v = pd.to_numeric(col(value_col, "--value-col"), errors="coerce").dropna()
        args = [v.to_numpy(dtype=float), popmean]
        code = f"{test}({value_col}, {popmean})"
    result = orig(test)(*args, **kwargs)
    rec = build_record(test, tuple(args), kwargs, result, site=CallSite(filename=label, lineno=0, code=code))
    if test in ("ttest_ind", "mannwhitneyu", "f_oneway", "kruskal"):
        rec.params["_labels"] = [f"{group_col}={lv}" for lv in levels]
    elif test != "ttest_1samp" and test not in ("chi2_contingency", "fisher_exact"):
        rec.params["_labels"] = [str(x_col), str(y_col)]
    else:
        rec.params["_labels"] = [str(value_col or f"{x_col} x {y_col}")]
    return audit_records([rec], alpha=alpha, source=label)
