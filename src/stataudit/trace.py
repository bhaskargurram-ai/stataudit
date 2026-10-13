"""Tracing: record the statistical calls an analysis makes.

Two entry points:

* :func:`trace_script` runs a Python script in a subprocess. A bootstrap
  (:mod:`stataudit.bootstrap`) wraps selected ``scipy.stats`` functions before
  the script runs, and every call is written to a temporary directory as it
  happens, so a crash or a timeout still leaves a partial trace.
* :func:`recording` is an in-process context manager for notebooks::

      with stataudit.recording() as calls:
          stats.ttest_ind(a, b)
      stataudit.report(calls)

Wrappers always return the original result object unchanged. Any failure
while *recording* is logged and swallowed; it never reaches the user's code.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import linecache
import logging
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import scipy.stats as _stats

from ._scipy import ORIGINAL_ATTR
from .models import CallRecord, CallSite

log = logging.getLogger("stataudit.trace")

__all__ = [
    "TRACED_FUNCTIONS",
    "DATA_PARAMS",
    "build_record",
    "install",
    "uninstall",
    "recording",
    "Recording",
    "TraceResult",
    "trace_script",
    "save_record",
    "load_records",
]

#: scipy.stats functions that are wrapped, and which of their parameters hold data.
DATA_PARAMS: dict[str, tuple[str, ...]] = {
    "ttest_ind": ("a", "b"),
    "ttest_rel": ("a", "b"),
    "ttest_1samp": ("a", "popmean"),
    "pearsonr": ("x", "y"),
    "spearmanr": ("a", "b"),
    "f_oneway": ("samples",),
    "chi2_contingency": ("observed",),
    "mannwhitneyu": ("x", "y"),
    "wilcoxon": ("x", "y"),
    "kruskal": ("samples",),
    "linregress": ("x", "y"),
    "shapiro": ("x",),
    "fisher_exact": ("table",),
}
TRACED_FUNCTIONS: tuple[str, ...] = tuple(DATA_PARAMS)

_RESULT_FIELDS = (
    "statistic",
    "pvalue",
    "slope",
    "intercept",
    "rvalue",
    "stderr",
    "intercept_stderr",
    "dof",
    "df",
)

# ---------------------------------------------------------------------------
# Building records
# ---------------------------------------------------------------------------


def _to_array(value: Any) -> np.ndarray | None:
    """Convert array-like data to a float ndarray, or None if not numeric."""
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.dtype == object or arr.dtype.kind in "USV":
            arr = arr.astype(float)
        elif arr.dtype.kind not in "biuf":
            return None
        return np.array(arr, dtype=float, copy=True)
    except Exception:
        return None


def _scalar_param(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return repr(value)[:200]


def _result_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0:
        return float(arr)
    if arr.size > 10000:
        return None
    return arr.tolist()


def extract_result(result: Any) -> dict[str, Any]:
    """Pull the numeric fields (statistic, pvalue, slope, ...) out of a scipy result."""
    out: dict[str, Any] = {}
    if result is None:
        return out
    for name in _RESULT_FIELDS:
        try:
            v = getattr(result, name)
        except Exception:
            continue
        rv = _result_value(v)
        if rv is not None:
            out[name] = rv
    is_named = hasattr(result, "_fields") or "pvalue" in out
    if not is_named and isinstance(result, tuple) and len(result) >= 2:
        # plain (statistic, pvalue) tuple from an older scipy
        s, p = _result_value(result[0]), _result_value(result[1])
        if s is not None and "statistic" not in out:
            out["statistic"] = s
        if p is not None and "pvalue" not in out:
            out["pvalue"] = p
    return out


def build_record(
    func: str,
    args: tuple | list = (),
    kwargs: dict[str, Any] | None = None,
    result: Any = None,
    site: CallSite | None = None,
    index: int = 0,
    signature_of: Callable[..., Any] | None = None,
) -> CallRecord:
    """Turn a ``(function name, args, kwargs, result)`` tuple into a :class:`CallRecord`."""
    kwargs = dict(kwargs or {})
    data_names = DATA_PARAMS.get(func, ())
    inputs: dict[str, Any] = {}
    params: dict[str, Any] = {}
    target = signature_of if signature_of is not None else getattr(_stats, func, None)
    target = getattr(target, ORIGINAL_ATTR, target)
    bound: dict[str, Any] | None = None
    var_pos: str | None = None
    if target is not None:
        try:
            sig = inspect.signature(target)
            ba = sig.bind_partial(*args, **kwargs)
            bound = dict(ba.arguments)
            for p in sig.parameters.values():
                if p.kind is inspect.Parameter.VAR_POSITIONAL:
                    var_pos = p.name
                if p.kind is inspect.Parameter.VAR_KEYWORD and p.name in bound:
                    bound.update(bound.pop(p.name))
        except Exception:
            bound = None
    if bound is None:
        bound = {f"arg{i}": a for i, a in enumerate(args)}
        if data_names and data_names[0] == "samples":
            var_pos = "samples"
            bound = {"samples": tuple(args), **kwargs}
        else:
            for name, a in zip(data_names, args):
                bound[name] = a
            bound.update(kwargs)
    for name, value in bound.items():
        if name == var_pos:
            arrs = [_to_array(v) for v in value]
            if all(a is not None for a in arrs):
                inputs[name] = arrs
            else:
                params[name] = "<non-numeric>"
            continue
        if name in data_names:
            if np.ndim(value) == 0 and value is not None and name == "popmean":
                params[name] = _scalar_param(value)
                continue
            arr = _to_array(value)
            if arr is not None:
                inputs[name] = arr
            elif value is not None:
                params[name] = "<non-numeric>"
            continue
        params[name] = _scalar_param(value)
    return CallRecord(index=index, func=func, inputs=inputs, params=params, result=extract_result(result), site=site)


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SKIP_DIRS = tuple(
    os.path.dirname(os.path.abspath(m.__file__)) + os.sep
    for m in (np, _stats)
    if getattr(m, "__file__", None)
)
_scipy_root = os.path.dirname(os.path.dirname(os.path.abspath(_stats.__file__))) + os.sep
_SKIP_DIRS = _SKIP_DIRS + (_scipy_root, _PKG_DIR + os.sep)
_SKIP_FILES = {os.path.abspath(inspect.getfile(contextlib)), os.path.abspath(inspect.getfile(functools))}


def _find_call_site(start_depth: int = 2) -> CallSite | None:
    """Return the first stack frame that belongs to user code."""
    try:
        frame = sys._getframe(start_depth)
    except ValueError:
        return None
    while frame is not None:
        fn = frame.f_code.co_filename
        absfn = os.path.abspath(fn) if not fn.startswith("<") else fn
        skip = (
            absfn.startswith(_SKIP_DIRS)
            or absfn in _SKIP_FILES
            or fn.startswith("<frozen")
            or os.path.basename(fn) in ("runpy.py",)
        )
        if not skip:
            line = linecache.getline(fn, frame.f_lineno).strip() if not fn.startswith("<frozen") else ""
            return CallSite(filename=absfn, lineno=frame.f_lineno, function=frame.f_code.co_name, code=line)
        frame = frame.f_back
    return None


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

_local = threading.local()
_lock = threading.RLock()
_sinks: list[Callable[[CallRecord], None]] = []
_install_count = 0
_counter = 0
_originals: dict[str, Callable[..., Any]] = {}


def _busy() -> bool:
    return getattr(_local, "depth", 0) > 0


def _emit(func_name: str, original: Callable[..., Any], args: tuple, kwargs: dict, result: Any) -> None:
    global _counter
    try:
        site = _find_call_site(3)
        with _lock:
            idx = _counter
            _counter += 1
            sinks = list(_sinks)
        rec = build_record(func_name, args, kwargs, result, site=site, index=idx, signature_of=original)
        for sink in sinks:
            try:
                sink(rec)
            except Exception:  # pragma: no cover - defensive
                log.exception("stataudit: sink failed for %s", func_name)
    except Exception:  # pragma: no cover - defensive
        log.exception("stataudit: failed to record call to %s", func_name)


def _make_wrapper(func_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _busy():
            return original(*args, **kwargs)
        _local.depth = getattr(_local, "depth", 0) + 1
        try:
            result = original(*args, **kwargs)
        finally:
            _local.depth -= 1
        _local.depth += 1
        try:
            _emit(func_name, original, args, kwargs, result)
        except Exception:  # pragma: no cover - _emit already guards
            pass
        finally:
            _local.depth -= 1
        return result

    setattr(wrapper, ORIGINAL_ATTR, original)
    wrapper.__stataudit_name__ = func_name  # type: ignore[attr-defined]
    return wrapper


def _patch_statsmodels() -> None:
    """Wrap statsmodels OLS fits if statsmodels is importable (optional path).

    ``statsmodels.formula.api.ols(...).fit()`` and ``statsmodels.api.OLS(...).fit()``
    both go through ``RegressionModel.fit``; we wrap ``OLS.fit`` and record the
    design matrix, response, coefficients and p-values. Fits that use weights
    or are not plain OLS are ignored by the auditor.
    """
    try:
        import importlib.util

        if importlib.util.find_spec("statsmodels") is None:
            return
        from statsmodels.regression import linear_model as lm  # type: ignore
    except Exception:
        return
    cls = lm.OLS
    original = cls.fit
    if hasattr(original, ORIGINAL_ATTR):
        return

    @functools.wraps(original)
    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _busy():
            return original(self, *args, **kwargs)
        _local.depth = getattr(_local, "depth", 0) + 1
        try:
            res = original(self, *args, **kwargs)
        finally:
            _local.depth -= 1
        _local.depth += 1
        try:
            global _counter
            site = _find_call_site(2)
            with _lock:
                idx = _counter
                _counter += 1
                sinks = list(_sinks)
            names = list(getattr(self, "exog_names", None) or [])
            rec = CallRecord(
                index=idx,
                func="statsmodels_ols",
                inputs={"X": np.array(self.exog, dtype=float), "y": np.array(self.endog, dtype=float)},
                params={
                    "exog_names": names,
                    "cov_type": _scalar_param(kwargs.get("cov_type", "nonrobust")),
                },
                result={
                    "params": np.asarray(res.params, dtype=float).tolist(),
                    "pvalues": np.asarray(res.pvalues, dtype=float).tolist(),
                    "bse": np.asarray(res.bse, dtype=float).tolist(),
                },
                site=site,
            )
            for sink in sinks:
                sink(rec)
        except Exception:
            log.exception("stataudit: failed to record statsmodels OLS fit")
        finally:
            _local.depth -= 1
        return res

    setattr(fit, ORIGINAL_ATTR, original)
    _originals["__statsmodels_ols_fit__"] = original
    cls.fit = fit


def _unpatch_statsmodels() -> None:
    original = _originals.pop("__statsmodels_ols_fit__", None)
    if original is None:
        return
    try:
        from statsmodels.regression import linear_model as lm  # type: ignore

        lm.OLS.fit = original
    except Exception:  # pragma: no cover
        pass


def install(statsmodels: bool = True) -> None:
    """Replace the traced ``scipy.stats`` functions with recording wrappers.

    Reference counted: nested installs are fine, each must be paired with
    :func:`uninstall`.
    """
    global _install_count
    with _lock:
        _install_count += 1
        if _install_count > 1:
            return
        for name in TRACED_FUNCTIONS:
            current = getattr(_stats, name, None)
            if current is None or hasattr(current, ORIGINAL_ATTR):
                continue
            _originals[name] = current
            setattr(_stats, name, _make_wrapper(name, current))
        if statsmodels:
            _patch_statsmodels()


def uninstall() -> None:
    """Undo :func:`install` once the last user is done."""
    global _install_count
    with _lock:
        if _install_count == 0:
            return
        _install_count -= 1
        if _install_count > 0:
            return
        for name in TRACED_FUNCTIONS:
            original = _originals.pop(name, None)
            if original is not None:
                setattr(_stats, name, original)
        _unpatch_statsmodels()


def add_sink(sink: Callable[[CallRecord], None]) -> None:
    with _lock:
        _sinks.append(sink)


def remove_sink(sink: Callable[[CallRecord], None]) -> None:
    with _lock:
        if sink in _sinks:
            _sinks.remove(sink)


class Recording(list):
    """A list of :class:`CallRecord` objects captured by :func:`recording`."""

    def report(self, alpha: float = 0.05) -> Any:
        """Audit the recorded calls and return a :class:`~stataudit.models.Report`."""
        from .audits import audit_records

        return audit_records(list(self), alpha=alpha, source="<recording>")


@contextlib.contextmanager
def recording(rebind_globals: bool = True) -> Iterator[Recording]:
    """Record scipy.stats calls made inside the ``with`` block.

    Names that the caller already imported with ``from scipy.stats import
    ttest_ind`` are temporarily rebound in the caller's module globals (and
    restored afterwards), so notebook cells that imported earlier are traced
    too.
    """
    calls = Recording()
    install()
    add_sink(calls.append)
    rebound: list[tuple[dict[str, Any], str, Any]] = []
    if rebind_globals:
        try:
            frame = sys._getframe(2)  # caller of the generator-based context manager
            g = frame.f_globals
            lookup = {id(orig_fn): getattr(_stats, name) for name, orig_fn in _originals.items() if name in DATA_PARAMS}
            for key, val in list(g.items()):
                wrapped = lookup.get(id(val))
                if wrapped is not None:
                    rebound.append((g, key, val))
                    g[key] = wrapped
        except Exception:  # pragma: no cover - best effort
            log.debug("stataudit: could not rebind caller globals", exc_info=True)
    try:
        yield calls
    finally:
        for g, key, val in rebound:
            if getattr(g.get(key), ORIGINAL_ATTR, None) is val:
                g[key] = val
        remove_sink(calls.append)
        uninstall()


# ---------------------------------------------------------------------------
# Serialization (used by the subprocess bootstrap)
# ---------------------------------------------------------------------------


def save_record(rec: CallRecord, out_dir: str | os.PathLike[str]) -> None:
    """Append one record to ``calls.jsonl`` and store its arrays in ``call_<i>.npz``."""
    out = Path(out_dir)
    arrays: dict[str, np.ndarray] = {}
    spec: dict[str, Any] = {}
    for name, val in rec.inputs.items():
        if isinstance(val, list):
            keys = []
            for j, a in enumerate(val):
                k = f"{name}__{j}"
                arrays[k] = np.asarray(a, dtype=float)
                keys.append(k)
            spec[name] = {"list": keys}
        else:
            arrays[name] = np.asarray(val, dtype=float)
            spec[name] = {"array": name}
    npz = out / f"call_{rec.index:05d}.npz"
    np.savez(npz, **arrays)
    meta = {
        "index": rec.index,
        "func": rec.func,
        "inputs": spec,
        "params": rec.params,
        "result": rec.result,
        "site": rec.site.to_dict() if rec.site else None,
        "npz": npz.name,
    }
    with open(out / "calls.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, default=repr) + "\n")
        fh.flush()


def load_records(out_dir: str | os.PathLike[str]) -> list[CallRecord]:
    """Load records written by :func:`save_record`. Corrupt lines are skipped."""
    out = Path(out_dir)
    path = out / "calls.jsonl"
    records: list[CallRecord] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            meta = json.loads(line)
            with np.load(out / meta["npz"], allow_pickle=False) as z:
                inputs: dict[str, Any] = {}
                for name, s in meta["inputs"].items():
                    if "list" in s:
                        inputs[name] = [np.array(z[k]) for k in s["list"]]
                    else:
                        inputs[name] = np.array(z[s["array"]])
            site = CallSite(**meta["site"]) if meta.get("site") else None
            records.append(
                CallRecord(
                    index=meta["index"],
                    func=meta["func"],
                    inputs=inputs,
                    params=meta.get("params", {}),
                    result=meta.get("result", {}),
                    site=site,
                )
            )
        except Exception:
            log.warning("stataudit: skipping unreadable trace record", exc_info=True)
    records.sort(key=lambda r: r.index)
    return records


# ---------------------------------------------------------------------------
# Running scripts
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    """Outcome of :func:`trace_script`."""

    script: str
    records: list[CallRecord] = field(default_factory=list)
    returncode: int | None = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def _src_root() -> str:
    return os.path.dirname(_PKG_DIR)


def trace_script(
    script: str | os.PathLike[str],
    args: list[str] | tuple[str, ...] = (),
    timeout: float | None = 300,
    cwd: str | os.PathLike[str] | None = None,
    python: str | None = None,
) -> TraceResult:
    """Run ``script`` in a fresh Python subprocess and record its statistical calls.

    The script runs exactly as ``python script.py args...`` would (``__name__``
    is ``"__main__"``, ``sys.argv`` is set, the script directory is on
    ``sys.path``). Only the given file is executed; no shell is involved.
    """
    script_path = Path(script).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"script not found: {script}")
    with tempfile.TemporaryDirectory(prefix="stataudit-") as tmp:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (_src_root(), env.get("PYTHONPATH", "")) if p)
        env.setdefault("MPLBACKEND", "Agg")
        env["PYTHONIOENCODING"] = env.get("PYTHONIOENCODING", "utf-8")
        cmd = [python or sys.executable, "-m", "stataudit.bootstrap", tmp, str(script_path), *map(str, args)]
        result = TraceResult(script=str(script_path))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            result.returncode = proc.returncode
            result.stdout, result.stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.error = f"script timed out after {timeout} s; showing calls recorded before the timeout"
            result.stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            result.stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        result.records = load_records(tmp)
        status_file = Path(tmp) / "status.json"
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
                if status.get("error"):
                    result.error = status["error"]
            except Exception:
                pass
        elif result.error is None and result.returncode not in (0, None):
            tail = "\n".join(result.stderr.strip().splitlines()[-15:])
            result.error = f"tracer exited with code {result.returncode}\n{tail}".strip()
    return result
