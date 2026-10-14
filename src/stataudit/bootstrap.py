"""Subprocess entry point used by :func:`stataudit.trace.trace_script` and notebook tracing.

Usage (internal)::

    python -m stataudit.bootstrap OUT_DIR SCRIPT [ARGS...]
    python -m stataudit.bootstrap OUT_DIR --notebook CELLS_JSON NOTEBOOK_PATH

Installs the recording wrappers, then runs SCRIPT with :func:`runpy.run_path`
as ``__main__``, or executes the notebook's (already magic-stripped) code
cells in order in one shared ``__main__`` namespace, as Jupyter's "Run all"
does. Each cell is compiled under the file name ``<stataudit-cell-N>`` so that
call sites map directly to ``cell N, line M``; execution stops at the first
cell that raises. Every recorded call is flushed to OUT_DIR immediately. If
the code raises, the traceback is stored in ``OUT_DIR/status.json`` and the
calls recorded so far are kept.
"""

from __future__ import annotations

import builtins
import json
import linecache
import os
import runpy
import sys
import traceback
import types

CELL_FILENAME = "<stataudit-cell-{}>"


def _display(*objs: object, **_: object) -> None:
    """Stand-in for IPython's ``display`` when IPython is not installed."""
    for o in objs:
        print(o)


def _run_notebook(cells_json: str, nb_path: str) -> None:
    with open(cells_json, encoding="utf-8") as fh:
        cells = json.load(fh)
    nb_dir = os.path.dirname(os.path.abspath(nb_path))
    sys.argv = [nb_path]
    sys.path.insert(0, nb_dir)
    if not hasattr(builtins, "display"):
        try:
            from IPython.display import display as ipy_display  # type: ignore

            builtins.display = ipy_display  # type: ignore[attr-defined]
        except Exception:
            builtins.display = _display  # type: ignore[attr-defined]
    main = types.ModuleType("__main__")
    main.__dict__["__builtins__"] = builtins
    sys.modules["__main__"] = main
    for cell in cells:
        n = int(cell["index"])
        code = str(cell["code"])
        fname = CELL_FILENAME.format(n)
        lines = code.splitlines(keepends=True)
        linecache.cache[fname] = (len(code), None, lines, fname)
        compiled = compile(code, fname, "exec", dont_inherit=True)
        exec(compiled, main.__dict__)  # noqa: S102 - running the user's own notebook is the point


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: python -m stataudit.bootstrap OUT_DIR SCRIPT [ARGS...]", file=sys.stderr)
        return 2
    out_dir, *rest = argv
    notebook = rest[0] == "--notebook"
    if notebook and len(rest) != 3:
        print("usage: python -m stataudit.bootstrap OUT_DIR --notebook CELLS_JSON NOTEBOOK", file=sys.stderr)
        return 2
    from stataudit import trace

    def sink(rec: "trace.CallRecord") -> None:  # type: ignore[name-defined]
        try:
            trace.save_record(rec, out_dir)
        except Exception as exc:  # never break the user's script
            print(f"[stataudit] could not save record: {exc!r}", file=sys.stderr)

    trace.install()
    trace.add_sink(sink)

    status: dict[str, object] = {"error": None, "exit_code": 0}
    code = 0
    try:
        if notebook:
            _run_notebook(rest[1], rest[2])
        else:
            script, *script_args = rest
            script = os.path.abspath(script)
            sys.argv = [script, *script_args]
            sys.path.insert(0, os.path.dirname(script))
            runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        c = exc.code
        code = c if isinstance(c, int) else (0 if c is None else 1)
        status["exit_code"] = code
        if code != 0:
            status["error"] = f"script called sys.exit({c!r})"
    except BaseException as exc:  # noqa: BLE001 - we want everything, incl. KeyboardInterrupt
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        # drop bootstrap/runpy frames from the top of the traceback for readability
        status["error"] = "".join(tb[-12:])
        status["exit_code"] = 1
        code = 1
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    finally:
        try:
            with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as fh:
                json.dump(status, fh)
        except Exception:
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
