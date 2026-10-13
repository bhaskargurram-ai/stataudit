"""Subprocess entry point used by :func:`stataudit.trace.trace_script`.

Usage (internal)::

    python -m stataudit.bootstrap OUT_DIR SCRIPT [ARGS...]

Installs the recording wrappers, then runs SCRIPT with :func:`runpy.run_path`
as ``__main__``. Every recorded call is flushed to OUT_DIR immediately. If the
script raises, the traceback is stored in ``OUT_DIR/status.json`` and the
calls recorded so far are kept.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import traceback


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: python -m stataudit.bootstrap OUT_DIR SCRIPT [ARGS...]", file=sys.stderr)
        return 2
    out_dir, script, *script_args = argv
    from stataudit import trace

    def sink(rec: "trace.CallRecord") -> None:  # type: ignore[name-defined]
        try:
            trace.save_record(rec, out_dir)
        except Exception as exc:  # never break the user's script
            print(f"[stataudit] could not save record: {exc!r}", file=sys.stderr)

    trace.install()
    trace.add_sink(sink)

    status: dict[str, object] = {"error": None, "exit_code": 0}
    script = os.path.abspath(script)
    sys.argv = [script, *script_args]
    sys.path.insert(0, os.path.dirname(script))
    code = 0
    try:
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
