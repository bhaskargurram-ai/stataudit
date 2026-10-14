"""Jupyter notebooks: read code cells, strip IPython syntax, trace and map call sites to cells.

``stataudit run analysis.ipynb`` executes the notebook's code cells in order
(like "Run all") in a subprocess under the tracer, without Jupyter. Call sites
are reported as ``analysis.ipynb cell N, line M``, where **N counts code cells
only** (1-based, markdown cells are skipped) and M is the line inside the cell.

IPython-only syntax is not Python, so it is replaced before execution, keeping
every line in place (line numbers stay exact):

* line magics (``%matplotlib inline``) and shell lines (``!pip install x``)
  become ``pass``; ``%time expr`` / ``%timeit expr`` keep ``expr``;
* ``x = !ls`` / ``x = %sx ls`` become ``x = None``;
* cell magics: the body of ``%%time``, ``%%timeit``, ``%%capture`` is kept,
  other cells (``%%bash``, ``%%writefile``, ``%%html``, ...) are skipped;
* help lines (``obj?``) are dropped.

Each replacement is reported as a note. ``display()`` works without IPython
(it prints). The notebook file is read with :mod:`json`; nbformat is not
needed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import CallSite

__all__ = [
    "NotebookError",
    "Cell",
    "load_notebook",
    "code_cells",
    "strip_magics",
    "trace_notebook",
    "notebook_to_percent",
    "percent_to_cells",
    "replace_cells",
    "PERCENT_MARKER",
]

MAX_NOTEBOOK_BYTES = 50_000_000
#: Cell magics whose body is ordinary Python.
PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun", "debug"}
#: Line magics followed by a Python statement that should still run.
PYTHON_LINE_MAGICS = {"time", "timeit", "prun"}
PERCENT_MARKER = re.compile(r"^# %% \[cell (\d+)\][^\n]*$", re.M)


class NotebookError(ValueError):
    """The file is not a readable Jupyter notebook."""


@dataclass
class Cell:
    index: int  # 1-based among code cells
    position: int  # 1-based among all cells
    source: str
    cell_id: str | None = None


def load_notebook(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if p.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise NotebookError(f"{p} is larger than {MAX_NOTEBOOK_BYTES // 1_000_000} MB")
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookError(f"{p} is not a valid notebook (JSON): {exc}") from exc
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        if isinstance(nb, dict) and "worksheets" in nb:
            raise NotebookError(f"{p} uses notebook format 3; convert it with `jupyter nbconvert --to notebook`")
        raise NotebookError(f"{p} is not a Jupyter notebook (no 'cells' list)")
    return nb


def _source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(str(s) for s in src)
    return str(src)


def code_cells(nb: dict[str, Any]) -> list[Cell]:
    out: list[Cell] = []
    k = 0
    for pos, cell in enumerate(nb.get("cells") or [], 1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        k += 1
        out.append(Cell(index=k, position=pos, source=_source(cell), cell_id=cell.get("id")))
    return out


_ASSIGN_MAGIC = re.compile(r"^(\s*[A-Za-z_][\w.\[\], ]*=\s*)([!%])(.*)$")
_HELP = re.compile(r"^\s*%?[A-Za-z_][\w.]*\s*\?\??\s*$|^\s*\?\??[A-Za-z_][\w.]*\s*$")


def strip_magics(source: str, cell_index: int | None = None) -> tuple[str, list[str]]:
    """Replace IPython syntax in one cell with plain Python of the same line count.

    Returns ``(code, notes)``.
    """
    where = f"cell {cell_index}" if cell_index is not None else "cell"
    lines = source.split("\n")
    notes: list[str] = []
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is not None and lines[first].lstrip().startswith("%%"):
        name = lines[first].strip()[2:].split(None, 1)[0] if lines[first].strip()[2:] else ""
        if name in PYTHON_CELL_MAGICS:
            lines[first] = ""
            notes.append(f"{where}: removed the cell magic %%{name} (its Python body still runs)")
        else:
            notes.append(f"{where}: skipped (cell magic %%{name} is not Python)")
            return "\n".join("" for _ in lines), notes
    out: list[str] = []
    continued = False
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        indent = ln[: len(ln) - len(stripped)]
        if continued or not stripped:
            out.append(ln)
        elif stripped.startswith(("%", "!")):
            body = stripped[1:]
            name = body.split(None, 1)[0] if body else ""
            if stripped.startswith("%") and name in PYTHON_LINE_MAGICS and len(body.split(None, 1)) == 2:
                rest = body.split(None, 1)[1]
                # %timeit accepts options such as -n1 -r1 before the statement
                rest = re.sub(r"^(?:-[A-Za-z]\S*\s+)+", "", rest)
                out.append(indent + rest)
                notes.append(f"{where}, line {i + 1}: removed %{name} (the statement still runs)")
            else:
                out.append(indent + "pass")
                what = "shell command" if stripped.startswith("!") else f"magic %{name}"
                notes.append(f"{where}, line {i + 1}: skipped {what}: {stripped[:60]}")
        elif _ASSIGN_MAGIC.match(ln) and not re.match(r"^\s*[A-Za-z_][\w.\[\], ]*=\s*[!%]=", ln):
            m = _ASSIGN_MAGIC.match(ln)
            assert m is not None
            target, kind, rest = m.groups()
            name = rest.split(None, 1)[0] if rest else ""
            if kind == "%" and name in PYTHON_LINE_MAGICS and len(rest.split(None, 1)) == 2:
                out.append(target + rest.split(None, 1)[1])
                notes.append(f"{where}, line {i + 1}: removed %{name} (the expression still runs)")
            else:
                out.append(target + "None")
                notes.append(f"{where}, line {i + 1}: replaced IPython {'shell' if kind == '!' else 'magic'} "
                             f"assignment with None: {stripped[:60]}")
        elif _HELP.match(ln):
            out.append(indent + "pass")
            notes.append(f"{where}, line {i + 1}: skipped help request {stripped[:40]}")
        else:
            out.append(ln)
        continued = ln.rstrip().endswith("\\")
    return "\n".join(out), notes


@dataclass
class NotebookTrace:
    """A :class:`~stataudit.trace.TraceResult` plus notebook-specific notes."""

    trace: Any
    notes: list[str] = field(default_factory=list)
    cells: int = 0


def _map_site(site: CallSite | None, nb_path: str) -> CallSite | None:
    if site is None:
        return None
    m = re.fullmatch(r"<stataudit-cell-(\d+)>", site.filename or "")
    if not m:
        return site
    return CallSite(filename=nb_path, lineno=site.lineno, function="" if site.function == "<module>" else site.function,
                    code=site.code, cell=int(m.group(1)))


def _clean_error(err: str | None) -> str | None:
    if not err:
        return err
    err = re.sub(r'File "<stataudit-cell-(\d+)>", line (\d+)', r"cell \1, line \2", err)
    return err


def prepared_cells(nb: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    cells: list[dict[str, Any]] = []
    notes: list[str] = []
    for c in code_cells(nb):
        code, n = strip_magics(c.source, c.index)
        notes.extend(n)
        cells.append({"index": c.index, "code": code})
    return cells, notes


def trace_notebook(
    path: str | os.PathLike[str],
    timeout: float | None = 300,
    python: str | None = None,
    env: dict[str, str] | None = None,
    notebook: dict[str, Any] | None = None,
) -> NotebookTrace:
    """Execute the notebook's code cells under the tracer (cwd = the notebook's directory)."""
    from .trace import run_bootstrap

    nb_path = Path(path).resolve()
    nb = notebook if notebook is not None else load_notebook(nb_path)
    cells, notes = prepared_cells(nb)

    def prepare(tmp: str) -> list[str]:
        cj = os.path.join(tmp, "cells.json")
        with open(cj, "w", encoding="utf-8") as fh:
            json.dump(cells, fh)
        return ["--notebook", cj, str(nb_path)]

    tr = run_bootstrap([], str(nb_path), timeout=timeout, cwd=nb_path.parent, python=python, env=env, prepare=prepare)
    for rec in tr.records:
        rec.site = _map_site(rec.site, str(nb_path))
    tr.error = _clean_error(tr.error)
    return NotebookTrace(trace=tr, notes=notes, cells=len(cells))


# --------------------------------------------------------------------------- percent format (agent)


def notebook_to_percent(nb: dict[str, Any]) -> str:
    """The notebook's code cells as one ``# %% [cell N]`` script (what the agent reads and edits)."""
    parts = []
    for c in code_cells(nb):
        parts.append(f"# %% [cell {c.index}]\n{c.source.rstrip()}\n")
    return "\n".join(parts)


def percent_to_cells(text: str) -> dict[int, str]:
    """Parse ``# %% [cell N]`` sections back into ``{N: source}``. Raises ValueError on duplicates."""
    out: dict[int, str] = {}
    matches = list(PERCENT_MARKER.finditer(text))
    if not matches:
        raise ValueError("no '# %% [cell N]' markers found")
    for i, m in enumerate(matches):
        n = int(m.group(1))
        if n in out:
            raise ValueError(f"cell {n} appears twice")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() + 1 if m.end() < len(text) else m.end() : end]
        out[n] = body.rstrip("\n")
    return out


def replace_cells(nb: dict[str, Any], sources: dict[int, str]) -> dict[str, Any]:
    """A copy of ``nb`` whose code cells get the given sources; changed cells lose their outputs."""
    import copy

    new = copy.deepcopy(nb)
    k = 0
    for cell in new.get("cells") or []:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        k += 1
        if k in sources and sources[k] != _source(cell).rstrip("\n"):
            src = sources[k]
            lines = src.split("\n")
            cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if src else []
            cell["outputs"] = []
            cell["execution_count"] = None
    return new
