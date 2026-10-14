"""Notebooks: magic stripping, execution through the tracer, cell/line mapping."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner
from helpers import EXAMPLES, TempDirMixin

import stataudit
from stataudit.cli import main as cli
from stataudit.models import Status
from stataudit.notebook import (
    NotebookError,
    code_cells,
    load_notebook,
    notebook_to_percent,
    percent_to_cells,
    replace_cells,
    strip_magics,
    trace_notebook,
)


def write_nb(directory: str, cells: list[tuple[str, str]], name: str = "nb.ipynb") -> Path:
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in cells],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = Path(directory) / name
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


class TestStripMagics(unittest.TestCase):
    def test_line_magics_and_shell(self):
        src = "%matplotlib inline\nimport numpy as np\n!pip install x\nfiles = !ls\nx = %time np.ones(3)\n%timeit -n1 -r1 np.zeros(2)\nnp?\n"
        code, notes = strip_magics(src, 2)
        lines = code.split("\n")
        self.assertEqual(len(lines), len(src.split("\n")))  # line numbers preserved
        self.assertEqual(lines[0], "pass")
        self.assertEqual(lines[1], "import numpy as np")
        self.assertEqual(lines[2], "pass")
        self.assertEqual(lines[3], "files = None")
        self.assertEqual(lines[4], "x = np.ones(3)")
        self.assertEqual(lines[5], "np.zeros(2)")
        self.assertEqual(lines[6], "pass")
        compile(code, "x", "exec")
        self.assertEqual(len(notes), 6)
        self.assertTrue(all(n.startswith("cell 2") for n in notes))

    def test_indented_magic_keeps_block_valid(self):
        code, _ = strip_magics("for i in range(2):\n    %time print(i)\n    !echo hi\n")
        compile(code, "x", "exec")
        self.assertIn("    print(i)", code)

    def test_cell_magics(self):
        code, notes = strip_magics("%%time\nx = 1\ny = 2\n")
        self.assertEqual(code.split("\n")[1:3], ["x = 1", "y = 2"])
        self.assertIn("%%time", notes[0])
        code, notes = strip_magics("%%bash\necho hi\nls\n")
        self.assertEqual(code.strip(), "")
        self.assertIn("skipped", notes[0])

    def test_plain_python_untouched(self):
        src = "a = 10 % 3\nb = a != 2\ns = '%d' % a\nprint(s)\n"
        code, notes = strip_magics(src)
        self.assertEqual(code, src)
        self.assertEqual(notes, [])


class TestTraceNotebook(TempDirMixin, unittest.TestCase):
    def test_cell_and_line_mapping(self):
        path = write_nb(self.tmp, [
            ("markdown", "# title"),
            ("code", "%matplotlib inline\nimport numpy as np\nfrom scipy import stats"),
            ("markdown", "text"),
            ("code", "a = np.array([1.0, 2, 3, 4, 5, 6])\nb = a + 0.5\n\nres = stats.ttest_ind(a, b, equal_var=False)"),
            ("code", "display(res.pvalue)\nstats.pearsonr(a, b * 2)"),
        ])
        nt = trace_notebook(path)
        tr = nt.trace
        self.assertIsNone(tr.error, tr.stderr)
        self.assertEqual([r.func for r in tr.records], ["ttest_ind", "pearsonr"])
        s1, s2 = tr.records[0].site, tr.records[1].site
        self.assertEqual((s1.cell, s1.lineno), (2, 4))   # code cells are counted, markdown cells are not
        self.assertEqual((s2.cell, s2.lineno), (3, 2))
        self.assertEqual(s1.short, "nb.ipynb cell 2, line 4")
        self.assertIn("stats.ttest_ind(a, b", s1.code)
        self.assertEqual(Path(s1.filename), path.resolve())
        self.assertTrue(any("%matplotlib" in n for n in nt.notes))
        self.assertIn("0.", tr.stdout)  # display() printed without IPython

    def test_runs_in_notebook_directory_with_shared_namespace(self):
        (Path(self.tmp) / "data.csv").write_text("x\n1\n2\n3\n4\n5\n", encoding="utf-8")
        path = write_nb(self.tmp, [
            ("code", "import pandas as pd\ndf = pd.read_csv('data.csv')"),
            ("code", "from scipy import stats\nstats.ttest_1samp(df['x'], 0.0)\nassert __name__ == '__main__'"),
        ])
        tr = trace_notebook(path).trace
        self.assertIsNone(tr.error, tr.error)
        self.assertEqual(len(tr.records), 1)

    def test_error_stops_and_names_the_cell(self):
        path = write_nb(self.tmp, [
            ("code", "from scipy import stats\nstats.ttest_1samp([1.0, 2, 3, 4], 0)"),
            ("code", "x = 1\nraise ValueError('boom')"),
            ("code", "stats.ttest_1samp([1.0, 2, 3, 5], 0)"),
        ])
        tr = trace_notebook(path).trace
        self.assertEqual(len(tr.records), 1)
        self.assertIn("boom", tr.error)
        self.assertIn("cell 2, line 2", tr.error)

    def test_audit_example_notebook(self):
        rep = stataudit.audit_script(EXAMPLES / "06_notebook.ipynb")
        self.assertIsNone(rep.script_error, rep.script_error)
        (res,) = rep.results
        self.assertIs(res.status, Status.FAILS)
        self.assertEqual(res.location, "06_notebook.ipynb cell 4, line 2")
        self.assertTrue(any(f.code == "notebook_syntax" for f in rep.script_findings))
        self.assertEqual(rep.to_dict()["results"][0]["location"], "06_notebook.ipynb cell 4, line 2")

    def test_cli_run_notebook(self):
        res = CliRunner().invoke(cli, ["run", str(EXAMPLES / "06_notebook.ipynb")])
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("cell 4, line 2", res.output)
        self.assertIn("IPython-only", res.output)

    def test_bad_notebook(self):
        bad = Path(self.tmp) / "bad.ipynb"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaises(NotebookError):
            load_notebook(bad)
        res = CliRunner().invoke(cli, ["run", str(bad)])
        self.assertEqual(res.exit_code, 2)


class TestPercentFormat(TempDirMixin, unittest.TestCase):
    def test_round_trip_and_replace(self):
        path = write_nb(self.tmp, [("markdown", "m"), ("code", "a = 1\nb = 2"), ("code", "%time c = a + b")])
        nb = load_notebook(path)
        text = notebook_to_percent(nb)
        self.assertIn("# %% [cell 1]\na = 1\nb = 2", text)
        cells = percent_to_cells(text)
        self.assertEqual(cells, {1: "a = 1\nb = 2", 2: "%time c = a + b"})
        new = replace_cells(nb, {1: "a = 10\nb = 2", 2: "%time c = a + b"})
        srcs = [c.source for c in code_cells(new)]
        self.assertEqual(srcs, ["a = 10\nb = 2", "%time c = a + b"])
        self.assertEqual(new["cells"][0]["source"], ["m"])  # markdown untouched
        with self.assertRaises(ValueError):
            percent_to_cells("# %% [cell 1]\nx\n# %% [cell 1]\ny")
        with self.assertRaises(ValueError):
            percent_to_cells("no markers")


if __name__ == "__main__":
    unittest.main()
