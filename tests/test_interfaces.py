"""CLI (click CliRunner), Flask web UI and MCP tool functions."""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner
from helpers import ROOT, SRC, TempDirMixin, write_script

from stataudit.cli import main as cli

GOOD = """
import numpy as np
from scipy import stats
r = np.random.default_rng(5)
a = r.normal(412, 35, 45)
b = r.normal(385, 33, 45)
stats.ttest_ind(a, b, equal_var=False)
"""

BAD = """
import numpy as np
from scipy import stats
r = np.random.default_rng(7)
control = r.normal(12.0, 1.5, 60).round(1)
treated = np.concatenate([r.normal(12.2, 1.5, 11), [21.5, 24.0, 26.5]]).round(1)
stats.ttest_ind(treated, control)
"""


def _csv(tmp: str) -> Path:
    r = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "g": ["a"] * 30 + ["b"] * 30,
            "y": np.r_[r.normal(0, 1, 30), r.normal(1, 1, 30)],
            "x": r.normal(size=60),
            "c1": r.choice(["u", "v"], 60),
            "c2": r.choice(["p", "q", "s"], 60),
        }
    )
    path = Path(tmp) / "data.csv"
    df.to_csv(path, index=False)
    return path


class TestCLI(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runner = CliRunner()

    def test_run_good_script_exit_0_and_writes_reports(self):
        script = write_script(self.tmp, GOOD)
        html, md, js = (os.path.join(self.tmp, f) for f in ("r.html", "r.md", "r.json"))
        res = self.runner.invoke(cli, ["run", str(script), "--html", html, "--md", md, "--json", js])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Conclusions hold", res.output)
        self.assertTrue(Path(html).read_text().startswith("<!doctype html>"))
        self.assertIn("# stataudit report", Path(md).read_text())
        self.assertEqual(json.loads(Path(js).read_text())["status"], "holds")

    def test_run_bad_script_exit_1_and_fail_on_never(self):
        script = write_script(self.tmp, BAD)
        res = self.runner.invoke(cli, ["run", str(script)])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("does NOT hold", res.output)
        res = self.runner.invoke(cli, ["run", str(script), "--fail-on", "never", "-q"])
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(res.output, "")

    def test_run_passes_script_args_after_double_dash(self):
        script = write_script(
            self.tmp,
            """
            import sys
            from scipy import stats
            stats.ttest_1samp([float(v) for v in sys.argv[1:]], 0.0)
            """,
        )
        res = self.runner.invoke(cli, ["run", str(script), "--alpha", "0.01", "--", "1", "2", "3", "4"])
        self.assertIn("alpha = 0.01", res.output)
        self.assertIn("One-sample t-test", res.output)

    def test_check_data_group_test(self):
        path = _csv(self.tmp)
        res = self.runner.invoke(cli, ["check-data", str(path), "--test", "ttest_ind", "--group-col", "g", "--value-col", "y"])
        self.assertIn(res.exit_code, (0, 1))
        self.assertIn("g=a vs g=b", res.output)

    def test_check_data_contingency_and_correlation(self):
        path = _csv(self.tmp)
        res = self.runner.invoke(cli, ["check-data", str(path), "--test", "chi2_contingency", "--x-col", "c1", "--y-col", "c2"])
        self.assertIn("Chi-square", res.output)
        res = self.runner.invoke(cli, ["check-data", str(path), "--test", "pearsonr", "--x-col", "x", "--y-col", "y"])
        self.assertIn("Pearson", res.output)

    def test_check_data_bad_column_is_usage_error(self):
        path = _csv(self.tmp)
        res = self.runner.invoke(cli, ["check-data", str(path), "--test", "ttest_ind", "--group-col", "nope", "--value-col", "y"])
        self.assertEqual(res.exit_code, 2)
        self.assertIn("not found", res.output)

    def test_python_dash_m_help(self):
        env = dict(os.environ, PYTHONPATH=str(SRC))
        out = subprocess.run([sys.executable, "-m", "stataudit", "--help"], capture_output=True, text=True, env=env, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        for cmd in ("run", "check-data", "ui", "mcp"):
            self.assertIn(cmd, out.stdout)


class TestWeb(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from stataudit.web import create_app

        self.client = create_app({"TESTING": True}).test_client()

    def test_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"<form", r.data)
        self.assertIn(b"ttest_ind", r.data)

    def test_upload_and_audit(self):
        data = _csv(self.tmp).read_bytes()
        r = self.client.post(
            "/audit",
            data={"file": (io.BytesIO(data), "data.csv"), "test": "ttest_ind", "group_col": "g", "value_col": "y"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 200, r.data[:500])
        self.assertIn(b"banner", r.data)
        self.assertIn(b"t-test", r.data)

    def test_bad_column_shows_error(self):
        data = _csv(self.tmp).read_bytes()
        r = self.client.post(
            "/audit",
            data={"file": (io.BytesIO(data), "data.csv"), "test": "ttest_ind", "group_col": "zzz", "value_col": "y"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"not found", r.data)

    def test_missing_file(self):
        r = self.client.post("/audit", data={"test": "ttest_ind"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)


class TestMCP(TempDirMixin, unittest.TestCase):
    def test_audit_test_tool(self):
        from stataudit.mcp_server import audit_test_tool

        r = np.random.default_rng(7)
        control = r.normal(12.0, 1.5, 60).round(1)
        treated = np.concatenate([r.normal(12.2, 1.5, 11), [21.5, 24.0, 26.5]]).round(1)
        md = audit_test_tool("ttest_ind", json.dumps({"a": treated.tolist(), "b": control.tolist()}), 0.05)
        self.assertIn("# stataudit report", md)
        self.assertIn("does NOT hold", md)

    def test_audit_test_tool_groups_and_tables(self):
        from stataudit.mcp_server import audit_test_tool

        md = audit_test_tool("f_oneway", json.dumps({"samples": [[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 9]]}))
        self.assertIn("ANOVA", md)
        md = audit_test_tool("chi2_contingency", json.dumps({"observed": [[8, 1], [2, 6]]}))
        self.assertIn("Fisher", md)
        md = audit_test_tool("ttest_1samp", json.dumps({"a": [1.0, 2, 3, 4, 5], "popmean": 1}))
        self.assertIn("One-sample", md)

    def test_audit_test_tool_validation(self):
        from stataudit.mcp_server import MAX_VALUES, audit_test_tool

        with self.assertRaises(ValueError):
            audit_test_tool("os.system", "{}")
        with self.assertRaises(ValueError):
            audit_test_tool("ttest_ind", "not json")
        with self.assertRaises(ValueError):
            audit_test_tool("ttest_ind", json.dumps({"a": ["1; rm -rf /"], "b": [1, 2]}))
        with self.assertRaises(ValueError):
            audit_test_tool("ttest_ind", json.dumps({"a": [0.0] * (MAX_VALUES + 1), "b": [1.0, 2.0]}))
        with self.assertRaises(ValueError):
            audit_test_tool("ttest_ind", json.dumps({"a": [1, 2], "b": [3, 4], "options": {"axis": 1}}))
        with self.assertRaises(ValueError):
            audit_test_tool("ttest_ind", json.dumps({"a": [1, 2, 3], "b": [3, 4, 5]}), alpha=2)

    def test_audit_script_tool(self):
        from stataudit.mcp_server import audit_script_tool

        script = write_script(self.tmp, GOOD)
        md = audit_script_tool(str(script), 0.05)
        self.assertIn("Conclusions hold", md)
        with self.assertRaises(ValueError):
            audit_script_tool(str(Path(self.tmp) / "data.csv"))
        with self.assertRaises(ValueError):
            audit_script_tool(str(Path(self.tmp) / "missing.py"))

    def test_server_registers_tools(self):
        from stataudit.mcp_server import build_server

        server = build_server()
        tools = asyncio.run(server.list_tools())
        self.assertEqual({t.name for t in tools}, {"audit_script", "audit_test"})
        result = asyncio.run(
            server.call_tool("audit_test", {"test_name": "pearsonr", "data_json": json.dumps({"x": [1, 2, 3, 4, 5], "y": [2, 1, 4, 3, 5]})})
        )
        text = result.content[0].text if hasattr(result, "content") else str(result)
        self.assertIn("Pearson", text)


if __name__ == "__main__":
    unittest.main()
