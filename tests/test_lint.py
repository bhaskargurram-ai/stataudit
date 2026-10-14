"""lint: every rule has a case it flags and a close case it does not."""

from __future__ import annotations

import json
import textwrap
import unittest
from pathlib import Path

from click.testing import CliRunner
from helpers import EXAMPLES, TempDirMixin

from stataudit.cli import main as cli
from stataudit.lint import RULES, lint_file, lint_source

HEADER = "import numpy as np\nfrom scipy import stats\n"


def rules(src: str, **kw) -> list[str]:
    return [f.rule for f in lint_source(HEADER + textwrap.dedent(src), "a.py", **kw).findings]


class TestRules(unittest.TestCase):
    def test_every_rule_documented(self):
        for r in RULES.values():
            self.assertRegex(r.id, r"^SA1\d\d$")
            self.assertTrue(r.explanation and r.fix and r.summary)
            self.assertIn(r.severity, ("warn", "info"))

    # SA101 ------------------------------------------------------------------
    def test_sa101_loop_without_correction(self):
        self.assertIn("SA101", rules("""
            for m in metrics:
                stats.ttest_ind(a[m], b[m], equal_var=False)
        """))

    def test_sa101_three_tests(self):
        self.assertIn("SA101", rules("""
            stats.ttest_ind(a, b, equal_var=False)
            stats.mannwhitneyu(c, d)
            stats.pearsonr(e, f)
            stats.spearmanr(e, f)
        """))

    def test_sa101_negative_with_correction(self):
        src = """
            from statsmodels.stats.multitest import multipletests
            ps = [stats.ttest_ind(a[m], b[m], equal_var=False).pvalue for m in metrics]
            reject, adj, _, _ = multipletests(ps, method="holm")
        """
        self.assertNotIn("SA101", rules(src))
        self.assertNotIn("SA101", rules("""
            ps = [stats.ttest_ind(a[m], b[m], equal_var=False).pvalue for m in metrics]
            adj = stats.false_discovery_control(ps)
        """))
        self.assertNotIn("SA101", rules("""
            alpha = 0.05 / len(metrics)
            for m in metrics:
                stats.ttest_ind(a[m], b[m], equal_var=False)
        """))

    def test_sa101_negative_two_tests(self):
        self.assertNotIn("SA101", rules("stats.ttest_ind(a, b, equal_var=False)\nstats.pearsonr(x, y)\n"))

    # SA102 ------------------------------------------------------------------
    def test_sa102_student_default_and_explicit(self):
        self.assertEqual(rules("stats.ttest_ind(a, b)\n"), ["SA102"])
        self.assertEqual(rules("stats.ttest_ind(a, b, equal_var=True)\n"), ["SA102"])
        self.assertEqual(rules("from scipy.stats import ttest_ind\nttest_ind(a, b)\n"), ["SA102"])

    def test_sa102_negative(self):
        self.assertEqual(rules("stats.ttest_ind(a, b, equal_var=False)\n"), [])
        self.assertEqual(rules("print(stats.levene(a, b))\nstats.ttest_ind(a, b)\n"), [])
        self.assertEqual(rules("stats.ttest_ind(a, b, trim=0.2)\n"), [])
        self.assertEqual(rules("stats.ttest_rel(a, b)\n"), [])

    # SA103 ------------------------------------------------------------------
    def test_sa103_pearson_alone(self):
        self.assertEqual(rules("r, p = stats.pearsonr(x, y)\n"), ["SA103"])

    def test_sa103_negative(self):
        self.assertEqual(rules("import matplotlib.pyplot as plt\nplt.scatter(x, y)\nstats.pearsonr(x, y)\n"), [])
        self.assertEqual(rules("stats.pearsonr(x, y)\nstats.spearmanr(x, y)\n"), [])
        self.assertEqual(rules("x = x[np.abs(stats.zscore(x)) < 3]\nstats.pearsonr(x, y)\n"), [])

    # SA104 ------------------------------------------------------------------
    def test_sa104_sparse_literal_table(self):
        found = lint_source(HEADER + "stats.chi2_contingency([[8, 1], [2, 1]])\n", "a.py").findings
        self.assertEqual([f.rule for f in found], ["SA104"])
        self.assertEqual(found[0].severity, "warn")
        self.assertIn("below 5", found[0].message)

    def test_sa104_table_by_name_and_computed_info(self):
        f = lint_source(HEADER + "t = np.array([[14, 6, 0], [1, 1, 1]])\nstats.chi2_contingency(t)\n").findings
        self.assertEqual([(x.rule, x.severity) for x in f], [("SA104", "warn")])
        f = lint_source(HEADER + "chi2, p, dof, exp = stats.chi2_contingency(pd.crosstab(a, b))\n").findings
        self.assertEqual([(x.rule, x.severity) for x in f], [("SA104", "info")])

    def test_sa104_negative(self):
        self.assertEqual(rules("stats.chi2_contingency([[80, 60], [50, 70]])\n"), [])
        self.assertEqual(rules("chi2, p, dof, exp = stats.chi2_contingency(tab)\nprint(exp.min())\n"), [])
        self.assertEqual(rules("res = stats.chi2_contingency(tab)\nprint(res.expected_freq)\n"), [])

    # SA105 ------------------------------------------------------------------
    def test_sa105_test_shopping(self):
        found = rules("""
            stats.ttest_ind(a, b, equal_var=False)
            stats.mannwhitneyu(a, b)
            stats.ks_2samp(b, a)
            from statsmodels.stats.multitest import multipletests
        """)
        self.assertIn("SA105", found)
        self.assertNotIn("SA101", found)

    def test_sa105_negative(self):
        self.assertNotIn("SA105", rules("""
            stats.ttest_ind(a, b, equal_var=False)
            stats.mannwhitneyu(c, d)
            stats.ks_2samp(e, f)
        """))
        self.assertNotIn("SA105", rules("stats.ttest_ind(a, b, equal_var=False)\nstats.mannwhitneyu(a, b)\n"))

    # SA106 ------------------------------------------------------------------
    def test_sa106_threshold_in_loop(self):
        found = lint_source(HEADER + textwrap.dedent("""
            for m in metrics:
                res = stats.ttest_ind(a[m], b[m], equal_var=False)
                if res.pvalue < 0.05:
                    print(m)
        """)).findings
        sa106 = [f for f in found if f.rule == "SA106"]
        self.assertEqual(len(sa106), 1)
        self.assertIn("res.pvalue < 0.05", sa106[0].message)

    def test_sa106_negative(self):
        # a single test compared with alpha is fine
        self.assertNotIn("SA106", rules("p = stats.ttest_ind(a, b, equal_var=False).pvalue\nif p < 0.05:\n    print(1)\n"))
        # corrected p-values compared with alpha are fine
        self.assertNotIn("SA106", rules("""
            ps = [stats.ttest_ind(a[m], b[m], equal_var=False).pvalue for m in metrics]
            adj = stats.false_discovery_control(ps)
            for p_adj in adj:
                if p_adj < 0.05:
                    print(p_adj)
        """))

    # SA107 ------------------------------------------------------------------
    def test_sa107_normality_pretest(self):
        self.assertIn("SA107", rules("""
            if stats.shapiro(a).pvalue > 0.05:
                res = stats.ttest_ind(a, b, equal_var=False)
            else:
                res = stats.mannwhitneyu(a, b)
        """))
        self.assertIn("SA107", rules("""
            w, p_norm = stats.shapiro(a)
            if p_norm > 0.05:
                res = stats.ttest_ind(a, b, equal_var=False)
        """))

    def test_sa107_negative(self):
        self.assertNotIn("SA107", rules("""
            print(stats.shapiro(a))
            if len(a) > 30:
                res = stats.ttest_ind(a, b, equal_var=False)
        """))


class TestBehaviour(TempDirMixin, unittest.TestCase):
    def test_suppression_comments(self):
        self.assertEqual(rules("stats.ttest_ind(a, b)  # stataudit: ignore\n"), [])
        self.assertEqual(rules("stats.ttest_ind(a, b)  # stataudit: ignore[SA102]\n"), [])
        self.assertEqual(rules("stats.ttest_ind(a, b)  # stataudit: ignore[SA103]\n"), ["SA102"])

    def test_select_and_ignore(self):
        src = "stats.ttest_ind(a, b)\nstats.pearsonr(x, y)\n"
        self.assertEqual(rules(src), ["SA102", "SA103"])
        self.assertEqual(rules(src, select=["SA103"]), ["SA103"])
        self.assertEqual(rules(src, ignore=["sa103"]), ["SA102"])

    def test_syntax_error_is_a_note(self):
        rep = lint_source("def (:\n", "bad.py")
        self.assertEqual(rep.findings, [])
        self.assertTrue(rep.notes)

    def test_locally_defined_function_is_not_a_test(self):
        self.assertEqual(rules("def pearsonr(a, b):\n    return 0\npearsonr(x, y)\n"), [])

    def test_examples(self):
        expected = {
            "01_ttest_outliers.py": ["SA102"],
            "02_pearson_leverage.py": ["SA103"],
            "03_chisq_sparse.py": ["SA104"],
            "04_many_tests.py": ["SA101", "SA106"],
            "05_good_analysis.py": [],
            "06_notebook.ipynb": ["SA102"],
        }
        for name, want in expected.items():
            with self.subTest(name=name):
                self.assertEqual([f.rule for f in lint_file(EXAMPLES / name).findings], want)

    def test_notebook_cells_are_one_program(self):
        nb = {"cells": [
            {"cell_type": "code", "metadata": {}, "outputs": [], "source": ["%matplotlib inline\n", "from scipy import stats"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["x"]},
            {"cell_type": "code", "metadata": {}, "outputs": [], "source": ["x = 1\n", "stats.ttest_ind(a, b)"]},
            {"cell_type": "code", "metadata": {}, "outputs": [], "source": ["this is not python ("]},
        ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        path = Path(self.tmp) / "n.ipynb"
        path.write_text(json.dumps(nb), encoding="utf-8")
        rep = lint_file(path)
        (f,) = rep.findings
        self.assertEqual((f.rule, f.cell, f.line), ("SA102", 2, 2))
        self.assertIn("cell 3", rep.notes[0])
        self.assertEqual(f.location(str(path)), "n.ipynb cell 2, line 2")

    def test_cli(self):
        runner = CliRunner()
        res = runner.invoke(cli, ["lint", str(EXAMPLES / "01_ttest_outliers.py")])
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("SA102", res.output)
        self.assertIn("fix:", res.output)
        res = runner.invoke(cli, ["lint", str(EXAMPLES / "05_good_analysis.py")])
        self.assertEqual(res.exit_code, 0, res.output)
        js = str(Path(self.tmp) / "l.json")
        res = runner.invoke(cli, ["lint", str(EXAMPLES / "04_many_tests.py"), "--fail-on", "never", "-q", "--json", js])
        self.assertEqual(res.exit_code, 0)
        data = json.loads(Path(js).read_text())
        self.assertEqual([f["rule"] for f in data["findings"]], ["SA101", "SA106"])
        self.assertTrue(all(f["fix"] and f["explanation"] for f in data["findings"]))
        res = runner.invoke(cli, ["lint", str(EXAMPLES / "03_chisq_sparse.py"), "--select", "SA999"])
        self.assertEqual(res.exit_code, 2)
        res = runner.invoke(cli, ["lint", "--list-rules"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("SA107", res.output)


if __name__ == "__main__":
    unittest.main()
