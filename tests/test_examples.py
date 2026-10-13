"""The shipped examples produce the verdicts the README advertises."""

from __future__ import annotations

import unittest

from helpers import EXAMPLES

import stataudit
from stataudit.models import Status


class TestExamples(unittest.TestCase):
    def audit(self, name: str):
        rep = stataudit.audit_script(EXAMPLES / name)
        self.assertIsNone(rep.script_error, rep.script_error)
        return rep

    def test_01_ttest_flips(self):
        rep = self.audit("01_ttest_outliers.py")
        (res,) = rep.results
        self.assertLess(res.reported_p, 0.05)
        self.assertIs(res.status, Status.FAILS)
        names = {a.name: a for a in res.alternatives}
        self.assertFalse(names["Welch's t-test"].significant)
        self.assertFalse(names["Mann-Whitney U"].significant)

    def test_02_pearson_leverage(self):
        rep = self.audit("02_pearson_leverage.py")
        (res,) = rep.results
        self.assertIs(res.status, Status.FAILS)
        hl = next(f for f in res.findings if f.code == "high_leverage")
        self.assertEqual(sorted(hl.evidence["indices"]), [23, 24])

    def test_03_chisq_sparse(self):
        rep = self.audit("03_chisq_sparse.py")
        (res,) = rep.results
        self.assertLess(res.reported_p, 0.05)
        self.assertIs(res.status, Status.FAILS)
        self.assertTrue(any(a.name.startswith("Monte Carlo") and not a.significant for a in res.alternatives))

    def test_04_many_tests(self):
        rep = self.audit("04_many_tests.py")
        self.assertEqual(len(rep.results), 15)
        mf = next(f for f in rep.script_findings if f.code == "multiple_comparisons")
        self.assertGreater(mf.evidence["significant"], mf.evidence["survive_holm"])
        self.assertEqual(mf.evidence["survive_holm"], 2)
        self.assertIs(rep.status, Status.FRAGILE)

    def test_05_good(self):
        rep = self.audit("05_good_analysis.py")
        self.assertIs(rep.status, Status.HOLDS)


if __name__ == "__main__":
    unittest.main()
