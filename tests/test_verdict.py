"""Status decisions, verdict text and script-level checks."""

from __future__ import annotations

import unittest

import numpy as np
from helpers import make_record

from stataudit.audits import audit_records
from stataudit.audits.common import fmt_p, make_alt, new_result
from stataudit.models import Finding, Role, Severity, Status
from stataudit.verdict import decide_status, write_verdict


def _res(p: float, alts=(), findings=()):
    rec = make_record("ttest_ind", [1.0, 2, 3], [2.0, 3, 4])
    r = new_result(rec, "Student's t-test", 0.05)
    r.reported_p = p
    r.alternatives = list(alts)
    r.findings = list(findings)
    return r


class TestDecideStatus(unittest.TestCase):
    def test_all_recommended_disagree_fails(self):
        r = _res(0.01, [make_alt("Welch", 0.2, 0.05, Role.RECOMMENDED), make_alt("MW", 0.3, 0.05, Role.RECOMMENDED)])
        self.assertIs(decide_status(r), Status.FAILS)

    def test_partial_disagreement_is_fragile(self):
        r = _res(0.01, [make_alt("Welch", 0.2, 0.05, Role.RECOMMENDED), make_alt("MW", 0.01, 0.05, Role.RECOMMENDED)])
        self.assertIs(decide_status(r), Status.FRAGILE)

    def test_sensitivity_flip_is_fragile(self):
        r = _res(0.01, [make_alt("LOO", 0.2, 0.05, Role.SENSITIVITY)])
        self.assertIs(decide_status(r), Status.FRAGILE)

    def test_holm_loss_is_fragile(self):
        r = _res(0.02, [make_alt("Holm", 0.3, 0.05, Role.MULTIPLICITY, family_size=15)])
        self.assertIs(decide_status(r), Status.FRAGILE)
        r.status = Status.FRAGILE
        self.assertIn("Holm correction for 15 tests", write_verdict(r))

    def test_agreement_holds(self):
        r = _res(0.01, [make_alt("Welch", 0.02, 0.05, Role.RECOMMENDED)])
        self.assertIs(decide_status(r), Status.HOLDS)

    def test_non_significant_that_becomes_significant_fails(self):
        r = _res(0.2, [make_alt("Fisher", 0.01, 0.05, Role.RECOMMENDED)])
        self.assertIs(decide_status(r), Status.FAILS)

    def test_critical_finding_fails(self):
        r = _res(0.01, findings=[Finding("x", Severity.CRITICAL, "bad")])
        self.assertIs(decide_status(r), Status.FAILS)


class TestVerdictText(unittest.TestCase):
    def test_mentions_location_p_and_alternatives(self):
        r = _res(0.031, [make_alt("Welch's t-test", 0.21, 0.05, Role.RECOMMENDED)],
                 [Finding("outliers", Severity.WARN, "t", short="group B has 3 outliers")])
        r.status = decide_status(r)
        v = write_verdict(r)
        self.assertIn(r.location, v)
        self.assertIn("p = 0.031 (significant)", v)
        self.assertIn("Group B has 3 outliers", v)
        self.assertIn("Welch's t-test gives p = 0.21", v)
        self.assertIn("does NOT hold", v)

    def test_fmt_p(self):
        self.assertEqual(fmt_p(0.031), "0.031")
        self.assertEqual(fmt_p(0.2), "0.2")
        self.assertEqual(fmt_p(3.2e-5), "3.2e-05")
        self.assertEqual(fmt_p(None), "n/a")
        self.assertEqual(fmt_p(float("nan")), "n/a")


class TestScriptLevel(unittest.TestCase):
    def test_multiplicity_with_holm(self):
        r = np.random.default_rng(0)
        recs = []
        for i in range(15):
            a = r.normal(0, 1, 40)
            b = r.normal(1.0 if i < 2 else 0.0, 1, 40)
            recs.append(make_record("ttest_ind", a, b, equal_var=False))
        rep = audit_records(recs)
        self.assertEqual(len(rep.multiplicity), 15)
        mf = next(f for f in rep.script_findings if f.code == "multiple_comparisons")
        self.assertEqual(mf.evidence["m"], 15)
        surv = sum(row["survives"] for row in rep.multiplicity)
        self.assertEqual(mf.evidence["survive_holm"], surv)
        self.assertGreaterEqual(surv, 2)
        # Holm-adjusted alternatives attached to significant results only
        for res in rep.results:
            has = any(a.role is Role.MULTIPLICITY for a in res.alternatives)
            self.assertEqual(has, bool(res.significant))

    def test_single_test_has_no_multiplicity(self):
        rep = audit_records([make_record("ttest_ind", [1.0, 2, 3, 4], [5.0, 6, 7, 8])])
        self.assertEqual(rep.multiplicity, [])

    def test_test_shopping_detected(self):
        r = np.random.default_rng(1)
        a, b = r.normal(0, 1, 20), r.normal(0.5, 1, 20)
        recs = [
            make_record("ttest_ind", a, b),
            make_record("mannwhitneyu", a, b),
            make_record("ttest_ind", a, b, equal_var=False),
            make_record("kruskal", a, b),
        ]
        rep = audit_records(recs)
        codes = {f.code for f in rep.script_findings}
        self.assertIn("test_shopping", codes)
        self.assertNotEqual(rep.status, Status.HOLDS)

    def test_small_samples_flagged(self):
        rep = audit_records([make_record("ttest_ind", [1.0, 2, 3, 4], [5.0, 6, 7, 9])])
        self.assertIn("small_samples", {f.code for f in rep.script_findings})


if __name__ == "__main__":
    unittest.main()
