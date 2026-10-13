"""Each auditor flags the bad case and passes a clean case."""

from __future__ import annotations

import unittest

import numpy as np
from helpers import make_record, rng
from scipy import stats

import stataudit
from stataudit.audits import audit_record, audit_records
from stataudit.models import Role, Severity, Status


def codes(result) -> set[str]:
    return {f.code for f in result.findings}


def alt(result, name_start: str):
    for a in result.alternatives:
        if a.name.startswith(name_start):
            return a
    return None


def flipping_ttest_data():
    r = np.random.default_rng(7)
    control = r.normal(12.0, 1.5, 60).round(1)
    treated = np.concatenate([r.normal(12.2, 1.5, 11), [21.5, 24.0, 26.5]]).round(1)
    return treated, control


class TestTTestInd(unittest.TestCase):
    def test_flags_outliers_unequal_variance_and_flips(self):
        treated, control = flipping_ttest_data()
        rep = stataudit.audit_ttest(treated, control)
        res = rep.results[0]
        self.assertLess(res.reported_p, 0.05)
        self.assertTrue({"outliers", "unequal_variances", "non_normal"} <= codes(res))
        welch = alt(res, "Welch")
        mw = alt(res, "Mann-Whitney")
        self.assertIsNotNone(welch)
        self.assertIsNotNone(mw)
        # Welch's p is exactly scipy's
        self.assertAlmostEqual(welch.pvalue, stats.ttest_ind(treated, control, equal_var=False).pvalue)
        self.assertFalse(welch.significant)
        self.assertIn("rank_biserial", res.effect_sizes)
        self.assertIs(res.status, Status.FAILS)
        self.assertIn("does NOT hold", res.verdict)
        self.assertIs(rep.status, Status.FAILS)

    def test_clean_case_holds(self):
        r = rng(1)
        a, b = r.normal(0, 1, 50), r.normal(0.7, 1, 50)
        rep = stataudit.audit_ttest(a, b, equal_var=False)
        res = rep.results[0]
        self.assertIs(res.status, Status.HOLDS)
        self.assertFalse([f for f in res.findings if f.severity is not Severity.INFO])
        self.assertEqual(res.alternatives, [])
        self.assertIn("holds", res.verdict)

    def test_effect_sizes(self):
        a, b = np.array([1.0, 2, 3, 4, 5]), np.array([3.0, 4, 5, 6, 7])
        res = stataudit.audit_ttest(a, b).results[0]
        self.assertAlmostEqual(res.effect_sizes["cohens_d"], -2 / np.sqrt(2.5))
        self.assertLess(abs(res.effect_sizes["hedges_g"]), abs(res.effect_sizes["cohens_d"]))

    def test_welch_used_by_user_is_not_recomputed(self):
        treated, control = flipping_ttest_data()
        res = stataudit.audit_ttest(treated, control, equal_var=False).results[0]
        self.assertIsNone(alt(res, "Welch"))
        self.assertEqual(res.test_label, "Welch's t-test")

    def test_nan_with_propagate_is_critical(self):
        a = np.array([1.0, 2, 3, np.nan, 5, 6])
        b = np.array([2.0, 3, 4, 5, 6, 7])
        res = audit_record(make_record("ttest_ind", a, b))[0]
        self.assertIn("missing_values", codes(res))
        self.assertTrue(any(f.severity is Severity.CRITICAL for f in res.findings))

    def test_labels_from_call_site(self):
        rec = make_record("ttest_ind", [1.0, 2, 3, 4], [2.0, 3, 4, 6], code="t, p = stats.ttest_ind(treated, control)")
        res = audit_record(rec)[0]
        self.assertEqual(res.target, "treated vs control")


class TestPairedAndOneSample(unittest.TestCase):
    def test_paired_skewed_differences_get_wilcoxon(self):
        r = rng(2)
        before = r.normal(50, 5, 15)
        after = before + np.r_[r.normal(0.3, 0.5, 13), [12.0, 15.0]]
        res = stataudit.audit_paired_ttest(before, after).results[0]
        w = alt(res, "Wilcoxon")
        self.assertIsNotNone(w)
        self.assertAlmostEqual(w.pvalue, stats.wilcoxon(before - after).pvalue)

    def test_paired_clean(self):
        r = rng(3)
        before = r.normal(50, 5, 40)
        after = before - 2 + r.normal(0, 1, 40)
        res = stataudit.audit_paired_ttest(before, after).results[0]
        self.assertIs(res.status, Status.HOLDS)
        self.assertEqual(res.alternatives, [])

    def test_one_sample_uses_popmean(self):
        x = np.array([0.1, 0.3, 0.2, 0.4, 0.2, 0.3, 9.0, 0.25, 0.35, 0.15])
        res = stataudit.audit_1samp_ttest(x, popmean=0.0).results[0]
        w = alt(res, "Wilcoxon")
        self.assertIsNotNone(w)
        self.assertAlmostEqual(w.pvalue, stats.wilcoxon(x - 0.0).pvalue)
        self.assertIn("vs 0", res.target)


class TestPearson(unittest.TestCase):
    def leverage_data(self):
        r = np.random.default_rng(11)
        x = np.append(r.normal(10, 2, 23), [31.0, 34.0])
        y = np.append(r.normal(200, 25, 23), [290.0, 305.0])
        return x, y

    def test_flags_high_leverage_points(self):
        x, y = self.leverage_data()
        res = stataudit.audit_pearson(x, y).results[0]
        self.assertIn("high_leverage", codes(res))
        hl = next(f for f in res.findings if f.code == "high_leverage")
        self.assertEqual(sorted(hl.evidence["indices"]), [23, 24])
        sp = alt(res, "Spearman")
        self.assertAlmostEqual(sp.pvalue, stats.spearmanr(x, y).pvalue)
        self.assertIs(res.status, Status.FAILS)

    def test_leave_one_out_reports_most_influential_index(self):
        r = rng(4)
        x = r.normal(size=20)
        y = 0.1 * x + r.normal(size=20)
        x[5], y[5] = 4.0, 4.0
        res = stataudit.audit_pearson(x, y).results[0]
        loo = [f for f in res.findings if f.code == "loo_flip"]
        if res.significant:
            self.assertTrue(loo)
            self.assertEqual(loo[0].evidence["most_influential_index"], 5)

    def test_clean_correlation_holds(self):
        r = rng(5)
        x = r.normal(size=60)
        y = 0.6 * x + r.normal(scale=0.8, size=60)
        res = stataudit.audit_pearson(x, y).results[0]
        self.assertIs(res.status, Status.HOLDS)


class TestANOVA(unittest.TestCase):
    def test_unequal_variances_trigger_welch_anova(self):
        r = rng(6)
        g1, g2, g3 = r.normal(0, 1, 30), r.normal(0.3, 1, 30), r.normal(0.8, 6, 8)
        res = stataudit.audit_anova(g1, g2, g3).results[0]
        self.assertIn("unequal_variances", codes(res))
        w = alt(res, "Welch's ANOVA")
        from stataudit.stats_extra import welch_anova

        self.assertAlmostEqual(w.pvalue, welch_anova(g1, g2, g3)[1])

    def test_non_normal_small_groups_trigger_kruskal(self):
        r = rng(7)
        groups = [r.exponential(1, 12) ** 2 for _ in range(3)]
        res = stataudit.audit_anova(*groups).results[0]
        kw = alt(res, "Kruskal")
        self.assertIsNotNone(kw)
        self.assertAlmostEqual(kw.pvalue, stats.kruskal(*groups).pvalue)

    def test_clean_anova(self):
        r = rng(8)
        groups = [r.normal(m, 1, 40) for m in (0, 0.2, 0.8)]
        res = stataudit.audit_anova(*groups).results[0]
        self.assertIs(res.status, Status.HOLDS)
        self.assertNotIn("unequal_variances", codes(res))


class TestChiSquare(unittest.TestCase):
    def test_2x2_small_counts_use_fisher(self):
        t = np.array([[8, 1], [2, 6]])
        res = stataudit.audit_chi2(t).results[0]
        self.assertIn("low_expected", codes(res))
        fe = alt(res, "Fisher")
        self.assertAlmostEqual(fe.pvalue, stats.fisher_exact(t).pvalue)
        self.assertIs(fe.role, Role.RECOMMENDED)

    def test_sparse_rxc_uses_monte_carlo_and_flips(self):
        t = np.array([[14, 6, 3, 0], [14, 9, 4, 0], [1, 1, 1, 1]])
        res = stataudit.audit_chi2(t).results[0]
        self.assertLess(res.reported_p, 0.05)
        mc = alt(res, "Monte Carlo")
        self.assertIsNotNone(mc)
        self.assertGreater(mc.pvalue, 0.05)
        self.assertIs(res.status, Status.FAILS)

    def test_large_counts_pass(self):
        t = np.array([[120, 80], [90, 110]])
        res = stataudit.audit_chi2(t).results[0]
        self.assertNotIn("low_expected", codes(res))
        self.assertIs(res.status, Status.HOLDS)

    def test_non_integer_table_is_critical(self):
        t = np.array([[0.2, 0.3], [0.25, 0.25]])
        rec = make_record("chi2_contingency", t)
        res = audit_record(rec)[0]
        self.assertIn("non_integer_counts", codes(res))


class TestRegression(unittest.TestCase):
    def hetero_data(self, seed=9, n=80):
        r = rng(seed)
        x = r.uniform(0, 10, n)
        y = 1 + 0.15 * x + r.normal(size=n) * (0.2 + 0.5 * x)
        return x, y

    def test_linregress_heteroskedastic_gets_hc3(self):
        x, y = self.hetero_data()
        res = stataudit.audit_linregress(x, y).results[0]
        self.assertIn("heteroskedasticity", codes(res))
        hc3 = alt(res, "OLS with HC3")
        from stataudit import stats_extra as sx

        fit = sx.ols_fit(sx.add_constant(x), y)
        self.assertAlmostEqual(hc3.pvalue, sx.hc3_tests(fit)[2][1])
        self.assertAlmostEqual(res.reported_p, stats.linregress(x, y).pvalue)

    def test_linregress_clean(self):
        r = rng(10)
        x = r.uniform(0, 10, 60)
        y = 2 + 0.5 * x + r.normal(size=60)
        res = stataudit.audit_linregress(x, y).results[0]
        self.assertIs(res.status, Status.HOLDS)

    def test_audit_ols_multiple_predictors(self):
        r = rng(11)
        X = r.normal(size=(100, 2))
        y = 1 + 0.5 * X[:, 0] + r.normal(size=100) * np.exp(X[:, 1])
        rep = stataudit.audit_ols(X, y, names=["dose", "age"])
        self.assertEqual([res.target for res in rep.results], ["slope of dose", "slope of age"])
        self.assertIn("heteroskedasticity", codes(rep.results[0]))

    def test_audit_ols_accepts_dataframe(self):
        import pandas as pd

        r = rng(12)
        df = pd.DataFrame({"a": r.normal(size=40), "b": r.normal(size=40)})
        rep = stataudit.audit_ols(df, df["a"] * 2 + r.normal(size=40))
        self.assertEqual(rep.results[0].target, "slope of a")


class TestNonparametric(unittest.TestCase):
    def test_mannwhitney_ties_info(self):
        x = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
        y = [2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
        res = audit_record(make_record("mannwhitneyu", x, y))[0]
        self.assertIn("ties", codes(res))
        self.assertTrue(all(f.severity is Severity.INFO for f in res.findings if f.code == "ties"))

    def test_wilcoxon_small_n(self):
        res = audit_record(make_record("wilcoxon", [1.0, 2, -0.5, 3, 4]))[0]
        self.assertIn("cannot_reach_alpha", codes(res))

    def test_kruskal_small_groups(self):
        res = audit_record(make_record("kruskal", [1, 2, 3], [4, 5, 6, 7], [2, 3, 4, 5, 6]))[0]
        self.assertIn("chi2_approx_small", codes(res))

    def test_spearman_small(self):
        res = audit_record(make_record("spearmanr", [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]))[0]
        self.assertIn("asymptotic_small", codes(res))

    def test_shapiro_is_not_in_multiplicity_family(self):
        res = audit_record(make_record("shapiro", np.arange(10.0)))[0]
        self.assertFalse(res.in_family)


class TestRobustness(unittest.TestCase):
    def test_unsupported_shape_is_reported_not_raised(self):
        r = rng(20)
        rec = make_record("ttest_ind", r.normal(size=(3, 4)), r.normal(size=(3, 4)), axis=1)
        res = audit_record(rec)[0]
        self.assertIn("unsupported", codes(res))
        self.assertIs(res.status, Status.NOT_AUDITED)

    def test_audit_call_validates_name(self):
        with self.assertRaises(ValueError):
            stataudit.audit_call("not_a_test", [[1, 2]])

    def test_audit_call_with_given_result(self):
        a, b = [1.0, 2, 3, 4, 5], [2.0, 3, 4, 5, 9]
        result = stats.ttest_ind(a, b)
        rep = stataudit.audit_call("ttest_ind", [a, b], {}, result)
        self.assertAlmostEqual(rep.results[0].reported_p, result.pvalue)

    def test_empty_records(self):
        rep = audit_records([])
        self.assertIs(rep.status, Status.NOT_AUDITED)
        self.assertIn("nothing_traced", {f.code for f in rep.script_findings})


if __name__ == "__main__":
    unittest.main()
