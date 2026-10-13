"""Statistical helpers checked against independent calculations and textbook values."""

from __future__ import annotations

import math
import unittest

import helpers  # noqa: F401  (puts src/ on sys.path)
import numpy as np
from scipy import stats

from stataudit import stats_extra as sx


def _design(seed: int = 0, n: int = 40, hetero: bool = False):
    r = np.random.default_rng(seed)
    x1 = r.uniform(0, 10, n)
    x2 = r.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])
    sd = 0.5 + (x1 if hetero else 1.0)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + r.normal(0, 1, n) * sd
    return X, y


class TestOLS(unittest.TestCase):
    def test_matches_linregress(self):
        r = np.random.default_rng(1)
        x = r.normal(size=30)
        y = 2 * x + r.normal(size=30)
        fit = sx.ols_fit(sx.add_constant(x), y)
        lr = stats.linregress(x, y)
        self.assertAlmostEqual(fit.beta[1], lr.slope, places=10)
        self.assertAlmostEqual(fit.beta[0], lr.intercept, places=10)
        self.assertAlmostEqual(fit.se[1], lr.stderr, places=10)
        self.assertAlmostEqual(fit.pvalues[1], lr.pvalue, places=10)

    def test_matches_lstsq(self):
        X, y = _design(2)
        fit = sx.ols_fit(X, y)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(fit.beta, beta, rtol=1e-10)

    def test_hat_diagonal_matches_explicit_matrix(self):
        X, _ = _design(3)
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        np.testing.assert_allclose(sx.hat_diagonal(X), np.diag(H), rtol=1e-10)
        self.assertAlmostEqual(sx.hat_diagonal(X).sum(), X.shape[1], places=10)

    def test_has_constant(self):
        X, _ = _design(0)
        self.assertTrue(sx.has_constant(X))
        self.assertFalse(sx.has_constant(X[:, 1:]))

    def test_needs_more_rows_than_columns(self):
        with self.assertRaises(ValueError):
            sx.ols_fit(np.ones((2, 2)), np.ones(2))


class TestCooksDistance(unittest.TestCase):
    def test_matches_leave_one_out_definition(self):
        """D_i = sum_j (yhat_j - yhat_j(i))^2 / (k s^2), computed by brute force."""
        X, y = _design(4, n=25)
        fit = sx.ols_fit(X, y)
        n, k = X.shape
        brute = np.empty(n)
        for i in range(n):
            keep = np.arange(n) != i
            b_i = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            brute[i] = np.sum((X @ fit.beta - X @ b_i) ** 2) / (k * fit.sigma2)
        np.testing.assert_allclose(sx.cooks_distance(fit), brute, rtol=1e-8)

    def test_flags_planted_point(self):
        r = np.random.default_rng(5)
        x = r.normal(size=30)
        y = r.normal(size=30)
        x[0], y[0] = 8.0, 8.0
        d = sx.cooks_distance(sx.ols_fit(sx.add_constant(x), y))
        self.assertEqual(int(np.argmax(d)), 0)
        self.assertGreater(d[0], 4 / 30)


class TestHC3(unittest.TestCase):
    def test_matches_direct_matrix_formula(self):
        X, y = _design(6, hetero=True)
        fit = sx.ols_fit(X, y)
        n = X.shape[0]
        XtX_inv = np.linalg.inv(X.T @ X)
        H = X @ XtX_inv @ X.T
        e = y - H @ y
        meat = np.zeros((X.shape[1], X.shape[1]))
        for i in range(n):  # explicit sum over observations
            xi = X[i][:, None]
            meat += (e[i] ** 2 / (1 - H[i, i]) ** 2) * (xi @ xi.T)
        expected = XtX_inv @ meat @ XtX_inv
        np.testing.assert_allclose(sx.hc3_cov(fit), expected, rtol=1e-9)

    def test_hc3_tests_shapes_and_pvalues(self):
        X, y = _design(7, hetero=True)
        fit = sx.ols_fit(X, y)
        se, t, p = sx.hc3_tests(fit)
        np.testing.assert_allclose(t, fit.beta / se)
        np.testing.assert_allclose(p, 2 * stats.t.sf(np.abs(t), fit.df_resid))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_hc3_close_to_classical_when_homoskedastic(self):
        X, y = _design(8, n=4000)
        fit = sx.ols_fit(X, y)
        se, _, _ = sx.hc3_tests(fit)
        np.testing.assert_allclose(se, fit.se, rtol=0.1)


class TestBreuschPagan(unittest.TestCase):
    def test_koenker_matches_n_r_squared(self):
        X, y = _design(9, hetero=True)
        e = sx.ols_fit(X, y).resid
        e2 = e**2
        b = np.linalg.lstsq(X, e2, rcond=None)[0]
        r2 = 1 - np.sum((e2 - X @ b) ** 2) / np.sum((e2 - e2.mean()) ** 2)
        lm, p, df = sx.breusch_pagan(e, X, robust=True)
        self.assertAlmostEqual(lm, len(y) * r2, places=8)
        self.assertEqual(df, 2)
        self.assertAlmostEqual(p, stats.chi2.sf(len(y) * r2, 2), places=10)

    def test_original_bp_matches_half_ess(self):
        X, y = _design(10, hetero=True)
        e = sx.ols_fit(X, y).resid
        g = e**2 / np.mean(e**2)
        b = np.linalg.lstsq(X, g, rcond=None)[0]
        ess = np.sum((X @ b - g.mean()) ** 2)
        lm, _, _ = sx.breusch_pagan(e, X, robust=False)
        self.assertAlmostEqual(lm, ess / 2, places=8)

    def test_detects_heteroskedasticity(self):
        X, y = _design(11, n=200, hetero=True)
        _, p, _ = sx.breusch_pagan(sx.ols_fit(X, y).resid, X)
        self.assertLess(p, 0.001)

    def test_null_is_not_rejected_on_average(self):
        rejections = 0
        for seed in range(40):
            X, y = _design(100 + seed, n=80)
            _, p, _ = sx.breusch_pagan(sx.ols_fit(X, y).resid, X)
            rejections += p < 0.05
        self.assertLessEqual(rejections, 8)


class TestWelchANOVA(unittest.TestCase):
    def test_hand_computed_example(self):
        """Three small groups; every intermediate quantity written out by hand."""
        g1 = [10.0, 12.0, 14.0, 16.0]  # mean 13, var 6.6667
        g2 = [20.0, 21.0, 22.0]  # mean 21, var 1
        g3 = [5.0, 9.0, 13.0, 17.0, 21.0]  # mean 13, var 40
        n = np.array([4, 3, 5.0])
        m = np.array([13.0, 21.0, 13.0])
        v = np.array([20 / 3, 1.0, 40.0])
        w = n / v  # 0.6, 3, 0.125
        W = w.sum()  # 3.725
        mw = (w * m).sum() / W
        A = (w * (m - mw) ** 2).sum() / 2
        lam = (((1 - w / W) ** 2) / (n - 1)).sum()
        B = 1 + 2 * 1 / 8 * lam
        F = A / B
        df2 = 8 / (3 * lam)
        f, p, d1, d2 = sx.welch_anova(g1, g2, g3)
        self.assertAlmostEqual(f, F, places=10)
        self.assertEqual(d1, 2.0)
        self.assertAlmostEqual(d2, df2, places=10)
        self.assertAlmostEqual(p, stats.f.sf(F, 2, df2), places=12)
        # sanity on the hand numbers
        self.assertAlmostEqual(W, 3.725, places=12)

    def test_two_groups_equals_welch_t_squared(self):
        r = np.random.default_rng(12)
        a, b = r.normal(0, 1, 15), r.normal(0.8, 3, 25)
        f, p, _, df2 = sx.welch_anova(a, b)
        t = stats.ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(f, t.statistic**2, places=8)
        self.assertAlmostEqual(p, t.pvalue, places=8)

    def test_matches_scipy_if_available(self):
        r = np.random.default_rng(13)
        groups = [r.normal(0, s, n) for s, n in ((1, 10), (2, 20), (4, 15))]
        try:
            ref = stats.f_oneway(*groups, equal_var=False)
        except TypeError:
            self.skipTest("this scipy has no Welch option in f_oneway")
        f, p, _, _ = sx.welch_anova(*groups)
        self.assertAlmostEqual(f, ref.statistic, places=8)
        self.assertAlmostEqual(p, ref.pvalue, places=8)

    def test_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            sx.welch_anova([1, 2, 3])
        with self.assertRaises(ValueError):
            sx.welch_anova([1, 1, 1], [1, 2, 3])


class TestHolm(unittest.TestCase):
    def test_hand_example(self):
        # sorted: .005 (x4=.02), .01 (x3=.03), .03 (x2=.06), .04 (x1=.04 -> monotone .06)
        p = [0.01, 0.04, 0.03, 0.005]
        np.testing.assert_allclose(sx.holm(p), [0.03, 0.06, 0.06, 0.02])

    def test_caps_at_one_and_handles_nan(self):
        out = sx.holm([0.5, np.nan, 0.9])
        self.assertTrue(math.isnan(out[1]))
        np.testing.assert_allclose(out[[0, 2]], [1.0, 1.0])

    def test_rejections_match_step_down_definition(self):
        r = np.random.default_rng(14)
        p = np.concatenate([r.uniform(0, 0.01, 4), r.uniform(0, 1, 16)])
        adj = sx.holm(p)
        order = np.argsort(p)
        m = len(p)
        rejected = np.zeros(m, bool)
        for i, idx in enumerate(order):  # textbook step-down
            if p[idx] <= 0.05 / (m - i):
                rejected[idx] = True
            else:
                break
        np.testing.assert_array_equal(adj <= 0.05, rejected)

    def test_empty(self):
        self.assertEqual(sx.holm([]).size, 0)


class TestContingency(unittest.TestCase):
    def test_chi2_statistic_matches_scipy(self):
        t = np.array([[12, 5, 7], [3, 9, 14]])
        ref = stats.chi2_contingency(t, correction=False)
        self.assertAlmostEqual(sx.chi2_statistic(t), ref.statistic, places=10)
        np.testing.assert_allclose(sx.expected_counts(t), ref.expected_freq)

    def test_monte_carlo_matches_exact_permutation_for_2x2(self):
        """Exact p = P(chi2 >= observed) under the hypergeometric null with fixed margins."""
        t = np.array([[7, 2], [3, 8]])
        obs = sx.chi2_statistic(t)
        r1, c1, n = t[0].sum(), t[:, 0].sum(), t.sum()
        exact = 0.0
        for a in range(max(0, r1 + c1 - n), min(r1, c1) + 1):
            tab = np.array([[a, r1 - a], [c1 - a, n - r1 - c1 + a]])
            if sx.chi2_statistic(tab) >= obs - 1e-9:
                exact += stats.hypergeom.pmf(a, n, r1, c1)
        _, p = sx.chi2_monte_carlo(t, n_resamples=20000, seed=1)
        self.assertAlmostEqual(p, exact, delta=0.01)

    def test_monte_carlo_is_reproducible(self):
        t = np.array([[3, 1, 0], [1, 4, 2], [0, 1, 5]])
        self.assertEqual(sx.chi2_monte_carlo(t, 2000, seed=7), sx.chi2_monte_carlo(t, 2000, seed=7))

    def test_monte_carlo_rejects_non_counts(self):
        with self.assertRaises(ValueError):
            sx.chi2_monte_carlo(np.array([[0.5, 1], [2, 3]]))

    def test_cramers_v_bounds(self):
        self.assertAlmostEqual(sx.cramers_v(np.array([[10, 0], [0, 10]])), 1.0)
        self.assertAlmostEqual(sx.cramers_v(np.array([[5, 5], [5, 5]])), 0.0)


class TestDescriptives(unittest.TestCase):
    def test_robust_z_hand_values(self):
        x = np.array([1.0, 2, 3, 4, 100])  # median 3, MAD = median(|.|)= median(2,1,0,1,97)=1
        np.testing.assert_allclose(sx.robust_z(x), 0.6745 * (x - 3) / 1.0)
        np.testing.assert_array_equal(sx.mad_outliers(x), [4])

    def test_robust_z_zero_mad(self):
        x = np.array([5.0, 5, 5, 5, 9])
        self.assertEqual(int(np.argmax(np.abs(sx.robust_z(x)))), 4)
        np.testing.assert_array_equal(sx.robust_z(np.ones(4)), np.zeros(4))

    def test_iqr_outliers(self):
        x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 50])
        np.testing.assert_array_equal(sx.iqr_outliers(x), [8])

    def test_cohens_d_and_hedges_g(self):
        a, b = np.array([2.0, 4, 6, 8]), np.array([1.0, 3, 5, 7])
        sp = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        self.assertAlmostEqual(sx.cohens_d(a, b), 1 / sp)
        self.assertAlmostEqual(sx.hedges_g(a, b), (1 / sp) * (1 - 3 / (4 * 6 - 1)))

    def test_rank_biserial_extremes(self):
        a, b = [10, 11, 12], [1, 2, 3, 4]
        u = stats.mannwhitneyu(a, b).statistic
        self.assertAlmostEqual(sx.rank_biserial(u, 3, 4), 1.0)

    def test_eta_squared(self):
        g = [[1.0, 2, 3], [4.0, 5, 6]]
        # grand mean 3.5; SSb = 3*(1.5^2)*2 = 13.5; SSt = 17.5
        self.assertAlmostEqual(sx.eta_squared(*g), 13.5 / 17.5)

    def test_safe_shapiro_limits(self):
        self.assertIsNone(sx.safe_shapiro([1.0, 2.0]))
        self.assertIsNone(sx.safe_shapiro(np.ones(10)))
        self.assertIsNotNone(sx.safe_shapiro(np.arange(10.0)))


if __name__ == "__main__":
    unittest.main()
