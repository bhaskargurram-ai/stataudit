"""Poisson and logistic regression: the numpy fit, the auditor and the statsmodels hook."""

from __future__ import annotations

import importlib.util
import math
import os
import unittest
from pathlib import Path

import numpy as np
from helpers import TempDirMixin, write_script

import stataudit
from stataudit import trace
from stataudit.audits.common import Unsupported
from stataudit.audits.glm import audit_glm_record
from stataudit.glmfit import glm_fit, lr_tests, quasi_tests, sandwich_tests
from stataudit.models import CallRecord, Status

HAVE_STATSMODELS = importlib.util.find_spec("statsmodels") is not None


def _design(n, seed=0, k=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    return rng, x, np.column_stack([np.ones(n), x])


class FitTests(unittest.TestCase):
    def test_poisson_solves_the_score_equations(self):
        rng, x, X = _design(400)
        y = rng.poisson(np.exp(0.3 + 0.5 * x[:, 0]))
        f = glm_fit(X, y, "poisson")
        self.assertTrue(f.converged)
        np.testing.assert_allclose(X.T @ (y - f.mu), 0, atol=1e-7)
        self.assertAlmostEqual(f.beta[1], 0.5, delta=0.1)
        self.assertLess(abs(f.dispersion - 1), 0.25)

    def test_logistic_solves_the_score_equations(self):
        rng, x, X = _design(400, 1)
        y = (rng.random(400) < 1 / (1 + np.exp(-(-0.5 + 1.2 * x[:, 0])))).astype(float)
        f = glm_fit(X, y, "binomial")
        np.testing.assert_allclose(X.T @ (y - f.mu), 0, atol=1e-7)

    def test_offset(self):
        rng, x, X = _design(600, 2)
        off = np.log(rng.uniform(1, 5, 600))
        y = rng.poisson(np.exp(0.3 + 0.5 * x[:, 0] + off))
        f = glm_fit(X, y, "poisson", off)
        np.testing.assert_allclose(X.T @ (y - f.mu), 0, atol=1e-7)
        self.assertAlmostEqual(f.beta[1], 0.5, delta=0.1)

    def test_quasi_scales_by_dispersion(self):
        rng, x, X = _design(300, 3)
        lam = np.exp(0.5 + 0.3 * x[:, 0])
        y = rng.negative_binomial(1.0, 1.0 / (1.0 + lam))
        f = glm_fit(X, y, "poisson")
        se, _, p = quasi_tests(f)
        np.testing.assert_allclose(se, f.se * math.sqrt(f.dispersion))
        self.assertGreater(p[1], f.pvalues[1])
        rse, _, _ = sandwich_tests(f)
        self.assertGreater(rse[1], f.se[1])

    def test_lr_test_survives_separation(self):
        x = np.r_[np.linspace(-2, -0.1, 20), np.linspace(0.1, 2, 20)]
        y = (x > 0).astype(float)
        X = np.column_stack([np.ones(40), x])
        f = glm_fit(X, y, "binomial")
        self.assertGreater(f.pvalues[1], 0.9)  # Wald: huge coefficient, huge SE
        _, lr_p = lr_tests(f)
        self.assertLess(lr_p[1], 1e-6)

    def test_bad_responses(self):
        X = np.column_stack([np.ones(5), np.arange(5.0)])
        with self.assertRaises(ValueError):
            glm_fit(X, np.array([0, 1, 2, 0, 1.0]), "binomial")
        with self.assertRaises(ValueError):
            glm_fit(X, np.array([0, 1, -2, 0, 1.0]), "poisson")

    @unittest.skipUnless(HAVE_STATSMODELS, "statsmodels is not installed")
    def test_matches_statsmodels(self):
        import statsmodels.api as sm  # noqa: PLC0415

        rng, x, X = _design(300, 4, k=2)
        y = rng.poisson(np.exp(0.2 + 0.4 * x[:, 0] - 0.3 * x[:, 1]))
        ours = glm_fit(X, y, "poisson")
        theirs = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        np.testing.assert_allclose(ours.beta, theirs.params, rtol=1e-6)
        np.testing.assert_allclose(ours.se, theirs.bse, rtol=1e-5)
        np.testing.assert_allclose(ours.pearson_chi2, theirs.pearson_chi2, rtol=1e-6)
        robust = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC0")
        np.testing.assert_allclose(sandwich_tests(ours)[0], robust.bse, rtol=1e-4)
        yb = (rng.random(300) < 1 / (1 + np.exp(-(0.3 * x[:, 0])))).astype(float)
        ours = glm_fit(X, yb, "binomial")
        theirs = sm.Logit(yb, X).fit(disp=0)
        np.testing.assert_allclose(ours.beta, theirs.params, rtol=1e-6)
        np.testing.assert_allclose(ours.se, theirs.bse, rtol=1e-5)


class AuditTests(unittest.TestCase):
    def test_overdispersion_is_flagged_with_quasi_poisson(self):
        rng, x, _ = _design(200, 1)
        lam = np.exp(0.5 + 0.15 * x[:, 0])
        y = rng.negative_binomial(1.5, 1.5 / (1.5 + lam))
        (r,) = stataudit.audit_glm(x, y, family="poisson").results
        self.assertIn("overdispersion", {f.code for f in r.findings})
        names = [a.name for a in r.alternatives]
        self.assertIn("Quasi-Poisson (dispersion-scaled SE)", names)
        self.assertIn("rate_ratio", r.effect_sizes)

    def test_overdispersion_can_overturn_significance(self):
        # weak effect on very overdispersed counts: Poisson p < 0.05, quasi-Poisson p > 0.05
        found = False
        for seed in range(40):
            rng, x, _ = _design(120, seed)
            lam = np.exp(1.0 + 0.1 * x[:, 0])
            y = rng.negative_binomial(0.5, 0.5 / (0.5 + lam))
            (r,) = stataudit.audit_glm(x, y, family="poisson").results
            quasi = next(a for a in r.alternatives if a.name.startswith("Quasi"))
            if r.significant and not quasi.significant:
                self.assertEqual(r.status, Status.FAILS, r.verdict)
                found = True
                break
        self.assertTrue(found, "no seed produced a significance flip")

    def test_separation_does_not_hold(self):
        x = np.r_[np.linspace(-2, -0.1, 20), np.linspace(0.1, 2, 20)]
        y = (x > 0).astype(float)
        (r,) = stataudit.audit_glm(x, y, family="binomial").results
        self.assertIn("separation", {f.code for f in r.findings})
        lr = next(a for a in r.alternatives if a.name == "Likelihood-ratio test")
        self.assertTrue(lr.significant)
        self.assertFalse(any(a.name.startswith("Robust") for a in r.alternatives))
        self.assertEqual(r.status, Status.FAILS)

    def test_few_events_per_variable(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 3))
        y = (rng.random(60) < 1 / (1 + np.exp(-(-2.5 + 0.9 * x[:, 0])))).astype(float)
        rep = stataudit.audit_glm(x, y, family="binomial")
        self.assertEqual(len(rep.results), 3)
        self.assertIn("few_events_per_variable", {f.code for f in rep.results[0].findings})

    def test_clean_logistic_holds(self):
        rng, x, _ = _design(400, 6)
        y = (rng.random(400) < 1 / (1 + np.exp(-(0.2 + 0.8 * x[:, 0])))).astype(float)
        (r,) = stataudit.audit_glm(x, y, family="binomial").results
        self.assertEqual(r.status, Status.HOLDS, r.verdict)
        self.assertEqual([f.code for f in r.findings], [])

    def test_unsupported(self):
        X = np.column_stack([np.ones(30), np.arange(30.0)])
        y = np.arange(30.0) % 3
        for params in ({"family": "gaussian"}, {"family": "poisson/identity"},
                       {"family": "poisson", "weighted": True}):
            rec = CallRecord(index=0, func="statsmodels_glm", inputs={"X": X, "y": y}, params=params, result={})
            with self.assertRaises(Unsupported):
                audit_glm_record(rec)
        rec = CallRecord(index=0, func="statsmodels_glm", inputs={"X": X, "y": y / 3},
                         params={"family": "binomial"}, result={})
        with self.assertRaises(Unsupported):
            audit_glm_record(rec)


FAKE_GLM = '''
import numpy as np


class _Res:
    def __init__(self, k):
        self.params = np.zeros(k)
        self.pvalues = np.full(k, 0.01)


class Log:
    pass


class Logit:
    pass


class Identity:
    pass


class Poisson:
    def __init__(self, link=None):
        self.link = link or Log()


class Binomial:
    def __init__(self, link=None):
        self.link = link or Logit()


class GLM:
    def __init__(self, endog, exog, family=None, offset=None, exposure=None, freq_weights=None):
        self.endog = np.asarray(endog, dtype=float)
        self.exog = np.asarray(exog, dtype=float)
        self.family = family
        self.offset = offset
        self.exposure = None if exposure is None else np.log(exposure)
        self.freq_weights = freq_weights
        self.exog_names = ["const"] + [f"x{i}" for i in range(1, self.exog.shape[1])]

    def fit(self, cov_type="nonrobust"):
        return _Res(self.exog.shape[1])
'''

FAKE_DISCRETE = '''
import numpy as np


class _Res:
    def __init__(self, k):
        self.params = np.zeros(k)
        self.pvalues = np.full(k, 0.01)


class DiscreteModel:
    def __init__(self, endog, exog, offset=None, exposure=None):
        self.endog = np.asarray(endog, dtype=float)
        self.exog = np.asarray(exog, dtype=float)
        self.offset = offset
        self.exposure = None if exposure is None else np.log(exposure)
        self.exog_names = ["const"] + [f"x{i}" for i in range(1, self.exog.shape[1])]

    def fit(self, disp=0):
        return _Res(self.exog.shape[1])


class Logit(DiscreteModel):
    pass


class Poisson(DiscreteModel):
    pass
'''


class StatsmodelsHookTests(TempDirMixin, unittest.TestCase):
    """The GLM hooks, with a minimal stand-in statsmodels package."""

    def _trace(self, body):
        pkg = Path(self.tmp) / "fake" / "statsmodels"
        for sub in ("genmod", "discrete"):
            (pkg / sub).mkdir(parents=True, exist_ok=True)
            (pkg / sub / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "genmod" / "generalized_linear_model.py").write_text(FAKE_GLM, encoding="utf-8")
        (pkg / "discrete" / "discrete_model.py").write_text(FAKE_DISCRETE, encoding="utf-8")
        script = write_script(self.tmp, body)
        old = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(Path(self.tmp) / "fake")
        try:
            return trace.trace_script(script)
        finally:
            if old is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old

    def test_glm_logit_and_poisson_fits_are_recorded(self):
        tr = self._trace(
            """
            import numpy as np
            from statsmodels.genmod.generalized_linear_model import GLM, Poisson, Binomial, Identity
            from statsmodels.discrete.discrete_model import Logit, Poisson as DPoisson
            r = np.random.default_rng(0)
            x = r.normal(size=80)
            X = np.column_stack([np.ones(80), x])
            counts = r.poisson(np.exp(0.5 + 0.3 * x))
            binary = (r.random(80) < 0.5).astype(float)
            GLM(counts, X, family=Poisson(), exposure=np.full(80, 2.0)).fit()
            GLM(counts, X, family=Poisson(link=Identity())).fit()
            GLM(counts, X, family=Poisson(), freq_weights=np.full(80, 2.0)).fit()
            Logit(binary, X).fit(disp=0)
            DPoisson(counts, X, offset=np.zeros(80)).fit(disp=0)
            print("ok")
            """
        )
        self.assertIsNone(tr.error, tr.stderr)
        self.assertEqual(tr.stdout.strip(), "ok")
        recs = [r for r in tr.records if r.func == "statsmodels_glm"]
        self.assertEqual(len(recs), 5)
        self.assertEqual([r.params["family"] for r in recs],
                         ["poisson", "poisson/identity", "poisson", "binomial", "poisson"])
        np.testing.assert_allclose(recs[0].inputs["offset"], np.log(2.0))
        self.assertTrue(recs[2].params["weighted"])
        self.assertIn("offset", recs[4].inputs)
        rep = stataudit.audit_records(recs)
        audited = [r for r in rep.results if r.status != Status.NOT_AUDITED]
        self.assertEqual({r.test_label for r in audited}, {"Poisson regression", "Logistic regression"})

    @unittest.skipUnless(HAVE_STATSMODELS, "statsmodels is not installed")
    def test_real_statsmodels_formula_api(self):
        tr = trace.trace_script(write_script(
            self.tmp,
            """
            import numpy as np, pandas as pd
            import statsmodels.formula.api as smf
            r = np.random.default_rng(1)
            df = pd.DataFrame({"x": r.normal(size=150)})
            lam = np.exp(0.8 + 0.2 * df.x)
            df["y"] = r.negative_binomial(1.0, 1.0 / (1.0 + lam))
            df["b"] = (r.random(150) < 1 / (1 + np.exp(-df.x))).astype(int)
            smf.poisson("y ~ x", df).fit(disp=0)
            smf.logit("b ~ x", df).fit(disp=0)
            """,
        ))
        self.assertIsNone(tr.error, tr.stderr)
        recs = [r for r in tr.records if r.func == "statsmodels_glm"]
        self.assertEqual([r.params["family"] for r in recs], ["poisson", "binomial"])
        rep = stataudit.audit_records(recs)
        poisson = next(r for r in rep.results if r.test_label == "Poisson regression")
        self.assertEqual(poisson.target, "coefficient of x")
        self.assertIn("overdispersion", {f.code for f in poisson.findings})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
