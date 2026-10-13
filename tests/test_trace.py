"""The tracer records calls faithfully and never changes results."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.stats
from helpers import TempDirMixin, write_script

import stataudit
from stataudit import trace
from stataudit._scipy import ORIGINAL_ATTR
from stataudit.models import Status


class TestTraceScript(TempDirMixin, unittest.TestCase):
    def test_module_attribute_import_style(self):
        path = write_script(
            self.tmp,
            """
            import numpy as np
            from scipy import stats
            a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            b = np.array([2.0, 4.0, 5.0, 6.0, 9.0])
            res = stats.ttest_ind(a, b)
            print(repr(float(res.pvalue)))
            """,
        )
        tr = trace.trace_script(path)
        self.assertIsNone(tr.error, tr.stderr)
        self.assertEqual(len(tr.records), 1)
        rec = tr.records[0]
        self.assertEqual(rec.func, "ttest_ind")
        np.testing.assert_array_equal(rec.inputs["a"], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(rec.inputs["b"], [2, 4, 5, 6, 9])
        expected = scipy.stats.ttest_ind([1.0, 2, 3, 4, 5], [2.0, 4, 5, 6, 9]).pvalue
        self.assertAlmostEqual(rec.result["pvalue"], expected, places=12)
        # the script saw exactly the same value as an untraced call
        self.assertEqual(float(tr.stdout.strip()), float(expected))
        self.assertEqual(rec.site.lineno, 6)
        self.assertTrue(rec.site.filename.endswith("analysis.py"))
        self.assertIn("stats.ttest_ind(a, b)", rec.site.code)

    def test_from_import_style_and_kwargs(self):
        path = write_script(
            self.tmp,
            """
            from scipy.stats import pearsonr, ttest_ind, f_oneway, chi2_contingency
            x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            y = [1.5, 2.5, 2.9, 4.2, 5.1, 6.3]
            r = pearsonr(x, y, alternative="greater")
            t = ttest_ind(x, y, equal_var=False)
            f = f_oneway([1, 2, 3], [4, 5, 6], [7, 8, 9.5])
            c = chi2_contingency([[10, 20], [30, 25]])
            """,
        )
        tr = trace.trace_script(path)
        self.assertEqual([r.func for r in tr.records], ["pearsonr", "ttest_ind", "f_oneway", "chi2_contingency"])
        self.assertEqual(tr.records[0].params["alternative"], "greater")
        self.assertIs(tr.records[1].params["equal_var"], False)
        self.assertEqual(len(tr.records[2].inputs["samples"]), 3)
        self.assertEqual(tr.records[3].inputs["observed"].shape, (2, 2))

    def test_script_error_keeps_partial_trace(self):
        path = write_script(
            self.tmp,
            """
            from scipy import stats
            stats.ttest_ind([1.0, 2, 3, 4], [3.0, 4, 5, 7])
            raise RuntimeError("boom after first test")
            """,
        )
        tr = trace.trace_script(path)
        self.assertEqual(len(tr.records), 1)
        self.assertIn("boom after first test", tr.error)
        rep = stataudit.audit_script(path)
        self.assertEqual(len(rep.results), 1)
        self.assertEqual(rep.script_findings[0].code, "script_error")
        self.assertIn("boom", rep.to_markdown())

    def test_script_args_and_main_guard(self):
        path = write_script(
            self.tmp,
            """
            import sys
            from scipy import stats
            if __name__ == "__main__":
                n = int(sys.argv[1])
                stats.ttest_1samp(list(range(n)), 0.0)
            """,
        )
        tr = trace.trace_script(path, args=["7"])
        self.assertEqual(len(tr.records), 1)
        self.assertEqual(tr.records[0].inputs["a"].size, 7)

    def test_timeout_reports_error(self):
        path = write_script(
            self.tmp,
            """
            import time
            from scipy import stats
            stats.ttest_ind([1.0, 2, 3], [4.0, 5, 6])
            time.sleep(30)
            """,
        )
        tr = trace.trace_script(path, timeout=3)
        self.assertTrue(tr.timed_out)
        self.assertEqual(len(tr.records), 1)

    def test_missing_script(self):
        with self.assertRaises(FileNotFoundError):
            trace.trace_script(f"{self.tmp}/nope.py")

    def test_nested_scipy_calls_are_not_double_recorded(self):
        # chi2_contingency internally may call other stats functions; only the outer call counts.
        path = write_script(
            self.tmp,
            """
            from scipy import stats
            stats.chi2_contingency([[5, 6], [7, 9]])
            stats.mannwhitneyu([1, 2, 3, 4], [5, 6, 7, 8])
            """,
        )
        tr = trace.trace_script(path)
        self.assertEqual([r.func for r in tr.records], ["chi2_contingency", "mannwhitneyu"])


class TestRecording(unittest.TestCase):
    def test_records_and_restores(self):
        original = scipy.stats.ttest_ind
        a, b = [1.0, 2, 3, 4, 5], [2.0, 3, 4, 5, 8]
        untraced = scipy.stats.ttest_ind(a, b)
        with stataudit.recording() as calls:
            self.assertTrue(hasattr(scipy.stats.ttest_ind, ORIGINAL_ATTR))
            traced = scipy.stats.ttest_ind(a, b)
        self.assertIs(scipy.stats.ttest_ind, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(type(traced), type(untraced))
        self.assertEqual(traced.pvalue, untraced.pvalue)
        self.assertEqual(traced.statistic, untraced.statistic)

    def test_rebinds_names_imported_before_the_block(self):
        from scipy.stats import pearsonr  # bound before recording starts

        g = globals()
        g["_early_pearsonr"] = pearsonr
        try:
            with stataudit.recording() as calls:
                g["_early_pearsonr"]([1.0, 2, 3, 4], [1.0, 3, 2, 5])
            self.assertEqual(len(calls), 1)
            self.assertIs(g["_early_pearsonr"], pearsonr)  # restored
        finally:
            del g["_early_pearsonr"]

    def test_report_from_recording(self):
        r = np.random.default_rng(7)
        control = r.normal(12.0, 1.5, 60).round(1)
        treated = np.concatenate([r.normal(12.2, 1.5, 11), [21.5, 24.0, 26.5]]).round(1)
        with stataudit.recording() as calls:
            scipy.stats.ttest_ind(treated, control)
        rep = stataudit.report(calls)
        self.assertIs(rep.status, Status.FAILS)
        self.assertIs(calls.report().status, Status.FAILS)

    def test_user_errors_propagate_unchanged(self):
        with stataudit.recording() as calls:
            with self.assertRaises(TypeError):
                scipy.stats.ttest_ind()  # type: ignore[call-arg]
        self.assertEqual(len(calls), 0)

    def test_recording_failure_does_not_break_user_code(self):
        def bad_sink(rec):
            raise RuntimeError("sink exploded")

        with stataudit.recording() as calls, self.assertLogs("stataudit.trace", level="ERROR") as logs:
            trace.add_sink(bad_sink)
            try:
                res = scipy.stats.ttest_ind([1.0, 2, 3], [4.0, 5, 6])
            finally:
                trace.remove_sink(bad_sink)
        self.assertTrue(any("sink failed" in m for m in logs.output))
        self.assertTrue(np.isfinite(res.pvalue))
        self.assertEqual(len(calls), 1)

    def test_nested_recordings(self):
        with stataudit.recording() as outer:
            with stataudit.recording() as inner:
                scipy.stats.shapiro(np.arange(10.0))
            scipy.stats.shapiro(np.arange(12.0))
        self.assertEqual(len(inner), 1)
        self.assertEqual(len(outer), 2)
        self.assertFalse(hasattr(scipy.stats.shapiro, ORIGINAL_ATTR))


class TestBuildRecord(unittest.TestCase):
    def test_pandas_series_and_lists(self):
        import pandas as pd

        res = scipy.stats.ttest_ind(pd.Series([1.0, 2, 3]), [4, 5, 6])
        rec = trace.build_record("ttest_ind", (pd.Series([1.0, 2, 3]), [4, 5, 6]), {}, res)
        self.assertEqual(rec.inputs["a"].dtype, np.float64)
        self.assertAlmostEqual(rec.result["pvalue"], res.pvalue)

    def test_save_and_load_roundtrip(self):
        import tempfile

        res = scipy.stats.f_oneway([1, 2, 3], [2, 3, 4.5])
        rec = trace.build_record("f_oneway", ([1, 2, 3], [2, 3, 4.5]), {}, res, index=3)
        with tempfile.TemporaryDirectory() as d:
            trace.save_record(rec, d)
            (back,) = trace.load_records(d)
        self.assertEqual(back.index, 3)
        self.assertEqual(len(back.inputs["samples"]), 2)
        np.testing.assert_array_equal(back.inputs["samples"][1], [2, 3, 4.5])

    def test_linregress_result_fields(self):
        res = scipy.stats.linregress([1, 2, 3, 4], [2, 3, 5, 4])
        rec = trace.build_record("linregress", ([1, 2, 3, 4], [2, 3, 5, 4]), {}, res)
        self.assertAlmostEqual(rec.result["pvalue"], res.pvalue)
        self.assertAlmostEqual(rec.result["slope"], res.slope)


if __name__ == "__main__":
    unittest.main()
