"""stataudit fix with a scripted fake LLM: verified corrections are accepted, bad ones are not."""

from __future__ import annotations

import json
import os
import shutil
import time
import unittest
from pathlib import Path
from unittest import mock

from agent_fakes import FAKE_KEY, ScriptedTransport, factory
from click.testing import CliRunner
from helpers import EXAMPLES, TempDirMixin

import stataudit
from stataudit.agent import fix as fx
from stataudit.agent.sandbox import scrub_env, static_check
from stataudit.cli import main as cli
from stataudit.models import Status

EX01 = (EXAMPLES / "01_ttest_outliers.py").read_text(encoding="utf-8")
WELCH = EX01.replace("stats.ttest_ind(treated, control)", "stats.ttest_ind(treated, control, equal_var=False)")
DROPPED = EX01.replace("t_stat, p_value = stats.ttest_ind(treated, control)", "t_stat, p_value = 0.0, 1.0")
WORSE = WELCH + (
    "\n# an extra correlation driven by two extreme points\n"
    "x = np.append(rng.normal(10, 2, 23), [31.0, 34.0])\n"
    "y = np.append(rng.normal(200, 25, 23), [290.0, 305.0])\n"
    "print(stats.pearsonr(x, y))\n"
)


def submit(code: str, rationale: str = "Use Welch's t-test; the variances differ 16x.", **extra):
    return [("submit_fix", {"code": code, "rationale": rationale, **extra})]


class FixBase(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.script = Path(self.tmp) / "01_ttest_outliers.py"
        self.script.write_text(EX01, encoding="utf-8")


class TestRunFix(FixBase):
    def test_welch_fix_is_verified(self):
        tr = ScriptedTransport("anthropic", [[("read_script", {}), ("show_findings", {}), ("data_summary", {})],
                                             submit(WELCH)])
        res = fx.run_fix(self.script, factory(tr), rounds=1, timeout=60)
        self.assertEqual(res.status, fx.VERIFIED, res.to_text())
        self.assertEqual(res.before_status, "does_not_hold")
        self.assertEqual(res.verification.after_status, "holds")
        (row,) = res.verification.rows
        self.assertEqual((row["outcome"], row["matched_by"], row["after_test"]), ("improved", "same data", "Welch's t-test"))
        self.assertIn("+t_stat, p_value = stats.ttest_ind(treated, control, equal_var=False)", res.diff)
        self.assertEqual(res.model, "test-model")
        self.assertGreater(res.usage.total_tokens, 0)
        # untrusted data is delimited; the key never appears in what was sent as content
        text = tr.all_text()
        self.assertIn("<<<UNTRUSTED_", text)
        self.assertNotIn(FAKE_KEY, text)
        # nothing was written to the user's file
        self.assertEqual(self.script.read_text(encoding="utf-8"), EX01)
        self.assertFalse(list(Path(self.tmp).glob(".stataudit-fix-*")))
        out = res.to_text()
        self.assertIn("Verified fix", out)
        self.assertIn("does_not_hold -> holds", out)

    def test_dropping_the_analysis_is_rejected(self):
        tr = ScriptedTransport("anthropic", [submit(DROPPED, "Removed the test.")])
        res = fx.run_fix(self.script, factory(tr), rounds=1, timeout=60)
        self.assertEqual(res.status, fx.REJECTED)
        self.assertTrue(any("no longer answered" in r for r in res.verification.reasons), res.verification.reasons)
        self.assertEqual(res.verification.rows[0]["outcome"], "dropped")

    def test_fix_that_adds_a_failing_test_is_rejected(self):
        tr = ScriptedTransport("anthropic", [submit(WORSE)])
        res = fx.run_fix(self.script, factory(tr), rounds=1, timeout=60)
        self.assertEqual(res.status, fx.REJECTED)
        self.assertTrue(any("adds a test" in r and "Pearson" in r for r in res.verification.reasons),
                        res.verification.reasons)

    def test_refinement_round_after_rejection(self):
        tr = ScriptedTransport("anthropic", [submit(DROPPED), submit(WELCH)])
        res = fx.run_fix(self.script, factory(tr), rounds=2, timeout=60)
        self.assertEqual(res.status, fx.VERIFIED)
        self.assertEqual(res.rounds, 2)
        self.assertEqual([a["accepted"] for a in res.attempts], [False, True])
        # the second request carried the verification feedback, delimited as data
        last = json.dumps(tr.requests[-1]["payload"])
        self.assertIn("did NOT accept", last)
        self.assertIn("no longer answered", last)

    def test_denylisted_code_is_refused_before_execution(self):
        marker = Path(self.tmp) / "pwned"
        evil = "import subprocess\nsubprocess.run(['touch', %r])\n" % str(marker) + WELCH
        evil2 = WELCH + f"\nopen({str(marker)!r}, 'w').write('x')\n"
        evil3 = WELCH + "\nexec('print(1)')\n"
        tr = ScriptedTransport("anthropic", [submit(evil), submit(evil2), submit(evil3), "I give up.", "Still no.", "No."])
        res = fx.run_fix(self.script, factory(tr), rounds=1, timeout=60)
        self.assertEqual(res.status, fx.NO_SUBMISSION)
        self.assertFalse(marker.exists())
        replies = [json.dumps(r["payload"]) for r in tr.requests[1:4]]
        self.assertIn("import subprocess", replies[0])
        self.assertIn("open() for writing", replies[1])
        self.assertIn("call exec()", replies[2])
        # verify_candidate refuses it too, without running anything
        before = stataudit.audit_script(self.script)
        v = fx.verify_candidate(fx.load_target(self.script), before, fx.Candidate(evil), timeout=60)
        self.assertFalse(v.accepted)
        self.assertTrue(v.static_errors)
        self.assertIsNone(v.after)
        self.assertFalse(marker.exists())

    def test_timeout_is_enforced(self):
        target = fx.load_target(self.script)
        before = stataudit.audit_script(self.script)
        slow = "import time\ntime.sleep(60)\n" + WELCH
        t0 = time.monotonic()
        v = fx.verify_candidate(target, before, fx.Candidate(slow), timeout=2)
        self.assertLess(time.monotonic() - t0, 30)
        self.assertFalse(v.accepted)
        self.assertTrue(any("timed out" in r for r in v.reasons), v.reasons)

    def test_candidate_runs_with_scrubbed_environment(self):
        env = scrub_env({"STATAUDIT_LLM_API_KEY": "x", "GITHUB_TOKEN": "y", "AWS_SECRET_ACCESS_KEY": "z",
                         "MY_PASSWORD": "p", "PATH": "/bin", "HOME": "/h", "ACTIONS_ID_TOKEN_REQUEST_URL": "u"})
        self.assertEqual(env, {"PATH": "/bin", "HOME": "/h"})
        # verify_candidate runs the candidate with scrub_env(): a secret in the parent is not visible to it
        probe = Path(self.tmp) / "probe.py"
        probe.write_text("import os\nprint(sorted(k for k in os.environ if 'LLM' in k or 'TOKEN' in k))\n", encoding="utf-8")
        from stataudit.trace import trace_script

        with mock.patch.dict(os.environ, {"STATAUDIT_LLM_API_KEY": FAKE_KEY, "GITHUB_TOKEN": "ghs_x"}):
            seen = trace_script(probe, env=scrub_env()).stdout.strip()
            leaked = trace_script(probe).stdout.strip()
        self.assertEqual(seen, "[]")
        self.assertIn("STATAUDIT_LLM_API_KEY", leaked)  # the unscrubbed control

    def test_nothing_to_fix_never_calls_the_model(self):
        good = Path(self.tmp) / "good.py"
        shutil.copy(EXAMPLES / "05_good_analysis.py", good)
        tr = ScriptedTransport("anthropic", [])
        res = fx.run_fix(good, factory(tr), timeout=60)
        self.assertEqual(res.status, fx.NOTHING_TO_FIX)
        self.assertEqual(tr.requests, [])

    def test_raw_data_only_with_share_data(self):
        data = Path(self.tmp) / "rt.csv"
        rows = ["group,ms"] + [f"control,{400 + i * 1.5:.3f}" for i in range(30)] + \
               [f"treated,{390 + i * 1.25:.3f}" for i in range(12)] + ["treated,987.654", "treated,912.321"]
        data.write_text("\n".join(rows) + "\n", encoding="utf-8")
        script = Path(self.tmp) / "csv_analysis.py"
        script.write_text(
            "import pandas as pd\nfrom pathlib import Path\nfrom scipy import stats\n"
            "df = pd.read_csv(Path(__file__).with_name('rt.csv'))\n"
            "a = df.loc[df.group == 'treated', 'ms']\nb = df.loc[df.group == 'control', 'ms']\n"
            "print(stats.ttest_ind(a, b))\n", encoding="utf-8")
        steps = lambda: [[("data_summary", {})], [("show_data", {"file": "rt.csv"})], "stop", "stop", "stop"]  # noqa: E731
        tr = ScriptedTransport("anthropic", steps())
        fx.run_fix(script, factory(tr), rounds=1, timeout=60)
        text = tr.all_text()
        self.assertIn("rt.csv: 44 rows x 2 columns", text)   # schema is shared
        self.assertNotIn("987.654", text)                     # raw rows are not
        self.assertIn("unknown tool 'show_data'", text)
        tr = ScriptedTransport("anthropic", steps())
        fx.run_fix(script, factory(tr), rounds=1, timeout=60, share_data=True)
        self.assertIn("987.654", tr.all_text())


class TestSplitMode(FixBase):
    def test_propose_only_then_verify_only(self):
        findings = stataudit.audit_script(self.script).to_dict()
        tr = ScriptedTransport("openai", [[("data_summary", {})], submit(WELCH)])
        out = Path(self.tmp) / "cand"
        res = fx.propose_only(self.script, findings, factory(tr, "openai"), out)
        self.assertEqual(res.status, fx.PROPOSED)
        self.assertTrue((out / "manifest.json").is_file())
        self.assertEqual((out / "candidate.txt").read_text(encoding="utf-8"), WELCH)
        v = fx.verify_only(self.script, out, timeout=60)
        self.assertEqual(v.status, fx.VERIFIED, v.to_text())
        # a candidate for another version of the file is refused
        self.script.write_text(EX01 + "\n# edited\n", encoding="utf-8")
        with self.assertRaises(fx.FixError):
            fx.verify_only(self.script, out, timeout=60)

    def test_propose_only_rejects_non_report(self):
        with self.assertRaises(fx.FixError):
            fx.propose_only(self.script, {"kind": "lint"}, factory(ScriptedTransport("anthropic", [])), self.tmp)


class TestNotebookFix(TempDirMixin, unittest.TestCase):
    def test_notebook_welch_fix(self):
        nb = Path(self.tmp) / "06_notebook.ipynb"
        shutil.copy(EXAMPLES / "06_notebook.ipynb", nb)
        target = fx.load_target(nb)
        self.assertIn("# %% [cell 4]", target.text)
        fixed = target.text.replace("stats.ttest_ind(new, old)", "stats.ttest_ind(new, old, equal_var=False)")
        no_markers = fixed.replace("# %% [cell 2]\n", "")
        tr = ScriptedTransport("anthropic", [submit(no_markers), submit(fixed)])
        res = fx.run_fix(nb, factory(tr), rounds=1, timeout=60)
        self.assertEqual(res.status, fx.VERIFIED, res.to_text())
        self.assertIn("keep exactly", json.dumps(tr.requests[1]["payload"]))
        self.assertEqual(res.verification.rows[0]["location"], "06_notebook.ipynb cell 4, line 2")
        new_nb = json.loads(target.patched_file_text(res.candidate.code))
        cell = [c for c in new_nb["cells"] if c["cell_type"] == "code"][3]
        self.assertIn("equal_var=False", "".join(cell["source"]))
        self.assertEqual(cell["outputs"], [])


class TestCompare(unittest.TestCase):
    def test_named_replacement_matches_transformed_data(self):
        import numpy as np

        rng = np.random.default_rng(7)
        control = rng.normal(12.0, 1.5, 60).round(1)
        treated = np.concatenate([rng.normal(12.2, 1.5, 11), [21.5, 24.0, 26.5]]).round(1)
        before = stataudit.audit_call("ttest_ind", [treated, control], label="a.py")
        before.results[0].call.site = stataudit.models.CallSite("a.py", 25)
        after = stataudit.audit_call("mannwhitneyu", [np.log(treated), np.log(control)], label="a.py")
        after.results[0].call.site = stataudit.models.CallSite("a.py", 27)
        v = fx.compare_reports(before, after)
        self.assertFalse(v.accepted)
        v = fx.compare_reports(before, after, [{"original_location": "a.py:25", "patched_location": "a.py:27"}])
        self.assertTrue(v.accepted, v.reasons)
        self.assertEqual(v.rows[0]["matched_by"], "named replacement")

    def test_same_call_site_matches_when_the_data_change(self):
        import numpy as np

        from stataudit.models import CallSite

        rng = np.random.default_rng(7)
        control = rng.normal(12.0, 1.5, 60).round(1)
        treated = np.concatenate([rng.normal(12.2, 1.5, 11), [21.5, 24.0, 26.5]]).round(1)
        line = "res = stats.ttest_ind(treated, control, equal_var=False)"
        before = stataudit.audit_call("ttest_ind", [treated, control], {"equal_var": False}, label="a.py")
        after = stataudit.audit_call("ttest_ind", [treated[:-3], control], {"equal_var": False}, label="a.py")
        before.results[0].call.site = CallSite("a.py", 25, code=line)
        after.results[0].call.site = CallSite("a.py", 26, code=line)
        v = fx.compare_reports(before, after)
        self.assertEqual(v.rows[0]["matched_by"], "same call site")
        after.results[0].call.site = CallSite("a.py", 26, code=line.replace("treated", "t2"))
        self.assertEqual(fx.compare_reports(before, after).rows[0]["outcome"], "dropped")

    def test_no_improvement_is_rejected(self):
        rep = stataudit.audit_script(EXAMPLES / "04_many_tests.py")
        self.assertIs(rep.status, Status.FRAGILE)
        v = fx.compare_reports(rep, rep)
        self.assertFalse(v.accepted)
        self.assertIn("still 'fragile'", v.reasons[0])
        good = stataudit.audit_script(EXAMPLES / "05_good_analysis.py")
        v = fx.compare_reports(good, good)
        self.assertFalse(v.accepted)
        self.assertEqual(v.reasons, ["no result improved: every conclusion has the same status as before"])


class TestStatic(unittest.TestCase):
    def test_static_check(self):
        self.assertTrue(static_check("import numpy as np\nx = np.ones(3)\n").ok)
        self.assertFalse(static_check("def (:\n").ok)
        self.assertFalse(static_check("").ok)
        for bad in ("import os", "import importlib", "from shutil import rmtree", "import socket", "import ctypes",
                    "eval('1')", "__import__('os')", "open('f', 'w')", "open('f', mode='a')", "x.__globals__",
                    "import urllib.request", "open('f', m)", "from . import x"):
            with self.subTest(code=bad):
                self.assertFalse(static_check(bad + "\n").ok)
        self.assertTrue(static_check("open('data.csv')\nopen('d.csv', 'r')\n").ok)
        # what the original already does is allowed, anything more is not
        original = "import os\npath = os.path.join('a', 'b')\n"
        self.assertTrue(static_check(original + "x = 1\n", original).ok)
        self.assertFalse(static_check(original + "import subprocess\n", original).ok)
        self.assertFalse(static_check("import os\nimport os.path\n", "import os\n").ok)


class TestFixCLI(FixBase):
    def test_apply_and_out(self):
        tr = ScriptedTransport("anthropic", [submit(WELCH)])
        out = Path(self.tmp) / "fixed.py"
        js = Path(self.tmp) / "fix.json"
        with mock.patch("stataudit.agent.common.llm_from_env", lambda **kw: factory(tr)()):
            res = CliRunner().invoke(cli, ["fix", str(self.script), "--rounds", "1", "--out", str(out), "--apply",
                                           "--json", str(js), "--timeout", "60"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertEqual(out.read_text(encoding="utf-8"), WELCH)
        self.assertEqual(self.script.read_text(encoding="utf-8"), WELCH)
        data = json.loads(js.read_text())
        self.assertEqual((data["status"], data["before_status"], data["after_status"]), ("verified", "does_not_hold", "holds"))

    def test_rejected_fix_writes_nothing(self):
        tr = ScriptedTransport("anthropic", [submit(DROPPED)])
        with mock.patch("stataudit.agent.common.llm_from_env", lambda **kw: factory(tr)()):
            res = CliRunner().invoke(cli, ["fix", str(self.script), "--rounds", "1", "--apply", "--timeout", "60"])
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertEqual(self.script.read_text(encoding="utf-8"), EX01)

    def test_missing_llm_config_is_a_usage_error(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STATAUDIT_LLM_")}
        with mock.patch.dict(os.environ, env, clear=True):
            res = CliRunner().invoke(cli, ["fix", str(self.script), "--timeout", "60"])
        self.assertEqual(res.exit_code, 2, res.output)
        self.assertIn("STATAUDIT_LLM_PROVIDER", res.output)


if __name__ == "__main__":
    unittest.main()
