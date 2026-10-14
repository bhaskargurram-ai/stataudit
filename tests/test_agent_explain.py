"""stataudit explain: grounded LLM text is used, anything with an unsupported number is rejected."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from agent_fakes import ScriptedTransport, factory
from click.testing import CliRunner
from helpers import EXAMPLES, TempDirMixin

import stataudit
from stataudit.agent.explain import deterministic_summary, explain, numbers_in_text, ungrounded_numbers
from stataudit.cli import main as cli


def findings_01() -> dict:
    return stataudit.audit_script(EXAMPLES / "01_ttest_outliers.py").to_dict()


class TestGrounding(unittest.TestCase):
    def test_numbers_in_text(self):
        nums = [(n.raw, n.value, n.decimals, n.percent) for n in numbers_in_text(
            "p = 0.066, 3 outliers, 16× apart, 1,200 users, 5%, 3.2e-04, H0 and x2 ignored")]
        self.assertEqual([n[1] for n in nums], [0.066, 3, 16, 1200, 5, 3.2e-4])
        self.assertTrue(nums[4][3])

    def test_rounding_and_percent(self):
        f = {"p": 0.06612, "ratio": 15.9, "alpha": 0.05, "text": "Shapiro-Wilk p = 9.0e-04"}
        self.assertEqual(ungrounded_numbers("p = 0.066, about 0.07, variances 16 times, alpha 5%, p = 0.0009", f), [])
        self.assertEqual(ungrounded_numbers("p = 0.065", f), ["0.065"])
        self.assertEqual(ungrounded_numbers("a 42% drop", f), ["42%"])


class TestExplain(unittest.TestCase):
    def test_grounded_text_is_used(self):
        f = findings_01()
        text = ("Your t-test reported p = 0.00032, but three users left the tab open. Welch's t-test gives p = 0.066 "
                "and Mann-Whitney U gives p = 0.073, so the conclusion does not hold.")
        tr = ScriptedTransport("anthropic", [[("submit_explanation", {"summary": text})]])
        res = explain(f, factory(tr))
        self.assertEqual(res.source, "llm", res.notes)
        self.assertEqual(res.text, text)
        payload = tr.requests[0]["payload"]
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "submit_explanation"})
        self.assertIn("<<<UNTRUSTED_", json.dumps(payload))

    def test_ungrounded_number_is_rejected(self):
        f = findings_01()
        text = "The new flow is 23% faster (p = 0.066)."
        tr = ScriptedTransport("openai", [[("submit_explanation", {"summary": text})]])
        res = explain(f, factory(tr, "openai"))
        self.assertEqual(res.source, "deterministic")
        self.assertEqual(res.rejected_numbers, ["23%"])
        self.assertEqual(res.text, deterministic_summary(f))
        self.assertIn("does NOT hold", res.text)

    def test_model_without_submission_or_unconfigured(self):
        f = findings_01()
        tr = ScriptedTransport("anthropic", ["Here is my explanation in plain text."])
        self.assertEqual(explain(f, factory(tr)).source, "deterministic")
        self.assertEqual(explain(f, None).source, "deterministic")

    def test_deterministic_summaries_for_each_kind(self):
        rc = stataudit.check_report_file(EXAMPLES / "reports" / "results.md").to_dict()
        s = deterministic_summary(rc)
        self.assertIn("8 reported result(s)", s)
        self.assertIn("F(2, 69) = 2.95", s)
        lint = stataudit.lint_file(EXAMPLES / "04_many_tests.py").to_dict()
        self.assertIn("SA101", json.dumps(lint))
        self.assertIn("no code was run", deterministic_summary(lint))
        # the deterministic summary is itself grounded
        self.assertEqual(ungrounded_numbers(deterministic_summary(rc), rc), [])


class TestExplainCLI(TempDirMixin, unittest.TestCase):
    def test_no_llm_and_json_input(self):
        js = Path(self.tmp) / "r.json"
        res = CliRunner().invoke(cli, ["run", str(EXAMPLES / "02_pearson_leverage.py"), "-q", "--json", str(js)])
        self.assertEqual(res.exit_code, 1)
        res = CliRunner().invoke(cli, ["explain", str(js), "--no-llm"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Pearson", res.output)

    def test_llm_from_env_and_report_input(self):
        rc = stataudit.check_report_file(EXAMPLES / "reports" / "results.md").to_dict()
        good = "Two results flip significance: F(2, 69) = 2.95 gives p = 0.059 and z = 1.70 gives p = 0.0891."
        self.assertEqual(ungrounded_numbers(good, rc), [])
        tr = ScriptedTransport("anthropic", [[("submit_explanation", {"summary": good})]])
        out = Path(self.tmp) / "e.md"
        with mock.patch("stataudit.agent.common.llm_from_env", lambda **kw: factory(tr)()):
            res = CliRunner().invoke(cli, ["explain", str(EXAMPLES / "reports" / "results.md"), "--out", str(out)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertEqual(out.read_text(encoding="utf-8").strip(), good)

    def test_unconfigured_llm_falls_back(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STATAUDIT_LLM_")}
        with mock.patch.dict(os.environ, env, clear=True):
            res = CliRunner().invoke(cli, ["explain", str(EXAMPLES / "reports" / "results.md")])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("deterministic summary", res.output)


if __name__ == "__main__":
    unittest.main()
