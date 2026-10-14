"""stataudit init, and that examples/workflows is exactly what it generates."""

from __future__ import annotations

import unittest
from pathlib import Path

from click.testing import CliRunner
from helpers import ROOT, TempDirMixin

from stataudit import __version__
from stataudit.cli import main as cli
from stataudit.config import load_config
from stataudit.init_cmd import AGENT_PATH, CHECK_PATH, COMMENT_PATH, CONFIG_PATH, InitError, init_repo

EXAMPLES_WF = ROOT / "examples" / "workflows"


class TestInit(TempDirMixin, unittest.TestCase):
    def test_writes_files_and_refuses_to_overwrite(self):
        res = init_repo(self.tmp)
        self.assertEqual(res.written, [CHECK_PATH, COMMENT_PATH, CONFIG_PATH])
        for f in res.written:
            self.assertTrue((Path(self.tmp) / f).is_file())
        self.assertIn(f"check.yml@v{__version__}", (Path(self.tmp) / CHECK_PATH).read_text())
        cfg = load_config(self.tmp)
        self.assertEqual((cfg.run, cfg.reports, cfg.alpha, cfg.fail_on, cfg.agent_enabled),
                         (["analysis/**/*.py"], ["reports/**/*.md"], 0.05, "fail", False))
        with self.assertRaises(InitError):
            init_repo(self.tmp)
        before = (Path(self.tmp) / CONFIG_PATH).read_text()
        with self.assertRaises(InitError):
            init_repo(self.tmp, agent=True)  # nothing at all is written when any target exists
        self.assertFalse((Path(self.tmp) / AGENT_PATH).exists())
        self.assertEqual((Path(self.tmp) / CONFIG_PATH).read_text(), before)
        res = init_repo(self.tmp, agent=True, force=True, ref="v1.2.3")
        self.assertIn(AGENT_PATH, res.written)
        self.assertIn("agent.yml@v1.2.3", (Path(self.tmp) / AGENT_PATH).read_text())
        self.assertTrue(load_config(self.tmp).agent_enabled)
        self.assertTrue(any("STATAUDIT_LLM_API_KEY" in s for s in res.next_steps))

    def test_install_command_detection_and_validation(self):
        Path(self.tmp, "requirements.txt").write_text("statsmodels\n")
        self.assertEqual(init_repo(self.tmp).install_command, "pip install -r requirements.txt")
        self.assertIn("install-command: 'pip install -r requirements.txt'", (Path(self.tmp) / CHECK_PATH).read_text())
        with self.assertRaises(InitError):
            init_repo(self.tmp, force=True, ref="v1;rm -rf /")
        with self.assertRaises(InitError):
            init_repo(self.tmp, force=True, install_command="a\nb")

    def test_cli(self):
        res = CliRunner().invoke(cli, ["init", "--repo", self.tmp, "--agent"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("wrote .github/workflows/stataudit-agent.yml", res.output)
        self.assertIn("Next steps", res.output)
        res = CliRunner().invoke(cli, ["init", "--repo", self.tmp])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("--force", res.output)

    def test_examples_are_what_init_generates(self):
        init_repo(self.tmp, agent=True)
        pairs = {
            CHECK_PATH: "stataudit.yml",
            COMMENT_PATH: "stataudit-comment.yml",
            AGENT_PATH: "stataudit-agent.yml",
            CONFIG_PATH: "stataudit.toml",
        }
        for generated, example in pairs.items():
            with self.subTest(example=example):
                self.assertEqual((EXAMPLES_WF / example).read_text(encoding="utf-8"),
                                 (Path(self.tmp) / generated).read_text(encoding="utf-8"),
                                 f"examples/workflows/{example} is out of date: regenerate it with `stataudit init --agent`")
        self.assertEqual(sorted(p.name for p in EXAMPLES_WF.iterdir()), sorted(pairs.values()))


if __name__ == "__main__":
    unittest.main()
