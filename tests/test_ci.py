"""stataudit ci: changed files, result.json validation, the card and its data block, the comment upsert."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from agent_fakes import ScriptedTransport, factory
from click.testing import CliRunner
from helpers import EXAMPLES, TempDirMixin

from stataudit.agent.ci import agent_propose, agent_verify
from stataudit.card import CARD_MARKER, MAX_CARD_CHARS, escape_md, render_card
from stataudit.carddata import DATA_PREFIX, MAX_DATA_CHARS, card_data_block, parse_card_data
from stataudit.ci import changed_files, exit_code, run_ci, write_outputs
from stataudit.ci_result import ResultValidationError, overall_status, validate_result
from stataudit.cli import main as cli
from stataudit.config import Config, ConfigError, glob_match, load_config, parse_config
from stataudit.github_api import GitHubClient, find_card_comment
from stataudit.httpclient import HTTPResponse

HEAD = "a" * 40
BASE = "b" * 40


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def make_repo(root: Path) -> tuple[str, str]:
    """A repository whose second commit adds a flawed analysis, a report and a lint target."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    (root / "stataudit.toml").write_text('run = ["analysis/*.py"]\nreports = ["reports/*.md"]\nfail_on = "fail"\n')
    (root / "README.md").write_text("# demo\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    base = git(root, "rev-parse", "HEAD")
    (root / "analysis").mkdir()
    shutil.copy(EXAMPLES / "01_ttest_outliers.py", root / "analysis" / "ttest.py")
    shutil.copy(EXAMPLES / "05_good_analysis.py", root / "analysis" / "good.py")
    (root / "reports").mkdir()
    shutil.copy(EXAMPLES / "reports" / "results.md", root / "reports" / "results.md")
    (root / "notes.md").write_text("t(10) = 1.0, p = .01 (not under reports/, so not checked)\n")
    (root / "scratch.py").write_text("from scipy import stats\nstats.pearsonr([1, 2, 3], [2, 1, 3])\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "head")
    return base, git(root, "rev-parse", "HEAD")


def minimal_result(**over) -> dict:
    doc = {"schema": "stataudit.ci", "version": 1, "tool_version": "1.0.0", "repo": "o/r", "pr_number": 7,
           "head": HEAD, "base": BASE, "alpha": 0.05, "fail_on": "never", "status": "pass", "lint": [], "runs": [],
           "reports": [], "agent": None, "notes": []}
    doc.update(over)
    return doc


class TestRunCI(TempDirMixin, unittest.TestCase):
    def test_pr_flow(self):
        repo = Path(self.tmp) / "repo"
        base, head = make_repo(repo)
        self.assertIn("analysis/ttest.py", changed_files(repo, base, head))
        result = run_ci(repo, base=base, head=head, pr_number=12, repo_slug="o/r")
        valid = validate_result(result)
        self.assertEqual(valid["status"], "fail")
        self.assertEqual(sorted(r["path"] for r in valid["runs"]), ["analysis/good.py", "analysis/ttest.py"])
        by_path = {r["path"]: r for r in valid["runs"]}
        self.assertEqual(by_path["analysis/ttest.py"]["status"], "does_not_hold")
        self.assertEqual(by_path["analysis/good.py"]["status"], "holds")
        self.assertEqual([r["path"] for r in valid["reports"]], ["reports/results.md"])
        self.assertEqual(valid["reports"][0]["counts"]["gross"], 2)
        lint_paths = sorted(x["path"] for x in valid["lint"])
        self.assertEqual(lint_paths, ["analysis/good.py", "analysis/ttest.py", "scratch.py"])
        self.assertEqual(valid["config_source"], f"stataudit.toml@{base[:12]}")
        self.assertEqual((valid["head"], valid["base"], valid["pr_number"]), (head, base, 12))
        # the agent gets a trimmed report (no raw data) for each run
        report = by_path["analysis/ttest.py"]["report"]
        self.assertEqual(report["results"][0]["status"], "does_not_hold")
        self.assertIn("inputs", report["results"][0])
        out = Path(self.tmp) / "out"
        rj, card = write_outputs(result, out)
        text = card.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(CARD_MARKER))
        self.assertIn("does not hold", text)
        self.assertIn("F(2, 69) = 2.95", text)
        self.assertIn("SA102", text)
        data = parse_card_data(text)
        self.assertEqual(data["status"], "fail")
        self.assertEqual(data["pr"], 12)
        self.assertEqual((out / "pr_number").read_text().strip(), "12")
        self.assertEqual((out / "head_sha").read_text().strip(), head)

    def test_config_is_read_from_base(self):
        repo = Path(self.tmp) / "repo"
        base, head = make_repo(repo)
        # the PR switches the run off; the base config still applies
        (repo / "stataudit.toml").write_text('run = []\n')
        git(repo, "commit", "-qam", "weaken")
        head2 = git(repo, "rev-parse", "HEAD")
        result = run_ci(repo, base=base, head=head2)
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(load_config(repo).run, [])

    def test_files_option_and_exit_codes(self):
        repo = Path(self.tmp) / "repo"
        make_repo(repo)
        result = run_ci(repo, files=["./scratch.py", "notes.md"], config=Config(reports=["**/*.md"]))
        self.assertEqual([x["path"] for x in result["lint"]], ["scratch.py"])
        self.assertEqual(result["reports"][0]["path"], "notes.md")
        self.assertEqual(result["status"], "fail")  # notes.md has a gross inconsistency
        self.assertEqual(exit_code("fail", "fail"), 1)
        self.assertEqual(exit_code("warn", "fail"), 0)
        self.assertEqual(exit_code("warn", "warn"), 1)
        self.assertEqual(exit_code("fail", "never"), 0)

    def test_paths_outside_the_repo_are_refused(self):
        repo = Path(self.tmp) / "repo"
        make_repo(repo)
        outside = Path(self.tmp) / "secret.md"
        outside.write_text("t(10) = 1.0, p = .01\n")
        (repo / "link.md").symlink_to(outside)
        result = run_ci(repo, files=["link.md", "../secret.md", "/etc/x.md"], config=Config(reports=["**/*.md"]))
        self.assertEqual([(r["path"], r["status"]) for r in result["reports"]], [("link.md", "error")])
        self.assertIn("outside the repository", result["notes"][0])
        validate_result(result)

    def test_cli(self):
        repo = Path(self.tmp) / "repo"
        base, head = make_repo(repo)
        out = Path(self.tmp) / "o"
        res = CliRunner().invoke(cli, ["ci", "--repo", str(repo), "--base", base, "--head", head, "--out-dir", str(out),
                                       "--pr", "3"])
        self.assertEqual(res.exit_code, 1, res.output)   # fail_on = "fail" from the base config
        self.assertIn("## stataudit:", res.output)
        res = CliRunner().invoke(cli, ["ci", "--repo", str(repo), "--base", base, "--head", head, "--out-dir", str(out),
                                       "--fail-on", "never", "-q"])
        self.assertEqual(res.exit_code, 0, res.output)
        summary = Path(self.tmp) / "summary.md"
        res = CliRunner().invoke(cli, ["summary", "--json", str(out / "result.json"), "--out", str(summary)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn(CARD_MARKER, summary.read_text())


class TestConfig(TempDirMixin, unittest.TestCase):
    def test_parse_and_errors(self):
        c = parse_config({"alpha": 0.01, "fail_on": "warn", "run": "a.py", "lint_ignore": ["sa107"],
                          "agent": {"enabled": True, "max_fixes": 2}, "llm": {"provider": "openai", "model": "m"}})
        self.assertEqual((c.alpha, c.fail_on, c.run, c.lint_ignore, c.agent_enabled, c.agent_max_fixes),
                         (0.01, "warn", ["a.py"], ["SA107"], True, 2))
        for bad in ({"alpha": 2}, {"fail_on": "sometimes"}, {"run": [1]}, {"nope": 1}, {"agent": {"x": 1}},
                    {"llm": {"api_key": "k"}}, {"timeout": 0}, {"agent": {"max_fixes": 99}}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                parse_config(bad)

    def test_pyproject_fallback(self):
        Path(self.tmp, "pyproject.toml").write_text('[project]\nname="x"\n[tool.stataudit]\nalpha = 0.01\n')
        self.assertEqual(load_config(self.tmp).alpha, 0.01)
        Path(self.tmp, "stataudit.toml").write_text("alpha = 0.02\n")
        self.assertEqual(load_config(self.tmp).alpha, 0.02)

    def test_glob(self):
        self.assertTrue(glob_match("analysis/a.py", "analysis/**/*.py"))
        self.assertTrue(glob_match("analysis/x/y/a.py", "analysis/**/*.py"))
        self.assertTrue(glob_match("a.py", "**/*.py"))
        self.assertFalse(glob_match("analysisx/a.py", "analysis/**/*.py"))
        self.assertFalse(glob_match("analysis/a.pyc", "analysis/*.py"))
        self.assertFalse(glob_match("analysis/x/a.py", "analysis/*.py"))


class TestValidation(unittest.TestCase):
    def test_minimal_and_overall(self):
        v = validate_result(minimal_result())
        self.assertEqual(v["status"], "pass")
        self.assertEqual(overall_status({"runs": [{"status": "fragile"}]}), "warn")
        self.assertEqual(overall_status({"reports": [{"status": "impossible"}]}), "fail")
        self.assertEqual(overall_status({"lint": [{"status": "info"}]}), "pass")

    def test_hostile_documents(self):
        bad_cases = [
            {"schema": "proofcheck"},
            {"version": 2},
            {"status": "great"},
            {"head": "not-a-sha"},
            {"repo": "../../x"},
            {"pr_number": -1},
            {"pr_number": True},
            {"alpha": float("nan")},
            {"runs": [{"path": "/etc/passwd", "status": "holds"}]},
            {"runs": [{"path": "a/../../b.py", "status": "holds"}]},
            {"runs": [{"path": "a.py", "status": "holds", "results": [{"location": "x", "test": "t", "status": "evil"}]}]},
            {"lint": [{"path": "a.py", "status": "warn", "findings": [{"rule": "XX1", "line": 1, "severity": "warn",
                                                                       "message": "m"}]}]},
            {"reports": [{"path": "a.md", "status": "consistent", "results": [{"location": "l", "label": "x" * 201,
                                                                                "status": "consistent"}]}]},
            {"agent": {"status": "verified", "fixes": [{"path": "a.py", "status": "verified", "diff": "x" * 20001}]}},
            {"notes": ["x"] * 21},
            {"runs": [{"path": "a.py", "status": "holds", "report": {"big": "x" * 500_000}}]},
        ]
        for over in bad_cases:
            with self.subTest(over=str(over)[:80]), self.assertRaises(ResultValidationError):
                validate_result(minimal_result(**over))
        with self.assertRaises(ResultValidationError):
            validate_result([])

    def test_unknown_fields_are_dropped(self):
        v = validate_result(minimal_result(extra="<script>", runs=[{"path": "a.py", "status": "holds", "evil": 1}]))
        self.assertNotIn("extra", v)
        self.assertNotIn("evil", v["runs"][0])


HOSTILE = "<!-- stataudit:card --> @octocat see https://evil.example/x ![img](http://x) [link](javascript:alert(1)) #12 --> ` |"


class TestCard(unittest.TestCase):
    def hostile_result(self) -> dict:
        return minimal_result(
            status="fail",
            runs=[{"path": "analysis/a`b|c.py", "status": "does_not_hold", "error": HOSTILE,
                   "counts": {"holds": 0, "fragile": 0, "does_not_hold": 1, "not_audited": 0},
                   "results": [{"location": "a.py:1", "test": HOSTILE[:100], "target": HOSTILE[:100], "reported_p": 0.01,
                                "status": "does_not_hold", "verdict": HOSTILE}]}],
            reports=[{"path": "r.md", "status": "gross", "counts": {"consistent": 0, "inconsistent": 0, "gross": 1,
                                                                    "impossible": 0},
                      "results": [{"location": "line 1", "label": HOSTILE[:100], "computed_p": 0.2, "status": "gross",
                                   "message": HOSTILE}]}],
            lint=[{"path": "a.py", "status": "warn", "findings": [{"rule": "SA101", "line": 3, "cell": None,
                                                                   "severity": "warn", "message": HOSTILE}]}],
            agent={"status": "verified", "notes": [HOSTILE], "fixes": [{
                "path": "a.py", "status": "verified", "rationale": HOSTILE, "before_status": "does_not_hold",
                "after_status": "holds", "model": "m", "reasons": [],
                "diff": "--- a\n+++ b\n-x = 1\n+x = 2  ``` </details> <!-- stataudit:data {\"v\":9} -->\n",
                "rows": [{"location": "a.py:1", "before_test": "t", "before_status": "does_not_hold",
                          "after_location": "a.py:1", "after_test": "w", "after_status": "holds", "outcome": "improved"}]}]},
        )

    def test_hostile_text_is_neutralised(self):
        card = render_card(validate_result(self.hostile_result()))
        self.assertEqual(card.count(CARD_MARKER), 1)
        self.assertTrue(card.startswith(CARD_MARKER))
        self.assertNotIn("@octocat", card)
        self.assertNotIn("https://evil", card)
        self.assertNotIn("](javascript:", card)
        # outside code spans and fences (where Markdown is inert) no image or link syntax survives
        import re

        prose = re.sub(r"(`{3,})diff\n.*?\n\1", "", card, flags=re.S)
        prose = re.sub(r"`[^`\n]*`", "", prose)
        self.assertNotIn("![img]", prose)
        self.assertNotRegex(prose, r"(?<!\\)\]\((?:http|javascript)")
        # the only data block that counts is ours, at the very end
        data = parse_card_data(card)
        self.assertEqual(data["v"], 1)
        self.assertEqual(data["status"], "fail")
        # the diff fence is longer than any backtick run inside it
        self.assertIn("````diff", card)
        self.assertEqual(prose.count("<details"), prose.count("</details>"))

    def test_escape_md(self):
        s = escape_md("**bold** <b>x</b> @me #5 http://a")
        self.assertNotIn("<b>", s)
        self.assertIn("\\*\\*bold", s)
        self.assertNotIn("@me", s)

    def test_size_cap(self):
        r = self.hostile_result()
        r["agent"]["fixes"] = [dict(r["agent"]["fixes"][0], diff="+" + "y" * 19_990) for _ in range(10)]
        r["runs"] = r["runs"] * 100
        card = render_card(validate_result(r))
        self.assertLessEqual(len(card), MAX_CARD_CHARS + 10)
        self.assertIsNotNone(parse_card_data(card))

    def test_data_block(self):
        r = validate_result(self.hostile_result())
        block = card_data_block(r)
        self.assertTrue(block.startswith(DATA_PREFIX))
        inner = block[len(DATA_PREFIX):-4]
        for ch in "-<>&@":
            self.assertNotIn(ch, inner)
        self.assertEqual(json.loads(inner)["runs"][0]["path"], "analysis/a`b|c.py")
        big = copy.deepcopy(r)
        big["runs"] = [dict(big["runs"][0], path=f"p{i}/" + "x" * 70) for i in range(100)]
        big["reports"] = [dict(big["reports"][0], path=f"r{i}/" + "y" * 70) for i in range(100)]
        b = card_data_block(big)
        self.assertLessEqual(len(b), MAX_DATA_CHARS)
        self.assertTrue(json.loads(b[len(DATA_PREFIX):-4]).get("runs_truncated"))
        # alpha in scientific notation stays valid JSON after escaping
        tiny = dict(r, alpha=1e-05)
        self.assertEqual(parse_card_data("x\n" + card_data_block(tiny))["alpha"], "1e-05")
        # only a trailing block counts
        self.assertIsNone(parse_card_data(block + "\nmore text"))
        self.assertIsNone(parse_card_data("<!-- stataudit:data {not json} -->"))


class FakeGitHub:
    """A tiny in-memory GitHub API for the comment upsert."""

    def __init__(self, head: str = HEAD, files: list[str] | None = None):
        self.comments: list[dict] = [{"id": 1, "body": "<!-- stataudit:card --> planted", "user": {"login": "eve", "type": "User"}}]
        self.labels: list[str] = ["stataudit: pass", "bug"]
        self.repo_labels: set[str] = {"bug"}
        self.head = head
        self.files = files or ["analysis/a.py"]
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, headers, body, timeout):
        path = url.split("api.github.com", 1)[1]
        self.calls.append((method, path))
        data = json.loads(body) if body else None
        if method == "GET" and path.startswith("/repos/o/r/pulls/7/files"):
            return HTTPResponse(200, json.dumps([{"filename": f} for f in self.files]).encode())
        if method == "GET" and path == "/repos/o/r/pulls/7":
            return HTTPResponse(200, json.dumps({"head": {"sha": self.head}}).encode())
        if method == "GET" and path.startswith("/repos/o/r/issues/7/comments"):
            return HTTPResponse(200, json.dumps(self.comments).encode())
        if method == "POST" and path == "/repos/o/r/issues/7/comments":
            c = {"id": 100 + len(self.comments), "body": data["body"], "user": {"login": "github-actions[bot]", "type": "Bot"}}
            self.comments.append(c)
            return HTTPResponse(201, json.dumps(c).encode())
        if method == "PATCH" and path.startswith("/repos/o/r/issues/comments/"):
            cid = int(path.rsplit("/", 1)[1])
            next(c for c in self.comments if c["id"] == cid)["body"] = data["body"]
            return HTTPResponse(200, b"{}")
        if method == "GET" and path.startswith("/repos/o/r/issues/7/labels"):
            return HTTPResponse(200, json.dumps([{"name": n} for n in self.labels]).encode())
        if method == "DELETE" and "/issues/7/labels/" in path:
            from urllib.parse import unquote

            self.labels.remove(unquote(path.rsplit("/", 1)[1]))
            return HTTPResponse(204, b"")
        if method == "GET" and path.startswith("/repos/o/r/labels/"):
            from urllib.parse import unquote

            return HTTPResponse(200 if unquote(path.rsplit("/", 1)[1]) in self.repo_labels else 404, b"{}")
        if method == "POST" and path == "/repos/o/r/labels":
            self.repo_labels.add(data["name"])
            return HTTPResponse(201, b"{}")
        if method == "POST" and path == "/repos/o/r/issues/7/labels":
            self.labels.extend(data["labels"])
            return HTTPResponse(200, b"[]")
        return HTTPResponse(404, b"{}")


class TestComment(TempDirMixin, unittest.TestCase):
    def write(self, doc: dict) -> Path:
        p = Path(self.tmp) / "result.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def invoke(self, fake: FakeGitHub, *args: str):
        env = {"GITHUB_TOKEN": "ghs_test", "GITHUB_REPOSITORY": "o/r", "GITHUB_API_URL": "https://api.github.com"}
        with mock.patch.dict(os.environ, env), mock.patch("stataudit.github_api.urllib_transport", fake):
            return CliRunner().invoke(cli, ["comment", *args])

    def test_upsert_create_then_update_and_labels(self):
        fake = FakeGitHub()
        p = self.write(minimal_result(status="fail"))
        res = self.invoke(fake, "--json", str(p), "--pr", "7", "--expect-head-sha", HEAD, "--labels")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("created", res.output)
        ours = [c for c in fake.comments if c["user"]["type"] == "Bot"]
        self.assertEqual(len(ours), 1)
        self.assertTrue(ours[0]["body"].startswith(CARD_MARKER))
        self.assertEqual(fake.comments[0]["body"], "<!-- stataudit:card --> planted")  # a human's comment is not reused
        self.assertEqual(sorted(fake.labels), ["bug", "stataudit: problems"])
        res = self.invoke(fake, "--json", str(p), "--pr", "7", "--expect-head-sha", HEAD)
        self.assertIn("updated", res.output)
        self.assertEqual(len([c for c in fake.comments if c["user"]["type"] == "Bot"]), 1)

    def test_refuses_mismatched_head_and_invalid_result(self):
        fake = FakeGitHub(head="c" * 40)
        p = self.write(minimal_result())
        res = self.invoke(fake, "--json", str(p), "--pr", "7", "--expect-head-sha", HEAD)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("does not match", res.output)
        self.assertFalse(any(m == "POST" for m, _ in fake.calls))
        res = self.invoke(fake, "--json", str(p), "--pr", "7", "--expect-head-sha", "d" * 40)
        self.assertIn("result.json head", res.output)
        p2 = self.write(minimal_result(status="<b>"))
        res = self.invoke(FakeGitHub(), "--json", str(p2), "--pr", "7")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("invalid result.json", res.output)
        res = self.invoke(FakeGitHub(), "--json", str(p), "--pr", "7; rm -rf /")
        self.assertEqual(res.exit_code, 1)

    def test_harness_warning_and_dry_run(self):
        fake = FakeGitHub(files=["stataudit.toml", ".github/workflows/x.yml"])
        p = self.write(minimal_result())
        res = self.invoke(fake, "--json", str(p), "--pr", "7")
        self.assertEqual(res.exit_code, 0, res.output)
        body = [c for c in fake.comments if c["user"]["type"] == "Bot"][0]["body"]
        self.assertIn("changes files that control how this card is produced", body)
        self.assertEqual(parse_card_data(body)["warnings"], 1)
        res = self.invoke(FakeGitHub(), "--json", str(p), "--pr", "7", "--dry-run")
        self.assertTrue(res.output.startswith(CARD_MARKER))

    def test_find_card_comment(self):
        comments = [{"id": 1, "body": CARD_MARKER, "user": {"login": "x", "type": "User"}},
                    {"id": 2, "body": CARD_MARKER + " a", "user": {"login": "github-actions[bot]", "type": "Bot"}},
                    {"id": 3, "body": "other", "user": {"login": "github-actions[bot]", "type": "Bot"}}]
        self.assertEqual(find_card_comment(comments)["id"], 2)
        self.assertIsNone(find_card_comment(comments, bot_login="someone-else"))
        with self.assertRaises(Exception):
            GitHubClient(None, "bad slug")


class TestAgentCI(TempDirMixin, unittest.TestCase):
    def test_propose_then_verify(self):
        repo = Path(self.tmp) / "repo"
        base, head = make_repo(repo)
        cfg = load_config(repo)
        cfg.agent_enabled = True
        result = run_ci(repo, base=base, head=head, pr_number=7, config=cfg)
        out = Path(self.tmp) / "out"
        write_outputs(result, out)
        script = (repo / "analysis" / "ttest.py").read_text(encoding="utf-8")
        welch = script.replace("stats.ttest_ind(treated, control)", "stats.ttest_ind(treated, control, equal_var=False)")
        tr = ScriptedTransport("anthropic", [[("submit_fix", {"code": welch, "rationale": "Welch."})]])
        cands = Path(self.tmp) / "cands"
        with mock.patch("stataudit.trace.run_bootstrap", side_effect=AssertionError("propose executed code")):
            summary = agent_propose(out / "result.json", repo, cands, factory(tr), config=cfg)
        self.assertEqual([p["status"] for p in summary["proposals"]], ["proposed"])
        # propose never executed anything: only the LLM was called, with the check job's findings
        self.assertEqual(len(tr.requests), 1)
        sent = tr.all_text()
        self.assertIn("treated has 3 outliers", sent)
        self.assertIn("Mann-Whitney U", sent)
        merged = agent_verify(out / "result.json", cands, repo, timeout=60)
        v = validate_result(merged)
        self.assertEqual(v["agent"]["status"], "verified")
        fx = v["agent"]["fixes"][0]
        self.assertEqual((fx["path"], fx["before_status"], fx["after_status"]), ("analysis/ttest.py", "does_not_hold", "holds"))
        card = render_card(v)
        self.assertIn("Suggested fix (verified", card)
        self.assertIn("equal_var=False", card)
        self.assertEqual(parse_card_data(card)["agent"], {"status": "verified", "fixes": 1, "verified": 1})

    def test_disabled_agent_and_foreign_proposals(self):
        repo = Path(self.tmp) / "repo"
        base, head = make_repo(repo)
        result = run_ci(repo, base=base, head=head)
        out = Path(self.tmp) / "out"
        write_outputs(result, out)
        cands = Path(self.tmp) / "cands"
        tr = ScriptedTransport("anthropic", [])
        summary = agent_propose(out / "result.json", repo, cands, factory(tr), base=base)
        self.assertEqual(summary["proposals"], [])
        self.assertIn("disabled", summary["notes"][0])
        self.assertEqual(tr.requests, [])
        # a proposals.json naming a file that was not an analysis of this run is ignored
        (cands / "proposals.json").write_text(json.dumps({"proposals": [{"index": 0, "path": "scratch.py", "status": "proposed"}]}))
        merged = agent_verify(out / "result.json", cands, repo)
        self.assertEqual(merged["agent"]["fixes"], [])
        self.assertIn("not an analysis", merged["agent"]["notes"][-1])


if __name__ == "__main__":
    unittest.main()
