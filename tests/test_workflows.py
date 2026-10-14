"""Security invariants of the GitHub workflows (skipped when PyYAML is not installed).

They encode docs/SECURITY_MODEL.md as checks, so a later edit cannot quietly
break the trust-zone split:

* no workflow is triggered by ``pull_request_target``;
* no ``run:`` script contains a ``${{ }}`` expression (values reach scripts
  through ``env:`` only, preventing script injection), in workflows and in action.yml;
* every checkout sets ``persist-credentials: false``;
* jobs that run PR code (check.yml, agent.yml's verify, ci.yml) reference no secrets
  and have read-only permissions;
* comment.yml never checks out the pull request and never installs anything from it;
* agent.yml: ``propose`` (holds the key) never installs or runs the PR, which it
  checks out into ``pr/`` as data; ``verify`` references no secrets; ``gate`` checks nothing out;
* no caches in the reusable/caller workflows; minimal permissions everywhere.
"""

from __future__ import annotations

import json
import re
import unittest

from helpers import ROOT

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional for the test suite
    yaml = None

WF_DIR = ROOT / ".github" / "workflows"
EXAMPLES = ROOT / "examples" / "workflows"
REUSABLE = ("check.yml", "comment.yml", "agent.yml")
CALLERS = ("stataudit.yml", "stataudit-comment.yml", "stataudit-agent.yml")


def load(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))  # YAML 1.1 parses the key `on` as True
    return data


def steps(job):
    return job.get("steps") or []


def checkouts(job):
    return [s for s in steps(job) if str(s.get("uses", "")).startswith("actions/checkout@")]


@unittest.skipIf(yaml is None, "PyYAML not installed")
class WorkflowSecurityTest(unittest.TestCase):
    def all_workflows(self):
        files = sorted(WF_DIR.glob("*.yml")) + sorted(EXAMPLES.glob("*.yml"))
        self.assertEqual(len(files), 12)
        return [(f, load(f)) for f in files]

    def test_no_pull_request_target_and_no_expressions_in_scripts(self):
        for path, wf in self.all_workflows():
            on = wf["on"]
            triggers = on if isinstance(on, (dict, list)) else [on]
            self.assertNotIn("pull_request_target", triggers, path.name)
            code = "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#"))
            self.assertNotIn("pull_request_target", code, path.name)
            for name, job in (wf.get("jobs") or {}).items():
                for s in steps(job):
                    if "run" in s:
                        self.assertNotIn("${{", s["run"], f"{path.name}:{name}:{s.get('name')}")
                        self.assertNotRegex(s["run"], r"github\.event\.", f"{path.name}:{name}")

    def test_every_checkout_drops_credentials(self):
        for path, wf in self.all_workflows():
            for name, job in (wf.get("jobs") or {}).items():
                for s in checkouts(job):
                    self.assertIs((s.get("with") or {}).get("persist-credentials"), False, f"{path.name}:{name}")

    def test_top_level_permissions_are_minimal(self):
        for path, wf in self.all_workflows():
            perms = wf.get("permissions")
            self.assertIn(perms, ({}, {"contents": "read"}), path.name)
            for name, job in (wf.get("jobs") or {}).items():
                jp = job.get("permissions", perms)  # a job without its own block inherits the workflow's
                writes = {k for k, v in jp.items() if v == "write"}
                allowed = {
                    ("comment.yml", "comment"): {"pull-requests"},
                    ("release.yml", "publish"): {"id-token"},
                    ("release.yml", "github-release"): {"contents"},
                    ("release.yml", "tag"): {"contents"},
                    ("pages.yml", "deploy"): {"pages", "id-token"},
                }.get((path.name, name), {"pull-requests"} if name == "comment" else set())
                self.assertLessEqual(writes, allowed, f"{path.name}:{name}")

    def test_jobs_running_pr_code_have_no_secrets(self):
        check = load(WF_DIR / "check.yml")
        job = check["jobs"]["check"]
        self.assertEqual(job["if"], "github.event_name == 'pull_request'")
        self.assertEqual(job["permissions"], {"contents": "read"})
        self.assertNotIn("secrets.", json.dumps(check))
        self.assertNotIn("github.token", json.dumps(check))
        names = [s.get("name") or s.get("uses") for s in steps(job)]
        self.assertLess(names.index("Install stataudit (isolated virtualenv)"),
                        names.index("Install the analyses' dependencies (runs PR code)"))
        install = next(s for s in steps(job) if s.get("name") == "Install the analyses' dependencies (runs PR code)")
        self.assertIn('eval "$INSTALL_COMMAND"', install["run"])
        self.assertEqual(install["env"], {"INSTALL_COMMAND": "${{ inputs.install-command }}"})
        ci = load(WF_DIR / "ci.yml")
        self.assertNotIn("secrets.", json.dumps(ci))
        self.assertEqual(list(ci["on"]), ["push", "pull_request"])

    def test_comment_workflow_never_touches_pr_code(self):
        wf = load(WF_DIR / "comment.yml")
        self.assertIn("workflow_call", wf["on"])
        job = wf["jobs"]["comment"]
        self.assertEqual(job["permissions"], {"contents": "read", "actions": "read", "pull-requests": "write"})
        for s in checkouts(job):
            self.assertEqual(s["with"]["repository"], "${{ inputs.stataudit-repository }}")  # the tool, never the PR
            self.assertNotIn("head", json.dumps(s))
        text = json.dumps(job)
        for forbidden in ("install-command", "INSTALL_COMMAND", "pull_request.head", "pip install -e", "secrets.", "eval "):
            self.assertNotIn(forbidden, text)
        post = next(s for s in steps(job) if s.get("name") == "Post or update the card")
        self.assertIn("--expect-head-sha", post["run"])

    def test_agent_zones(self):
        wf = load(WF_DIR / "agent.yml")
        jobs = wf["jobs"]
        self.assertEqual(set(jobs), {"gate", "propose", "verify"})
        self.assertEqual(checkouts(jobs["gate"]), [])

        propose = jobs["propose"]
        ptext = json.dumps(propose)
        for forbidden in ("install-command", "INSTALL_COMMAND", "eval ", "pip install -e", "pip install .", "stataudit run",
                          "agent verify", "requirements"):
            self.assertNotIn(forbidden, ptext)
        pr_checkouts = [s for s in checkouts(propose) if "repository" not in s.get("with", {})]
        self.assertEqual(len(pr_checkouts), 1)
        self.assertEqual(pr_checkouts[0]["with"]["path"], "pr")
        self.assertNotIn("write", json.dumps(propose["permissions"]))
        key_steps = [s for s in steps(propose) if "secrets." in json.dumps(s)]
        self.assertEqual([s.get("id") for s in key_steps], ["propose"])

        verify = jobs["verify"]
        vtext = json.dumps(verify)
        self.assertNotIn("secrets", vtext)
        self.assertNotIn("actions/cache", vtext)
        self.assertNotIn('"cache"', vtext)
        self.assertNotIn("STATAUDIT_LLM", vtext)
        self.assertEqual(verify["permissions"], {"contents": "read"})
        self.assertEqual(set(verify["needs"]), {"gate", "propose"})
        self.assertEqual([j for j in jobs if "secrets." in json.dumps(jobs[j])], ["gate", "propose"])
        gate_secret = re.findall(r"\$\{\{[^}]*secrets\.[^}]*\}\}", json.dumps(jobs["gate"]))
        self.assertEqual(gate_secret, ["${{ secrets.llm-api-key != '' }}"])

    def test_no_caches_outside_the_repository_ci(self):
        for path, wf in self.all_workflows():
            if path.name == "ci.yml":
                continue
            text = json.dumps(wf)
            self.assertNotIn("actions/cache", text, path.name)
            self.assertNotIn('"cache":', text, path.name)

    def test_callers(self):
        for d in (WF_DIR, EXAMPLES):
            ev = load(d / "stataudit.yml")
            self.assertEqual(list(ev["on"]), ["pull_request"])
            self.assertEqual(ev["name"], "stataudit")
            self.assertEqual(ev["permissions"], {"contents": "read"})
            for name in ("stataudit-comment.yml", "stataudit-agent.yml"):
                wf = load(d / name)
                self.assertEqual(wf["on"]["workflow_run"]["workflows"], ["stataudit"])
                self.assertEqual(wf["permissions"], {})
            for name in CALLERS:
                for job in load(d / name)["jobs"].values():
                    self.assertNotIn("steps", job)  # only calls reusable workflows
            agent = load(d / "stataudit-agent.yml")
            self.assertEqual(agent["jobs"]["comment"]["with"]["pr-number"], "${{ needs.agent.outputs.pr-number }}")
            self.assertEqual(agent["jobs"]["agent"]["secrets"], {"llm-api-key": "${{ secrets.STATAUDIT_LLM_API_KEY }}"})
            self.assertNotIn("secrets", json.dumps(ev))

    def test_callers_only_pass_declared_inputs(self):
        declared = {name: set(load(WF_DIR / name)["on"]["workflow_call"]["inputs"]) for name in REUSABLE}
        for d in (WF_DIR, EXAMPLES):
            for name in CALLERS:
                for job in load(d / name)["jobs"].values():
                    target = str(job["uses"]).split("@")[0].rsplit("/", 1)[-1]
                    with self.subTest(caller=f"{d.name}/{name}", target=target):
                        self.assertLessEqual(set(job.get("with") or {}), declared[target])

    def test_release_and_pages(self):
        rel = load(WF_DIR / "release.yml")
        self.assertEqual(rel["on"]["push"], {"tags": ["v*"]})
        self.assertEqual(set(rel["on"]["workflow_dispatch"]["inputs"]), {"version", "pypi"})
        # "Run workflow" creates the tag itself (only that job may write contents besides the release)
        self.assertEqual(rel["jobs"]["tag"]["if"], "github.event_name == 'workflow_dispatch'")
        self.assertEqual(rel["jobs"]["tag"]["permissions"], {"contents": "write"})
        for s in rel["jobs"]["build"]["steps"]:  # untrusted input reaches scripts via env only
            self.assertNotIn("${{", s.get("run", ""))
        pub = rel["jobs"]["publish"]
        self.assertEqual(pub["steps"][-1]["with"]["password"], "${{ secrets.PYPI_API_TOKEN }}")
        self.assertEqual(rel["jobs"]["publish"]["permissions"], {"id-token": "write"})
        self.assertEqual(rel["jobs"]["publish"]["environment"]["name"], "pypi")
        self.assertEqual(rel["jobs"]["github-release"]["needs"], ["build", "tag"])  # not gated on PyPI
        self.assertIn("--clobber", json.dumps(rel["jobs"]["github-release"]))
        pages = load(WF_DIR / "pages.yml")
        self.assertEqual(pages["jobs"]["deploy"]["if"], "github.event.repository.private == false")
        self.assertIn("workflow_dispatch", pages["on"])
        self.assertEqual(pages["jobs"]["deploy"]["steps"][2]["with"]["path"], "docs/site")
        ci = load(WF_DIR / "ci.yml")
        self.assertEqual(ci["jobs"]["test"]["strategy"]["matrix"]["python-version"], ["3.10", "3.11", "3.12", "3.13"])
        self.assertIn("twine check", json.dumps(ci["jobs"]["package"]))

    def test_action_yml_inputs_reach_scripts_via_env_only(self):
        action = load(ROOT / "action.yml")
        for s in action["runs"]["steps"]:
            self.assertNotIn("${{", s.get("run", ""), s.get("name"))
        commands = {"ci", "summary", "comment", "agent-propose", "agent-verify"}
        conditions = {re.search(r"inputs.command == '([a-z-]+)'", s.get("if", "")).group(1)
                      for s in action["runs"]["steps"] if s.get("if")}
        self.assertEqual(conditions, commands)
        key_steps = [s["name"] for s in action["runs"]["steps"] if "llm-api-key" in json.dumps(s.get("env", {}))]
        self.assertEqual(key_steps, ["Propose fixes (holds the LLM key; never runs PR code)"])


if __name__ == "__main__":
    unittest.main()
