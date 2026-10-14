# Contributing to stataudit

Thanks for helping. stataudit makes claims about other people's statistics, so
correctness matters more than features.

## Setup

```bash
git clone <your fork>
cd stataudit
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests -v   # or: pytest
```

## Layout

| Path | What lives there |
|---|---|
| `src/stataudit/trace.py`, `bootstrap.py` | recording wrappers and the subprocess runner |
| `src/stataudit/stats_extra.py` | statistical helpers (OLS, HC3, Breusch-Pagan, Welch ANOVA, Holm, ...) |
| `src/stataudit/audits/` | one module per family of tests |
| `src/stataudit/verdict.py` | statuses, verdict sentences, script-level checks |
| `src/stataudit/reports/` | text, Markdown, HTML (Jinja2) and JSON renderers |
| `src/stataudit/notebook.py` | notebook execution, IPython-syntax stripping, cell/line mapping |
| `src/stataudit/reportcheck.py` | `check-report`: APA-style result extraction and p-value recomputation |
| `src/stataudit/lint.py` | `lint`: static rules SA101–SA107 |
| `src/stataudit/agent/` | optional LLM agent: client (`llm.py`), `fix.py`, `explain.py`, guard rails (`sandbox.py`), CI steps (`ci.py`) |
| `src/stataudit/ci.py`, `ci_result.py`, `card.py`, `carddata.py`, `github_api.py`, `init_cmd.py`, `config.py` | GitHub integration |
| `src/stataudit/cli.py`, `web.py`, `mcp_server.py` | interfaces |
| `.github/workflows/`, `action.yml` | reusable workflows and the composite action; `tests/test_workflows.py` checks their security invariants |
| `examples/` | analyses with known verdicts; `tests/test_examples.py` pins them. `examples/workflows/` must equal `stataudit init --agent` output (`tests/test_init.py`) |

## Rules for changes

1. **New statistical helper**: add a test comparing it with an independent
   calculation, such as a direct matrix formula, a brute-force loop, a
   published worked example, or an exact enumeration. Agreeing with your own
   implementation is not enough.
2. **New check or auditor**: add at least one test where it fires on a bad case
   and one where it stays quiet on a clean case.
3. **Never break the user's script.** Wrappers must return the original result
   object unchanged. Recording errors are logged, never raised. Auditors must
   call the original scipy functions (`stataudit._scipy.ostats` / `orig`) so
   they are not traced themselves.
4. **No new runtime dependencies** beyond numpy, scipy, pandas, click and
   jinja2 (plus `tomli` on Python 3.10). Optional integrations go behind
   `try/except ImportError`; the LLM client uses only the standard library.
5. **An LLM never decides a verdict.** Anything a model writes is validated,
   and code is executed only in a subprocess and then judged by the audit.
   Agent changes need a test with the scripted fake LLM (`tests/agent_fakes.py`).
6. **Workflows** keep the trust split of docs/SECURITY_MODEL.md; extend
   `tests/test_workflows.py` when you add a workflow or step.
7. Keep the verdict wording honest. stataudit reports evidence and computed
   alternatives. It doesn't claim to know the study design.

## Style

Type hints and docstrings on public functions. Keep modules focused. Run the
full test suite and the examples before opening a PR:

```bash
for f in examples/*.py examples/*.ipynb; do stataudit run "$f" --fail-on never -q; done
stataudit check-report examples/reports/results.md --fail-on never
stataudit lint examples/*.py examples/*.ipynb --fail-on never
```

## Reporting a wrong verdict

Please include a minimal script (synthetic data with a fixed seed is ideal),
the `--json` output, and what you expected. Wrong verdicts are treated as bugs.
