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
| `src/stataudit/cli.py`, `web.py`, `mcp_server.py` | interfaces |
| `examples/` | analyses with known verdicts; `tests/test_examples.py` pins them |

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
   jinja2. Optional integrations go behind `try/except ImportError`.
5. Keep the verdict wording honest. stataudit reports evidence and computed
   alternatives. It doesn't claim to know the study design.

## Style

Type hints and docstrings on public functions. Keep modules focused. Run the
full test suite and the examples before opening a PR:

```bash
for f in examples/*.py; do stataudit run "$f" --fail-on never -q; done
```

## Reporting a wrong verdict

Please include a minimal script (synthetic data with a fixed seed is ideal),
the `--json` output, and what you expected. Wrong verdicts are treated as bugs.
