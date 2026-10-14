# Testing stataudit

stataudit is tested at four levels. The first three run offline (no network,
no GitHub, no LLM) on every push and pull request; the fourth needs a real
GitHub repository and, for the agent, an API key.

| Level | Where | Runs in CI | Needs |
|---|---|---|---|
| 1. Unit and integration tests | `tests/` | yes, Python 3.10–3.13 | `git` (for `stataudit ci` tests); `pdftotext` optional |
| 2. Workflow invariants | `tests/test_workflows.py` | yes | PyYAML (installed by `.[dev]`) |
| 3. Packaging | `package` job in `.github/workflows/ci.yml` | yes | `build`, `twine` (from PyPI) |
| 4. End-to-end on GitHub | a test repository | no (manual) | a repository, optionally an LLM key |

## 1. Unit and integration tests

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m unittest discover -s tests      # the same suite without pytest
```

Every test is a `unittest.TestCase`, so both runners execute the same suite.
`tests/helpers.py` puts `src/` on `sys.path`, so the tests also run from a
checkout without installing (`PYTHONPATH=src python -m pytest`).

What they cover:

- **Statistics** (`test_stats_extra.py`): every helper (HC3, Breusch-Pagan,
  Welch's ANOVA, Cook's distance, Holm, Monte Carlo chi-square, Yuen) against
  an independent calculation.
- **Tracer** (`test_trace.py`): calls are recorded faithfully and results are
  unchanged; crashes and timeouts still yield partial traces.
- **Auditors and verdicts** (`test_audits.py`, `test_verdict.py`,
  `test_examples.py`): each check flags a bad case and passes a clean one;
  the bundled examples keep their documented verdicts.
- **Reports** (`test_reports.py`): text, Markdown, HTML (escaping), JSON.
- **check-report** (`test_reportcheck.py`): recomputed p-values equal scipy's
  (`t`, `z`, `F`, `χ²`, `Q`, and `r` against `pearsonr` on real data);
  more than 40 real-world phrasings (Markdown italics, LaTeX, Unicode minus and
  en dash, `χ²`/`chi2`/`X2`/`chi-squared`, R output style, Welch df, brackets,
  `ps <`, scientific-notation p, `ns`); rounding to the reported decimals;
  `p < .001` and `p = .000`; one-tailed tests (same sentence vs elsewhere in
  the document); inconsistent vs gross, including alpha changing the
  verdict; `p = .05` at alpha .05; impossible values; corrected p-values;
  results wrapped across lines and table rows; HTML and PDF input (the PDF is
  generated in the test and read with `pdftotext` when installed); renderers
  and CLI exit codes.
- **Notebooks** (`test_notebook.py`): magic and shell-line stripping keeps line
  numbers; cell/line mapping (markdown cells not counted); shared namespace,
  working directory, error reporting by cell; the `# %% [cell N]` format used
  by the agent.
- **lint** (`test_lint.py`): every rule has at least one case it flags and a
  close case it does not; suppression comments, `--select`/`--ignore`,
  notebooks as one program, the expected findings on every example, the CLI.
- **Agent** (`test_agent_llm.py`, `test_agent_fix.py`, `test_agent_explain.py`),
  with a scripted fake LLM transport (`tests/agent_fakes.py`) and no network:
  - the client: both wire formats, forced tool choice, `reasoning_effort`,
    retries and `Retry-After`, errors without the key, `STATAUDIT_LLM_*`
    configuration (and that generic `LLM_*` names are ignored);
  - `fix`: a real correction of `examples/01` (Welch's t-test) is verified and
    the before/after table and diff are right; a fix that drops the analysis is
    rejected; a fix that adds a failing test is rejected; a rejected proposal
    followed by a good one is accepted in round 2 with the reasons fed back;
    denylisted code (subprocess, `open(..., "w")`, `exec`) is refused before
    anything runs (a marker file is never created); a candidate that sleeps is
    stopped by the timeout; the candidate's environment has no secrets; no
    model call when nothing needs fixing; raw data reaches the model only with
    `--share-data`; propose-only/verify-only split mode and stale-candidate
    refusal; a notebook fix (cell markers enforced, outputs of changed cells
    cleared); `--apply`/`--out` only for verified fixes;
  - `explain`: grounded text is used; text with a number not in the findings
    is rejected in favour of the deterministic summary; unconfigured LLM.
- **CI and GitHub** (`test_ci.py`, `test_init.py`): `stataudit ci` on a real
  temporary git repository (changed files, config read from the base commit
  even when the PR edits it, symlinks and paths outside the checkout refused);
  `result.json` validation against hostile documents; card escaping (HTML,
  Markdown links and images, mentions, URLs, forged markers and data blocks,
  backticks in diffs), size caps; the data block's escaping, size cap,
  truncation and trailing-only parsing; the comment upsert against an
  in-memory fake GitHub API (create then update, a human-planted marker is not
  reused, head-SHA mismatch refused, harness warning, labels); the agent's CI
  propose → verify → card flow; `stataudit init` and that
  `examples/workflows/` is exactly what `init --agent` generates.
- **Interfaces** (`test_interfaces.py`): CLI commands, the Flask UI, and the
  MCP tools including `check_report` and `lint_file`.

Several tests run real subprocesses (the tracer) and take a few seconds each;
the whole suite takes about a minute.

## 2. Workflow invariants

`tests/test_workflows.py` parses every workflow in `.github/workflows/` and
`examples/workflows/` plus `action.yml`, and checks the security model of
[SECURITY_MODEL.md](SECURITY_MODEL.md) as code: no `pull_request_target`; no
`${{ }}` or `github.event.*` inside `run:` scripts; `persist-credentials:
false` on every checkout; minimal permissions per job; no secrets in jobs that
run PR code; the comment job never checks out or installs the PR and always
passes `--expect-head-sha`; in the agent workflow `propose` (holds the key)
never installs or runs the PR and checks it out into `pr/`, `verify` sees no
secret, `gate` checks nothing out; no caches; callers only pass declared
inputs; release and pages settings; in `action.yml` the key only reaches the
propose step. The test is skipped when PyYAML is not installed.

[actionlint](https://github.com/rhysd/actionlint) catches expression and
schema mistakes the invariants do not look for; run it before changing a
workflow:

```bash
actionlint .github/workflows/*.yml examples/workflows/*.yml
```

## 3. Packaging

The `package` job builds the sdist and wheel, runs `twine check --strict`,
checks that key files (templates, agent, lint, check-report) are in both,
installs the wheel into a fresh virtualenv and, from outside the checkout,
runs the installed CLI on copies of the examples: `run` (script and
notebook, verdicts asserted), `check-report`, `lint`, `explain --no-llm`,
`init --agent` into an empty repository, and `stataudit ci` on a tiny
repository with the card's data block asserted.

## 4. End-to-end on GitHub (manual)

Not automated yet. To try the workflows on a real repository:

1. In a test repository, run `stataudit init --agent --ref <branch or SHA of
   stataudit to test>` and push the files to the default branch.
2. Open a pull request that adds a copy of `examples/01_ttest_outliers.py`
   under `analysis/` and `examples/reports/results.md` under `reports/`.
3. Check the `stataudit` run's job summary and artifact, then the PR comment
   (one comment, updated on new pushes), the `stataudit: problems` label, and
   the data block at the end of the comment.
4. For the agent: set the `STATAUDIT_LLM_API_KEY` secret and the
   `STATAUDIT_LLM_PROVIDER` / `STATAUDIT_LLM_MODEL` variables, and check that
   the agent workflow's gate, propose and verify jobs run and that the card
   gains a "Suggested fix" section.

## The card's data block

Tests and bots should read results from the data block rather than scraping
the card:

```python
from stataudit.carddata import parse_card_data

data = parse_card_data(comment_body)   # None unless the block ends the card
assert data["status"] == "fail" and data["runs"][0]["does_not_hold"] == 1
```
