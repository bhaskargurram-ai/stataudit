# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-14

### Added

- `stataudit check-report FILE...`: statcheck-style recomputation of APA-style
  results (`t`, `F`, `r`, `χ²`, `z`, `Q`, plus R's `t.test`/`chisq.test`/
  `oneway.test` print styles) in Markdown, text, HTML and PDF (via
  `pdftotext`). Allows for rounding to the reported decimals, handles
  `p < .001`, `ns`, one-tailed tests, corrected p-values, Unicode minus and
  `χ²`, results wrapped across lines; flags inconsistent results, gross
  inconsistencies (the decision flips at `--alpha`) and impossible values.
  Text, Markdown, HTML and JSON output.
- Notebooks: `stataudit run analysis.ipynb` executes the code cells in order
  under the tracer (IPython magics and shell lines are skipped or rewritten
  with a note) and reports call sites as `cell N, line M`.
- `stataudit lint FILE.py|.ipynb`: static rules SA101–SA107 (uncorrected
  multiple tests, Student's t-test without a variance check, Pearson without a
  plot or outlier check, chi-square with small expected counts, test shopping,
  p-threshold selection, normality pre-testing), each with an explanation and
  a suggested fix; suppression comments; `--select`/`--ignore`.
- Experimental agent (bring your own key; `STATAUDIT_LLM_PROVIDER`,
  `STATAUDIT_LLM_MODEL`, `STATAUDIT_LLM_API_KEY`, `STATAUDIT_LLM_BASE_URL`,
  `STATAUDIT_LLM_REASONING_EFFORT`; Anthropic Messages and OpenAI-compatible
  chat completions):
  - `stataudit fix`: the model proposes a corrected analysis through
    read-only tools; stataudit runs it in a subprocess (timeout, scrubbed
    environment, static denylist) under the tracer and accepts it only if
    every original question is still answered and the audit improves.
    Refinement rounds, `--apply`/`--out`, and `--propose-only`/`--verify-only`
    for CI.
  - `stataudit explain`: a plain-language summary whose every number must
    appear in the findings, with a deterministic fallback.
- GitHub integration: `stataudit ci`, `summary`, `comment`, `agent propose`,
  `agent verify`; reusable workflows `check.yml` (unprivileged), `comment.yml`
  (privileged `workflow_run`, re-validates the artifact) and `agent.yml`
  (gate / propose with the key / verify without secrets); a composite
  `action.yml`; a sticky PR card with a `stataudit: pass|warnings|problems`
  label and a machine-readable `<!-- stataudit:data ... -->` block.
- `stataudit init [--agent]` writes the caller workflows and a commented
  `stataudit.toml` (configuration is read from the PR's base commit in CI);
  `examples/workflows/` is kept identical to its output by a test.
- MCP tools `check_report` and `lint_file`; `audit_script` accepts notebooks.
- Python API: `audit_notebook`, `check_report_text`, `check_report_file`,
  `lint_code`, `lint_file`.
- Examples: `06_notebook.ipynb` and `reports/results.md`.
- Release engineering: packaging job in CI (sdist + wheel, `twine check`,
  installed-wheel smoke tests), `release.yml` (PyPI trusted publishing and a
  GitHub release from this changelog), `pages.yml`; docs/AGENT.md,
  docs/SECURITY_MODEL.md, docs/TESTING.md, docs/REFERENCE.md.

### Changed

- JSON run reports include per-input summaries (`inputs`: n, missing, mean,
  sd, min, max) for each result, and notebook call sites carry `cell`.
- `stataudit run` accepts `.ipynb` files.
- Python 3.10 needs `tomli` (installed automatically) to read `stataudit.toml`.

### Fixed

- A timed-out analysis is now stopped together with every process it
  started (the whole process group is killed), not only the Python process.
- Analysis output that is not valid UTF-8 no longer breaks tracing.

## [0.1.0] - 2026-10-13

### Added

- Tracer that runs an analysis script in a subprocess and records the inputs,
  options, results and call sites of 13 `scipy.stats` tests (`ttest_ind`,
  `ttest_rel`, `ttest_1samp`, `pearsonr`, `spearmanr`, `f_oneway`,
  `chi2_contingency`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `linregress`,
  `shapiro`, `fisher_exact`), plus statsmodels OLS fits when statsmodels is
  installed. Crashes and timeouts still yield a partial trace.
- `stataudit.recording()` context manager for notebooks.
- Auditors with assumption checks and alternative analyses: Welch, Yuen and
  Mann-Whitney for t-tests; Wilcoxon for paired and one-sample tests;
  Spearman, influence and leave-one-out checks for Pearson; Breusch-Pagan,
  HC3 and influence checks for regression; Welch's ANOVA and Kruskal-Wallis;
  Fisher's exact and Monte Carlo chi-square for sparse tables.
- Script-level checks: Holm-Bonferroni across all p-values, repeated testing
  of the same data, very small samples.
- Verdicts (holds / fragile / does not hold) with plain-English explanations.
- Text, Markdown, self-contained HTML (light/dark) and JSON reports.
- CLI: `stataudit run`, `stataudit check-data`, `stataudit ui`,
  `stataudit mcp`; `python -m stataudit`.
- Local Flask web UI and an MCP server (`stataudit-mcp`).
- Five example analyses with known verdicts.
