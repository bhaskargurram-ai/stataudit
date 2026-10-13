# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
