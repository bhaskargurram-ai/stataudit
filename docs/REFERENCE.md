# stataudit reference

Commands, options, configuration keys, rules, file formats and GitHub
integration of stataudit 1.0. See [AGENT.md](AGENT.md) for the agent,
[SECURITY_MODEL.md](SECURITY_MODEL.md) for the threat model and
[TESTING.md](TESTING.md) for the test suite.

## Commands

| Command | What it does | Executes code? |
|---|---|---|
| `stataudit run FILE [-- ARGS]` | run a `.py` script or `.ipynb` notebook under the tracer and audit every statistical test on its actual data | yes (your analysis) |
| `stataudit check-data DATA.csv --test T ...` | run one test on CSV columns and audit it | no |
| `stataudit check-report FILE...` | recompute the p-values of APA-style results in `.md`/`.txt`/`.html`/`.pdf` | no |
| `stataudit lint FILE...` | static checks of `.py`/`.ipynb` analysis code | no |
| `stataudit fix FILE` | *experimental*: an LLM proposes a corrected analysis, stataudit verifies it by re-running it | yes |
| `stataudit explain SOURCE` | *experimental*: plain-language summary of findings, grounded in them | only if SOURCE is an analysis |
| `stataudit ci` | check what a pull request changed; write `result.json` + `card.md` | yes (files under `run`) |
| `stataudit summary --json result.json` | render the card into the GitHub job summary | no |
| `stataudit comment --json result.json --pr N` | upsert the card as a PR comment (privileged job) | no |
| `stataudit agent propose` / `agent verify` | the agent's two CI steps | propose: no; verify: yes |
| `stataudit init` | write the GitHub workflows and `stataudit.toml` into a repository | no |
| `stataudit ui` | local web UI (needs `stataudit[web]`) | no |
| `stataudit mcp` / `stataudit-mcp` | MCP server over stdio (needs `stataudit[mcp]`) | `audit_script` only |

`python -m stataudit ...` works too. `stataudit --version` prints the version.

### `stataudit run FILE [SCRIPT_ARGS]`

| Option | Default | |
|---|---|---|
| `--alpha FLOAT` | 0.05 | significance level used to judge conclusions |
| `--html FILE`, `--md FILE`, `--json FILE` | | write reports |
| `--fail-on fragile\|fails\|never` | `fragile` | exit 1 when the overall status is at least this bad |
| `--timeout SECONDS` | 300 | stop the analysis (the whole process group is killed) |
| `--show-output` | | echo the analysis's own stdout/stderr |
| `-q`, `-v`, `--color/--no-color` | | quiet / verbose / colours |

Arguments after `--` go to the script. **Notebooks** (`.ipynb`) run their
code cells in order in one namespace (like "Run all", stopping at the first
error), in the notebook's directory, without Jupyter. IPython syntax is
replaced so that line numbers stay exact: line magics and `!shell` lines are
skipped (`%time stmt`/`%timeit stmt`/`%prun stmt` keep `stmt`), `x = !cmd` and
`x = %magic` become `x = None`, cell magics `%%time`, `%%timeit`, `%%capture`,
`%%prun`, `%%debug` keep their body and other cell magics (`%%bash`,
`%%writefile`, ...) skip the cell, help lines (`obj?`) are dropped; each
change is listed as a note. `display()` prints when IPython is not installed.
Call sites are reported as `name.ipynb cell N, line M`, where **N counts code
cells only** (1-based) and M is the line within that cell.

Exit codes: 0 when every conclusion holds (subject to `--fail-on`), 1
otherwise, 2 for usage errors.

### `stataudit check-report FILE...`

| Option | Default | |
|---|---|---|
| `--alpha FLOAT` | 0.05 | decides *gross* inconsistencies |
| `--fail-on inconsistent\|gross\|never` | `inconsistent` | exit 1 on any inconsistency or impossible value / only on gross ones and impossible values / never |
| `--json FILE` | | one object for one input, a list for several |
| `--md FILE`, `--html FILE` | | Markdown for all inputs; HTML for the first |
| `-v` | | also list consistent results |
| `-q`, `--color/--no-color` | | |

**Recognised results.** Before matching, each line is normalised: Unicode
minus and en dash become `-`, `χ`/`Χ`/math-italic chi become `χ`, `²` and other
superscripts become digits, `≤`/`≥` become `<=`/`>=`, Markdown emphasis
(`*t*`, `_p_`) and simple LaTeX (`$\chi^2$`, `\textit{p}`) are removed.
Paragraphs are joined, so results wrapped across lines are found; Markdown
table rows stand alone.

| Statistic | Forms |
|---|---|
| t | `t(28) = 2.20`, `t[28] = 2.2`, `t(27.4) = -2.10` (Welch df), R style `t = 2.20, df = 28` |
| F | `F(2, 45) = 3.81`, `F(1; 38) = 4.5`, `F(2/45)`, `F(1, 30) < 1`, R style `F = 3.2, num df = 2, denom df = 26.7` and `F = 3.2, df = 2, 45` |
| r | `r(48) = .32` (df = n − 2) |
| χ² | `χ2(2, N = 120) = 7.31`, `χ²(1)`, `χ^2(1)`, `chi2(1)`, `chi-square(1)`, `chi-squared(1)`, `X2(1)`, R style `X-squared = 7.31, df = 2` |
| z | `z = 2.31`, `Z = 2.31` |
| Q | `Q(12) = 25.1`, `Qw(...)`, `Qb(...)` (heterogeneity, χ² reference) |

followed within 90 characters (in the same sentence, before the next
statistic) by `p = .036`, `p < .001`, `p > .05`, `p ≤ .05`, `ps < .01`,
`p-value = 0.036`, `p = 2.3 × 10⁻⁵` / `p = 2.3e-5`, or `ns` / `n.s.` / "not
significant". A statistic without a p-value is skipped. Intervening
`MSE = 2.1,`-style values are allowed.

**Recomputation** uses scipy: two-tailed (or one-tailed) Student t for `t` and
`r` (via t = r·√(df/(1−r²))), the normal distribution for `z`, the upper tail of
F for `F`, of χ² for `χ²` and `Q`.

**Rounding.** A reported statistic `2.20` stands for [2.195, 2.205] (half a unit
in the last reported decimal; a decimal df likewise; integer df are exact),
and an exact `p = .036` for [.0355, .0365]. A result is consistent when some
statistic in its interval gives a p-value in the reported interval (`p < .05`
is [0, .05); `ns` is (alpha, 1]).

**Statuses**

| Status | Meaning |
|---|---|
| `consistent` | the reported p-value can follow from the reported statistic |
| `inconsistent` | it cannot, but the significance decision at alpha is the same |
| `gross` | it cannot, and the decision flips (reported significant but recomputed not, or the reverse). As in statcheck, `p = .05` counts as significant at alpha = .05 |
| `impossible` | p outside [0, 1] or exactly 0, df ≤ 0, \|r\| > 1, negative F or χ² |

**One-tailed tests.** If "one-tailed", "one-sided" or "directional test" appears
in the same sentence, the result is checked one-tailed. If it appears
elsewhere in the document and the result is consistent only one-tailed, it is
reported consistent with a note.

**Corrected p-values.** If "Bonferroni", "Holm", "Šidák", "Tukey",
"Benjamini", "FDR", "adjusted" or "corrected" appears in the clause before the
result or between statistic and p, only one thing is checked: a corrected p
cannot be smaller than the uncorrected p.

`p = .000` is accepted with a note to report `p < .001`.

**Documents.** `.md`, `.markdown`, `.txt`, `.rst` are read as text; `.html`,
`.htm` are converted to visible text (scripts and styles dropped, line numbers
refer to the extracted text); `.pdf` needs `pdftotext` (poppler-utils) and
locations are `page P, line L`.

**Not checked:** results split over table cells, tests without a df-based
reference distribution (U, W, H, ...), and exact values of corrected p-values.

### `stataudit lint FILE...`

| Option | Default | |
|---|---|---|
| `--select IDS`, `--ignore IDS` | all | comma-separated rule ids |
| `--fail-on warn\|info\|never` | `warn` | exit 1 when a finding of at least this severity exists |
| `--json FILE`, `--md FILE` | | reports |
| `-v` | | explain each rule |
| `--list-rules` | | list the rules and exit |

| Rule | Name | Severity | Fires when |
|---|---|---|---|
| SA101 | `uncorrected-multiple-tests` | warn | ≥ 3 hypothesis-test calls, or a test inside a loop or comprehension, and no multiple-comparison correction in the file (`multipletests`, `false_discovery_control`, `fdrcorrection`, Tukey HSD, names containing bonferroni/holm/fdr, or a manual `0.05 / n` / `alpha / n`) |
| SA102 | `student-t-unchecked-variances` | warn | `ttest_ind` without `equal_var=False` (or a non-zero `trim`) and no `levene`/`bartlett`/`fligner` in the file |
| SA103 | `pearson-unchecked` | warn | `pearsonr` and no scatter plot (`scatter`, `regplot`, `lmplot`, `jointplot`, `pairplot`, ...), outlier check (`zscore`, `iqr`, `quantile`, influence measures, names containing "outlier") or rank correlation in the file |
| SA104 | `chi2-small-expected` | warn / info | `chi2_contingency` on a table written literally (or assigned once to a name) whose expected counts fail Cochran's rule (any < 1 or > 20% < 5): warn. On a computed table whose expected counts are never inspected (4th return value unused, no `expected_freq`, no `fisher_exact`): info |
| SA105 | `test-shopping` | warn | the same data expressions passed to ≥ 3 different test functions |
| SA106 | `p-threshold-selection` | warn | `p < 0.05`-style comparisons (p-value names or `.pvalue` against a constant ≤ 0.1 or `alpha`) where SA101's conditions hold |
| SA107 | `normality-pretest` | info | an `if` whose condition uses a normality test (`shapiro`, `normaltest`, `jarque_bera`, `anderson`, ...) chooses between hypothesis tests |

Every finding has the rule's explanation and a suggested fix (`-v`, JSON).
Suppress one with a comment on the reported line: `# stataudit: ignore` or
`# stataudit: ignore[SA102, SA105]`. Notebooks are linted as one program
(imports in early cells count for later ones); cells that do not parse are
skipped with a note. Calls are recognised by their final name
(`stats.ttest_ind`, `scipy.stats.ttest_ind`, a `from scipy.stats import`),
unless the file defines a function of that name.

### `stataudit fix FILE` (experimental)

| Option | Default | |
|---|---|---|
| `--rounds N` | 2 (1–5) | proposals; each rejected one is fed back with the reasons |
| `--apply` | | overwrite FILE, only if the fix is verified |
| `--out FILE` | | write the corrected file, only if verified |
| `--json FILE`, `--md FILE` | | the fix result |
| `--share-data` | off | let the model see raw values (first CSV rows, first traced values) |
| `--timeout SECONDS` | 120 | per analysis run |
| `--max-tool-calls N` | 12 (1–50) | per proposal |
| `--alpha`, `--config FILE`, `-q` | | |
| `--propose-only --findings F --out-dir D` | | CI step 1: propose from a findings JSON; executes nothing |
| `--verify-only D` | | CI step 2: re-audit original and candidate; no LLM |

Exit 0 for `verified`, `nothing_to_fix` or (with `--propose-only`)
`proposed`; 1 for `rejected`, `no_submission`, `error`; 2 when the LLM is
not configured. Details in [AGENT.md](AGENT.md).

### `stataudit explain SOURCE` (experimental)

SOURCE: a stataudit JSON report, a `.py`/`.ipynb` analysis (audited first) or
a report document (checked first). Options: `--out FILE` (Markdown),
`--json FILE`, `--no-llm`, `--alpha`, `--config FILE`. Always exits 0 after
printing either the grounded LLM text or the deterministic summary.

### `stataudit ci`

| Option | Default | |
|---|---|---|
| `--repo DIR` | `.` | repository checkout |
| `--base REV` | | changed files are `git diff --diff-filter=ACMR BASE...HEAD`; the config is read from BASE |
| `--head REV` | `HEAD` | |
| `--files PATH` (repeatable) | | check these paths instead of a diff |
| `--all` | | lint/check every tracked file |
| `--config FILE` | `stataudit.toml` | |
| `--out-dir DIR` | `stataudit-out` | writes `result.json`, `card.md`, `pr_number`, `head_sha` |
| `--fail-on config\|never\|warn\|fail` | `config` | `config` uses `fail_on` from the configuration |
| `--pr N` | from `$GITHUB_EVENT_PATH` | |
| `-q` | | |

Without `--base` (and without `--files`), every tracked file is considered.
Lint runs on changed `.py`/`.ipynb` matching `lint`; check-report on changed
`.md`/`.markdown`/`.txt`/`.rst`/`.html`/`.htm` (and `.pdf` when `pdftotext`
exists) matching `reports`; `run` executes **every** tracked `.py`/`.ipynb`
matching `run` (changed or not, since a data change can change a result).

**Overall status:** `fail` if an analysis does not hold or a reported result
is a gross inconsistency or impossible; `warn` for fragile analyses,
inconsistent results, lint warnings, and files that could not be processed;
otherwise `pass`.

### `stataudit summary`, `stataudit comment`

`summary --json result.json [--out FILE]` validates result.json, renders the
card and appends it to `--out`, else `$GITHUB_STEP_SUMMARY`, else stdout.

`comment --json result.json [--pr N] [--repo owner/name] [--expect-head-sha SHA]
[--labels] [--bot-login LOGIN] [--check-pr-files/--no-check-pr-files] [--dry-run]`
validates result.json, re-renders the card, refuses unless result.json and the
PR's current head match `--expect-head-sha`, adds a warning when the PR changes
workflow or configuration files (`--check-pr-files`, on by default), and
creates or updates one comment starting with `<!-- stataudit:card -->` written
by a bot. `--labels` keeps exactly one of `stataudit: pass`,
`stataudit: warnings`, `stataudit: problems` on the PR (created with colours if
missing; other labels untouched). Needs `GITHUB_TOKEN` (`pull-requests: write`)
and `GITHUB_REPOSITORY`; honours `GITHUB_API_URL`.

### `stataudit agent propose` / `stataudit agent verify`

`agent propose --result result.json --repo-dir pr --base SHA --out-dir DIR`:
for each analysis in result.json that is fragile or does not hold (up to
`[agent] max_fixes`), ask the model for a fix, reading the checkout as data.
Does nothing unless `[agent] enabled = true` in the config at `--base`. Writes
`DIR/proposals.json` and `DIR/<index>/{manifest.json,candidate.txt}`.

`agent verify --result result.json --candidates DIR --repo-dir . --json OUT
[--out card.md] [--timeout 300]`: re-run each original and candidate, merge an
`agent` section into result.json. Proposals for paths that were not analyses
of the run are ignored.

### `stataudit init`

`init [--repo DIR] [--agent] [--force] [--ref REF] [--install-command CMD]`
writes `.github/workflows/stataudit.yml`, `.github/workflows/stataudit-comment.yml`,
with `--agent` `.github/workflows/stataudit-agent.yml`, and `stataudit.toml`
(with `--agent`, `[agent] enabled = true`). Refuses to overwrite any existing
target without `--force` (and then writes nothing). `--ref` defaults to
`v<version>`. The install command is detected: `pip install -r requirements.txt`
if `requirements.txt` exists, else `pip install -e .` for a `pyproject.toml`
or `setup.py`, else empty. `examples/workflows/` contains exactly what
`stataudit init --agent` writes into an empty repository.

## Configuration: `stataudit.toml`

Also read from `.stataudit.toml` or `[tool.stataudit]` in `pyproject.toml`. In
CI it is read from the pull request's base commit. Unknown keys are errors.

| Key | Default | |
|---|---|---|
| `alpha` | `0.05` | |
| `fail_on` | `"fail"` | `never` \| `warn` \| `fail`: exit code of `stataudit ci` |
| `run` | `[]` | globs of analyses executed in CI (opt-in) |
| `reports` | `["**/*.md", "**/*.txt", "**/*.html", "**/*.htm"]` | globs checked with check-report when changed (`init` writes `["reports/**/*.md"]`) |
| `lint` | `["**/*.py", "**/*.ipynb"]` | globs linted when changed |
| `lint_ignore` | `[]` | rule ids |
| `timeout` | `300` | seconds per executed analysis (1–7200) |
| `max_files` | `50` | per category per run (1–1000) |
| `[agent] enabled` | `false` | let `stataudit-agent.yml` propose fixes |
| `[agent] max_fixes` | `3` | analyses per PR (1–20) |
| `[agent] max_tool_calls` | `12` | per proposal (1–50) |
| `[agent] share_data` | `false` | let the model see raw rows |
| `[llm] provider`, `model`, `base_url` | | fallbacks for `STATAUDIT_LLM_*`; the API key is never read from a file |

Globs use `/`, `*` (within a directory), `?` and `**` (any number of
directories).

## Environment variables

| Variable | Used by |
|---|---|
| `STATAUDIT_LLM_PROVIDER`, `STATAUDIT_LLM_MODEL`, `STATAUDIT_LLM_API_KEY`, `STATAUDIT_LLM_BASE_URL`, `STATAUDIT_LLM_REASONING_EFFORT` | `fix`, `explain`, `agent propose` |
| `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `GITHUB_API_URL` | `comment` |
| `GITHUB_EVENT_PATH` | `ci`, `comment` (PR number) |
| `GITHUB_STEP_SUMMARY` | `summary` |
| `MPLBACKEND` | set to `Agg` for traced analyses unless already set |

## GitHub integration

`stataudit init` writes callers of three reusable workflows in this
repository (pin them to a release tag or commit SHA):

### `check.yml` (unprivileged, `pull_request` only)

| Input | Default | |
|---|---|---|
| `stataudit-ref` | `v1.0.0` | keep equal to the `@ref` of the workflow |
| `stataudit-repository` | `bhaskargurram-ai/stataudit` | |
| `python-version` | `"3.12"` | |
| `install-command` | `""` | installs what the analyses import (runs PR code; stataudit's virtualenv is first on `PATH`) |
| `fail-on` | `never` | `never` \| `warn` \| `fail` |
| `timeout-minutes` | `30` | |

Output: `status` (`pass` \| `warn` \| `fail`). Uploads the artifact
`stataudit-result` (`result.json`, `card.md`, `pr_number`, `head_sha`) and
writes the card to the job summary.

### `comment.yml` (privileged, `workflow_run` only)

| Input | Default | |
|---|---|---|
| `stataudit-ref`, `stataudit-repository`, `python-version` | as above | |
| `labels` | `true` | set the `stataudit: pass\|warnings\|problems` label |
| `pr-number` | `""` | trusted PR number (the agent passes the gate's output); default: from the artifact, validated |
| `artifact-run-id` | `""` | run that uploaded `stataudit-result`; default: the triggering run |

Job permissions: `contents: read`, `actions: read`, `pull-requests: write`.

### `agent.yml` (experimental, `workflow_run` only)

| Input | Default | |
|---|---|---|
| `stataudit-ref`, `stataudit-repository`, `python-version` | as above | |
| `install-command` | `""` | used in `verify` only |
| `llm-provider`, `llm-model`, `llm-base-url`, `llm-reasoning-effort` | `""` | the caller passes `vars.STATAUDIT_LLM_*` |
| `run-for` | `trusted` | `trusted` (owners/members/collaborators or the label) \| `everyone` |
| `label` | `stataudit:agent` | opt-in label for other authors |
| `timeout-minutes` | `30` | verify job |

Secret: `llm-api-key` (the caller passes `secrets.STATAUDIT_LLM_API_KEY`).
Outputs: `pr-number`, `verified`. Repository settings for the generated
caller: secret `STATAUDIT_LLM_API_KEY`; variables `STATAUDIT_LLM_PROVIDER`,
`STATAUDIT_LLM_MODEL`, optional `STATAUDIT_LLM_BASE_URL`,
`STATAUDIT_LLM_REASONING_EFFORT`.

### Composite action (`action.yml`)

`uses: bhaskargurram-ai/stataudit@v1.0.0` with `command:` one of `ci`
(default), `summary`, `comment`, `agent-propose`, `agent-verify`. Inputs:
`base`, `head`, `config`, `fail-on` (`config`), `output-dir`
(`stataudit-out`), `result` (`stataudit-out/result.json`), `pr-number`,
`expected-head-sha`, `labels`, `github-token`, `repo-dir`, `llm-provider`,
`llm-model`, `llm-base-url`, `llm-reasoning-effort`, `llm-api-key`,
`stataudit-version`. Output: `status`. The reusable workflows above are the
recommended way to use it; they already put each command in the right trust
zone.

## The card and its data block

The card is Markdown starting with `<!-- stataudit:card -->`: a headline
(✅ no problems / ⚠️ needs a second look / ❌ a conclusion does not hold or a
reported result is wrong), then sections for analyses re-run, reported results
recomputed, lint findings, and the agent's fixes (with the diff in a
collapsible block). It ends with one line:

```
<!-- stataudit:data {"v":1,"tool":"1.0.0","pr":12,"head":"<sha>","status":"fail","alpha":"0.05",
  "runs":[{"path":..,"status":..,"holds":0,"fragile":0,"does_not_hold":1}],
  "reports":[{"path":..,"status":..,"consistent":5,"inconsistent":1,"gross":2,"impossible":0}],
  "lint":{"files":3,"warn":2,"info":0,"rules":["SA102"]},
  "agent":{"status":"verified","fixes":1,"verified":1} | null, "warnings":0} -->
```

(one line in the card). `-`, `<`, `>`, `&`, `@` inside it are `\u` escapes;
`runs` and `reports` hold at most 20 entries and are dropped (with
`runs_truncated` / `reports_truncated`) to keep the block under 4,000
characters. `stataudit.carddata.parse_card_data(markdown)` returns the block
only when it is the last thing in the card.

## JSON outputs

All JSON documents carry `"tool": "stataudit"` and a `"schema"` number.

* `run --json`: `version`, `source`, `alpha`, `status` (`holds` \| `fragile` \|
  `does_not_hold` \| `not_audited`), `counts`, `script_error`,
  `script_findings`, `multiplicity`, `meta`, and `results[]` with `location`,
  `code`, `test`, `target`, `reported_p`, `status`, `verdict`, `findings`,
  `alternatives`, `effect_sizes`, `sample_sizes` and `inputs` (per input: shape,
  n, missing, mean, sd, min, max).
* `check-report --json`: `kind: "check-report"`, `status`, `counts`, and
  `results[]` with `kind`, `label`, `line`, `page`, `df1`, `df2`,
  `statistic`, `p_op`, `p`, `tails`, `adjusted`, `computed_p`,
  `computed_p_range`, `status`, `message`, `notes`.
* `lint --json`: `kind: "lint"`, `status` (`clean` \| `info` \| `warn`),
  `counts`, `tests_found`, `findings[]` with `rule`, `name`, `severity`,
  `line`, `cell`, `message`, `code`, `explanation`, `fix`.
* `fix --json`: `kind: "fix"`, `status`, `before_status`, `after_status`,
  `rationale`, `replacements`, `verification` (`accepted`, `reasons`, `rows`),
  `diff`, `attempts`, `usage`, `model`.
* `explain --json`: `kind: "explain"`, `source` (`llm` \| `deterministic`),
  `text`, `rejected_numbers`, `notes`, `usage`.
* `ci` (`result.json`): `schema: "stataudit.ci"`, `version: 1`, `status`
  (`pass` \| `warn` \| `fail`), `head`, `base`, `pr_number`, `lint[]`, `runs[]`,
  `reports[]`, `agent`, `notes`; validated by `stataudit.ci_result.validate_result`.

## MCP server

`stataudit-mcp` (stdio) offers four tools: `audit_script(path, alpha)` (a local
`.py` or `.ipynb`, subprocess, 120 s timeout), `audit_test(test_name,
data_json, alpha)`, `check_report(text | path, alpha)` and `lint_file(path)`.
The last two execute nothing.
