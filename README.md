<p align="center">
  <img src="docs/assets/banner.svg" alt="stataudit: Does the conclusion survive?" width="880">
</p>

<p align="center">
  <b>stataudit checks whether a statistical conclusion survives its own data.</b><br>
  It runs your analysis, records every test it performs, checks each test's assumptions on the data it actually received,<br>
  computes the alternatives a careful statistician would also run, and says whether the result still holds.
</p>

<p align="center">
  <a href="https://github.com/bhaskargurram-ai/stataudit/actions/workflows/ci.yml"><img src="https://github.com/bhaskargurram-ai/stataudit/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.10 to 3.13">
  <img src="https://img.shields.io/badge/version-1.0.0-informational" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/agent-experimental-orange" alt="Agent: experimental">
</p>

It doesn't matter whether a person or an AI assistant wrote the analysis. stataudit only looks at what the code did to the data.

<p align="center">
  <img src="docs/images/report-example-01.png" alt="The stataudit HTML report for examples/01_ttest_outliers.py: at least one conclusion does not hold. Student's t-test at line 25 reports p = 3.2e-04, but treated has 3 outliers, variances differ 15.9 times and treated is non-normal (n = 14). Welch's t-test gives p = 0.066, Yuen's 20% trimmed t-test 0.342 and Mann-Whitney U 0.073, none significant." width="760">
  <br><sub>Real output: <code>stataudit run examples/01_ttest_outliers.py --html report.html</code>, cropped below the result card. The script reported a significant difference; nothing that accounts for the three outliers agrees.</sub>
</p>

> [!NOTE]
> **Status: 1.0.0.** stataudit traces 13 `scipy.stats` tests plus statsmodels OLS, in scripts and Jupyter notebooks. It also checks reported results in text (`check-report`) and lints analysis code without running it (`lint`). The agent that proposes fixes (`fix`, `explain`) is **experimental and off by default**; it needs your own LLM API key, and nothing it writes counts until stataudit has run it.

## Contents

- [Why audit an analysis?](#why-audit-an-analysis)
- [How it works](#how-it-works)
- [Four analyses that look fine](#four-analyses-that-look-fine)
- [Quick start](#quick-start)
- [What each auditor checks](#what-each-auditor-checks)
- [How verdicts work](#how-verdicts-work)
- [Checking reported numbers](#checking-reported-numbers-check-report)
- [Linting without running](#linting-without-running-lint)
- [Agent: fix and explain (experimental)](#agent-fix-and-explain-experimental)
- [MCP server and web UI](#mcp-server-and-web-ui)
- [GitHub integration](#github-integration)
- [Security model](#security-model)
- [How stataudit compares](#how-stataudit-compares)
- [Configuration](#configuration) · [CLI reference](#cli-reference)
- [FAQ](#faq) · [Limitations](#limitations) · [Roadmap](#roadmap)
- [Contributing](#contributing) · [License](#license)

## Why audit an analysis?

More and more statistics is done by asking an AI assistant. The code runs and a p-value comes out, but the conclusion can still be wrong, because the test's assumptions don't fit the data:

- a Pearson correlation produced by two extreme points;
- a Student t-test on small, skewed samples whose variances differ sixteenfold;
- a chi-square test on a table whose expected counts are mostly below 5;
- an OLS slope whose standard error ignores heteroskedastic residuals;
- fifteen metrics tested, three "significant", none corrected.

None of these crash, and all of them look plausible. People make the same mistakes: in a static analysis of 45 Python data-science notebooks, only 11 checked every assumption their tests rely on and only 2 applied any multiple-comparison correction ([*The Fault in our Stats*, ASE 2025](https://conf.researchr.org/details/ase-2025/ase-2025-papers/226/The-Fault-in-our-Stats)).

stataudit doesn't judge the code or the author. It asks the questions a reviewing statistician asks first:

- **What did each test actually receive?** Not the variable names: the arrays, after your filtering and cleaning.
- **Do its assumptions hold on that data?** Normality, outliers, equal variances, leverage, expected counts.
- **What do the robust alternatives say?** Welch instead of Student, Spearman instead of Pearson, Fisher instead of chi-square, HC3 standard errors, Holm across all p-values.
- **Does the conclusion survive?** If every analysis a failed check calls for lands on the other side of α, it doesn't.

A careful analysis gets ✅ and no alternatives at all. ⚠️ means **fragile**, not wrong.

## How it works

<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="stataudit run executes a script or notebook in a subprocess and traces every scipy.stats and statsmodels OLS call. Each call is audited for normality, outliers, unequal variances, leverage, sparse expected counts and the number of p-values. Only when a check fails are alternatives computed on the same data. Fixed rules then give a verdict: holds, fragile or does not hold. Separately, stataudit lint and stataudit check-report run without executing anything. Results go to terminal, HTML, Markdown, JSON and PR comment reports." width="880">
</p>

1. **Trace.** `stataudit run` executes your script or notebook in a subprocess with a timeout. Thin wrappers around `scipy.stats` (and statsmodels OLS, when installed) record each call's input arrays, options, result and call site (`analysis.py:25`, or `cell 4, line 2` in a notebook). Your code runs unchanged and gets the real return values.
2. **Audit.** Each recorded call goes to the auditor for that test, which checks the assumptions on the recorded data. Every finding carries its evidence: Shapiro-Wilk p and skew, robust z of each outlier, variance ratio and Levene p, Cook's distance, expected counts.
3. **Alternatives.** Only when a check fails, stataudit computes the analyses that check calls for, on the same data: Welch's t when variances differ, Yuen's trimmed t and Mann-Whitney U with outliers, Spearman and a refit without high-leverage points for a correlation, HC3 standard errors for a regression, Fisher's exact or a Monte Carlo test for a sparse table, and Holm-Bonferroni across every p-value in the script.
4. **Verdict.** Fixed rules compare the conclusions at α: ✅ **holds**, ⚠️ **fragile** or ❌ **does not hold**, with one plain-English sentence per test. No LLM is involved.

Two more commands work without running anything: [`check-report`](#checking-reported-numbers-check-report) recomputes p-values from results reported in text, and [`lint`](#linting-without-running-lint) reads analysis code as a syntax tree.

## Four analyses that look fine

Each bundled example is the kind of analysis an assistant writes when asked a simple question. Every verdict below is real output of `stataudit run examples/0X_*.py`, and [`tests/test_examples.py`](tests/test_examples.py) pins them.

<p align="center">
  <img src="docs/assets/does-it-survive.svg" alt="Example 01 measured. Left: control has 60 values between 8 and 15 minutes with mean 11.7; treated has 11 values between 10 and 14 plus 3 users at 21.5, 24 and 26.5 minutes, which pull its mean from 12.0 to 14.6; variances differ 16 times. Right: on a log p-value scale with alpha = 0.05, Student's t-test, which the script ran, gives p = 0.0003; Welch's t gives 0.066, Mann-Whitney U 0.073 and Yuen's trimmed t 0.342, all above alpha. Verdict: does not hold." width="880">
</p>

| Example | What the script concluded | What stataudit found | Verdict |
|---|---|---|---|
| [`01_ttest_outliers.py`](examples/01_ttest_outliers.py)<br><sub>"Is the new onboarding flow different?"</sub> | Student's t-test, p = 3.2e-04 | treated has 3 outliers and is non-normal (n = 14); variances differ 15.9×. Welch p = 0.066, Yuen p = 0.342, Mann-Whitney p = 0.073 | ❌ does not hold |
| [`02_pearson_leverage.py`](examples/02_pearson_leverage.py)<br><sub>"Does marketing spend predict sign-ups?"</sub> | Pearson r = 0.81, p = 1.2e-06 | 2 high-leverage points (two national campaigns) create the correlation. Without them r = 0.06, p = 0.774; Spearman ρ = 0.29, p = 0.154 | ❌ does not hold |
| [`03_chisq_sparse.py`](examples/03_chisq_sparse.py)<br><sub>"Does churn reason depend on plan?"</sub> | chi-square on a 3×4 table, p = 0.03 | 8 of 12 cells (67 %) have expected count < 5, minimum 0.07. Monte Carlo permutation test (20,000 tables, fixed margins and seed): p = 0.08 | ❌ does not hold |
| [`04_many_tests.py`](examples/04_many_tests.py)<br><sub>"Which of 15 metrics changed?"</sub> | 15 Welch t-tests, 3 significant | no assumption problems, but with 15 tests the chance of at least one false positive is 54 %. After Holm, 2 of the 3 survive; one (p = 0.02) gets adjusted p = 0.256 | ⚠️ fragile |
| [`05_good_analysis.py`](examples/05_good_analysis.py) | Welch t-test, p = 0.003 | no assumption problems, so no alternatives are computed | ✅ holds |

The terminal report for example 01:

```text
$ stataudit run examples/01_ttest_outliers.py
stataudit report — examples/01_ttest_outliers.py
────────────────────────────────────────────────────────────────────────────────────────
❌ At least one conclusion does NOT hold  (1 test(s): 0 hold, 0 fragile, 1 do not hold; alpha = 0.05)

❌ [1] Student's t-test — treated vs control  (01_ttest_outliers.py:25)
    > t_stat, p_value = stats.ttest_ind(treated, control)
    Your Student's t-test (01_ttest_outliers.py:25) reports p = 3.2e-04 (significant).
    Treated has 3 outliers, variances differ 16× and treated is non-normal (n = 14);
    Welch's t-test gives p = 0.066, Yuen's 20% trimmed t-test gives p = 0.342 and
    Mann-Whitney U gives p = 0.073 — the conclusion does NOT hold.
      ! treated looks non-normal (Shapiro-Wilk p = 9.0e-04, skew = 1.53, excess kurtosis
      = 1.00)
      ! treated has 3 outliers (robust |z| > 3.5, max 7.3)
      ! Variances differ 15.9× (Levene p = 5.0e-04)
      alternatives on the same data:
        - Welch's t-test: p = 0.066 (n.s.)
        - Yuen's 20% trimmed t-test: p = 0.342 (n.s.)
        - Mann-Whitney U: p = 0.073 (n.s.)  [rank-biserial r = 0.31; tests whether one
        group tends to have larger values, not means]
      effect sizes: mean_difference = 2.87, cohens_d = 1.12, hedges_g = 1.11,
      rank_biserial = 0.311
```

## Quick start

### 1. Install

```bash
pip install stataudit                 # once 1.0.0 is published to PyPI
# or, straight from GitHub:
pip install "git+https://github.com/bhaskargurram-ai/stataudit@v1.0.0"

pip install "stataudit[web]"          # + Flask for the local web UI
pip install "stataudit[mcp]"          # + the MCP server
```

The core needs only numpy, scipy, pandas, click and jinja2 on Python 3.10 or newer. statsmodels is optional: when it is installed, its OLS fits are traced too. The agent's LLM client uses the standard library, so it adds no dependencies.

### 2. Audit a script or a notebook

```bash
stataudit run analysis.py                                  # terminal report
stataudit run analysis.ipynb                               # notebooks too
stataudit run analysis.py --html report.html --md report.md --json report.json
stataudit run analysis.py --alpha 0.01 -- --input data.csv # args after -- go to your script
```

Exit codes: `0` when every conclusion holds, `1` when any is fragile or does not hold. Use `--fail-on fails` to exit 1 only for ❌, or `--fail-on never` to always exit 0.

**Notebooks.** `stataudit run` executes the code cells in order through the same tracer. IPython magics (`%matplotlib`, `%%time`) and shell lines (`!pip install …`) are stripped, and the report says which ones were. Findings point at `cell N, line M`.

**From Python or inside Jupyter:**

```python
import scipy.stats as stats
import stataudit

with stataudit.recording() as calls:
    stats.ttest_ind(treated, control)
    stats.pearsonr(spend, signups)

stataudit.report(calls)          # renders as Markdown in Jupyter; .to_text(), .to_html(), .to_json()
```

`recording()` also catches names imported before the block (`from scipy.stats import ttest_ind`), because it temporarily rebinds them in the calling module. You can audit data directly as well:

```python
stataudit.audit_ttest(a, b)                       # Student by default, like scipy
stataudit.audit_pearson(x, y)
stataudit.audit_chi2(table)
stataudit.audit_ols(df[["dose", "age"]], df["response"])
stataudit.audit_call("f_oneway", [g1, g2, g3])    # any supported scipy.stats test by name
stataudit.audit_script("analysis.py")
```

**A CSV, no script:**

```bash
stataudit check-data data.csv --test ttest_ind --group-col group --value-col minutes
stataudit check-data data.csv --test chi2_contingency --x-col plan --y-col reason
```

### 3. Check reported results and lint code

```bash
stataudit check-report paper.md       # also .txt, .html, and .pdf when pdftotext is installed
stataudit lint analysis.py notebooks/eda.ipynb
```

### 4. Add it to pull requests

```bash
stataudit init            # add --agent to also set up the experimental fix agent
```

`init` writes the caller workflows described under [GitHub integration](#github-integration) plus a commented `stataudit.toml`, refuses to overwrite existing files unless you pass `--force`, and prints the next steps.

## What each auditor checks

| Test (`scipy.stats`) | Checks on the recorded data | Alternatives computed on the same data |
|---|---|---|
| `ttest_ind` | per-group normality (Shapiro-Wilk for 3 ≤ n ≤ 5000, skew, kurtosis); outliers (robust z by MAD > 3.5, Tukey IQR); variance ratio and Levene (median) | Welch's t (if Student and variances differ), Yuen's 20 % trimmed t (outliers), Mann-Whitney U with rank-biserial r (outliers or non-normal small n); Cohen's d, Hedges' g |
| `ttest_rel` | normality and outliers of the paired differences | Wilcoxon signed-rank |
| `ttest_1samp` | normality and outliers | Wilcoxon signed-rank against `popmean` |
| `pearsonr` | outliers, normality, leverage and Cook's distance (> 4/n), deleting high-leverage points as a group, whether dropping any one point flips significance | Spearman; Pearson without the high-leverage points; Pearson without the most influential point |
| `linregress`, `stataudit.audit_ols`, statsmodels OLS | residual normality, Breusch-Pagan (Koenker), Cook's distance and leverage, leave-one-out | HC3 robust SE and p-value for each slope; refit without high-influence points |
| `f_oneway` | per-group normality and outliers, Levene | Welch's ANOVA, Kruskal-Wallis |
| `chi2_contingency` | Cochran's rule (any expected count < 1, or > 20 % of cells < 5), non-integer "counts", Yates note | Fisher's exact (2×2) or a Monte Carlo permutation p-value with fixed margins and a fixed seed (r×c) |
| `fisher_exact`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `spearmanr`, `shapiro` | notes: ties, zero differences, very small n, smallest attainable p, normality pre-testing | none |
| **Whole script** | how many p-values were computed and how many are significant | Holm-Bonferroni adjusted p for every test, and which results survive |
| **Whole script** | the same arrays put through 3 or more different tests ("test shopping"); groups with n < 10 | none |

Every statistical helper (HC3, Breusch-Pagan, Welch's ANOVA, Cook's distance, Holm, Monte Carlo chi-square) is written with numpy and scipy, and the test suite checks each one against an independent calculation: direct matrix formulas, brute-force leave-one-out refits, exact hypergeometric enumeration and hand-worked examples.

## How verdicts work

| Status | When |
|---|---|
| ✅ **holds** | No assumption problem changes the conclusion. Warnings can still appear, but every recommended alternative agrees. |
| ⚠️ **fragile** | Some recommended alternatives disagree, *or* a sensitivity analysis (dropping one point, dropping the high-leverage points) changes the conclusion, *or* the result does not survive Holm-Bonferroni across the script. |
| ❌ **does not hold** | Every recommended alternative reaches the opposite conclusion at α, or there is a critical problem, such as NaNs that make the statistic NaN or a "contingency table" that isn't made of counts. |

"Recommended" alternatives are computed only when an assumption check fails: Welch's t-test runs only when the variances differ, so a clean analysis gets no alternatives at all. The overall status is the worst status among the tests, raised to fragile when the script tested the same data with three or more different tests.

## Checking reported numbers (`check-report`)

Analyses end up as prose: a report, a README, a paper draft, an assistant's summary. `stataudit check-report` reads APA-style results from Markdown, plain text, HTML, or PDF (when `pdftotext` is installed), recomputes each p-value with scipy from the reported statistic and degrees of freedom, and compares it with the reported one, in the spirit of [statcheck](https://github.com/MicheleNuijten/statcheck).

It recognises `t(df) = x`, `F(df1, df2) = x`, `r(df) = x`, `χ2(df) = x` / `χ²(df, N = n) = x` / `chi2(df) = x` and `z = x`, followed by `p = …`, `p < …`, `p > …` or `ns`, including Unicode minus signs and one-tailed mentions. Reported values are allowed to be rounded: a reported `p = .04` is consistent with any recomputed p that rounds to .04.

| Flag | Meaning |
|---|---|
| **inconsistent** | The recomputed p doesn't match the reported p, even allowing for rounding. |
| **gross inconsistency** | The mismatch flips significance at α (default .05): reported significant, recomputed not, or the other way round. |
| **impossible** | A value that can't occur: p > 1, a negative df, r outside [−1, 1]. |

For example, these sentences give the following results (p-values recomputed with scipy, two-tailed):

| Reported | Recomputed p | Result |
|---|---|---|
| `t(28) = 2.20, p = .04` | 0.036 | consistent (rounds to .04) |
| `χ2(1, N = 90) = 3.95, p < .05` | 0.047 | consistent |
| `z = 2.43, p = .03` | 0.015 | inconsistent; both significant |
| `F(2, 45) = 3.10, p = .04` | 0.055 | **gross**: not significant at .05 |
| `r(48) = .21, p = .01` | 0.143 | **gross**: not significant at .05 |
| `r(30) = 1.20, p < .001` | | **impossible**: r outside [−1, 1] |

Like statcheck, it can only check what is written in the sentence: it doesn't read tables, and it can't tell that a p-value was deliberately adjusted (Bonferroni, Holm) unless the text says so. A flag is a reason to look, not a verdict on the analysis. The report comes in the same formats as `run` (terminal, `--html`, `--md`, `--json`).

## Linting without running (`lint`)

Some analysis code can't be run in CI: the data is private, too large or behind a VPN. `stataudit lint` reads `.py` and `.ipynb` files as a syntax tree, executes nothing and needs no data. It is fast enough for every pull request.

| Rule | What it flags | Suggested fix |
|---|---|---|
| multiple tests, no correction | several hypothesis tests in one file and no `multipletests`, Holm, Bonferroni or FDR step | correct the family (Holm by default), or say which tests are confirmatory |
| Student's t by default | `ttest_ind(...)` without `equal_var=False`, with no variance check on the same data | use Welch's t-test (`equal_var=False`) |
| Pearson without a look | `pearsonr` with no plot or outlier check of the same variables | plot the data; report Spearman as well |
| chi-square on a small table | `chi2_contingency` on a table that looks small or sparse | check expected counts; use Fisher's exact or a Monte Carlo test |
| test shopping | the same arrays passed through 3 or more different tests | pick the test before looking at results; report all of them |
| `p < 0.05` over many tests | significance decided by comparing many p-values to 0.05 in a loop | adjust the p-values first |

Each finding has a rule id, the line, an explanation and the suggested fix. Lint findings are heuristics about code, not facts about data: `stataudit run` is the stronger check whenever you can run the analysis. The rules are in the same family as stat-lint's (see [How stataudit compares](#how-stataudit-compares)), but much simpler: they match patterns in one file and do no dataflow analysis.

## Agent: fix and explain (experimental)

Finding a fragile conclusion is half the job. The opt-in agent tries the other half, **proposing a corrected analysis**, and stataudit then checks the proposal the only way that counts: by running it. The agent is **off by default** and needs your own API key.

<p align="center">
  <img src="docs/assets/fix-loop.svg" alt="The fix agent. The model sees the script as text, the audit findings and a data summary (column names, dtypes and n; raw rows only with --share-data). It proposes a patch with read-only tools and a schema-checked submit; this step holds the API key and runs nothing. A static gate rejects patches that do not parse or that import os, subprocess, socket, shutil, ctypes or importlib, or use eval, exec, __import__ or open for writing. stataudit runs the patched script through the tracer in a subprocess with a timeout and no secrets, and accepts it only if the same questions are answered, the tests hold or improve and no new critical findings appear. Otherwise findings are fed back for up to two rounds; if nothing verifies, the plain audit report stands." width="880">
</p>

```bash
export STATAUDIT_LLM_PROVIDER=anthropic       # or openai (any OpenAI-compatible endpoint)
export STATAUDIT_LLM_MODEL=<model id>         # required; there is no default model
export STATAUDIT_LLM_API_KEY=...

stataudit fix analysis.py                     # prints a diff and before/after verdicts
stataudit fix analysis.ipynb --rounds 2 --apply
stataudit explain report.json                 # plain-language summary of a report
```

**`stataudit fix SCRIPT.py|NOTEBOOK.ipynb`**

- **What the model sees.** The script, the audit findings as JSON, and a summary of the data: column names, dtypes and n. Not the raw rows, unless you pass `--share-data`. All of it is delimited in the prompt as untrusted data.
- **What it can do.** Call read-only tools (read the script, show the findings, show the data summary) and submit one schema-validated patch. It can't write files, run code or reach the network.
- **What stataudit checks.** The patch must parse and pass a static denylist: no imports of `os`, `subprocess`, `socket`, `shutil`, `ctypes` or `importlib`, no `eval`, `exec` or `__import__`, no `open(...)` for writing. Then stataudit runs the patched script **through the tracer, in a subprocess, with a timeout and an environment scrubbed of secret-looking variables**, and audits it again. The fix is accepted only if every question the original answered is still answered (same call sites, or an explicit replacement such as Welch for Student), every patched test holds or improves, and no new critical findings appear.
- **What you get.** A unified diff plus a before/after verdict table. Files are changed only with `--apply`. If a round fails verification, the new findings go back to the model, for up to `--rounds` rounds (default 2). If nothing verifies, you get the normal audit report and no fix.

A verified fix is evidence that the patched analysis passes stataudit's checks. It is not proof that the patch asks the question you meant to ask: switching from a t-test to Mann-Whitney changes the hypothesis from means to distributions, and the before/after table says so. Read the diff.

**`stataudit explain`** writes a summary of the findings for a reader who isn't a statistician. It is grounded: every number in the model's text must appear in the findings JSON. If any number doesn't, the text is rejected and the deterministic summary is used instead.

**Honest limits.** We have not benchmarked the fix agent, so we don't publish a success rate. Expect it to come back empty-handed on analyses whose problem is the study design rather than the test. Token use is reported with every run. Some models need provider-specific settings: OpenAI's `gpt-6-luna` only calls tools through Chat Completions with `STATAUDIT_LLM_REASONING_EFFORT=none`, and GPT-6 models reject `temperature` unless reasoning is off. Details are in [docs/AGENT.md](docs/AGENT.md).

## MCP server and web UI

**MCP server.** `stataudit-mcp` (or `stataudit mcp`) is a Model Context Protocol server over stdio, so an assistant that writes an analysis can audit it before reporting a result:

| Tool | What it does |
|---|---|
| `audit_script(path, alpha)` | Runs a local `.py` file under the tracer (in a subprocess, 120-second timeout, no shell) and returns the Markdown report. |
| `audit_test(test_name, data_json, alpha)` | Audits arrays passed as JSON, for example `{"a": [...], "b": [...], "options": {"equal_var": true}}`. |
| `check_report(path, alpha)` | Runs `check-report` on a text, Markdown or HTML file. |
| `lint_file(path)` | Runs `lint` on a `.py` or `.ipynb` file. |

```json
{
  "mcpServers": {
    "stataudit": { "command": "stataudit-mcp", "args": [] }
  }
}
```

The server treats its inputs as data: it accepts only known test names and numeric arrays (up to 200,000 values), allows only a short list of scalar options, and runs only the script you name.

**Web UI.** `stataudit ui` starts a local Flask app at `http://127.0.0.1:8765`: upload a CSV, pick a test and its columns, and read the HTML report. It binds to 127.0.0.1 by default and stores nothing.

## GitHub integration

stataudit runs on pull requests as separate workflows, so that PR code and privileged tokens never share a job ([why](#security-model)). Each caller is a few lines pointing at a reusable workflow in this repository; `stataudit init` writes them for you. Pin `@v1.0.0`, or a full commit SHA if you want to be strict.

**`.github/workflows/stataudit.yml`** checks the PR. It is unprivileged, and it is where PR code runs.

```yaml
name: stataudit

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read

concurrency:
  group: stataudit-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  check:
    uses: bhaskargurram-ai/stataudit/.github/workflows/check.yml@v1.0.0
    with:
      stataudit-ref: v1.0.0
      python-version: "3.12"
      # Installs what your analyses import; this runs PR code (unprivileged job).
      install-command: 'pip install -r requirements.txt || true'
      fail-on: never   # or "fragile" / "fails" to fail the check
```

The check job:

- runs `stataudit lint` on the `.py` and `.ipynb` files the PR changes;
- runs `stataudit check-report` on changed `.md`, `.txt` and `.html` files that match `reports` in `stataudit.toml`;
- runs `stataudit run` **only** on files matching `run` in `stataudit.toml`, because that executes them. Nothing is executed unless you list it;
- writes the card to the job summary (so it shows up even on fork PRs) and uploads `result.json`, `card.md` and the PR number as an artifact.

**`.github/workflows/stataudit-comment.yml`** posts the card. It is privileged and never runs PR code.

```yaml
name: stataudit comment

on:
  workflow_run:
    workflows: ["stataudit"]   # must match the `name:` of stataudit.yml
    types: [completed]

permissions: {}

jobs:
  comment:
    if: github.event.workflow_run.event == 'pull_request' && github.event.workflow_run.conclusion != 'cancelled'
    permissions:
      contents: read
      actions: read          # download the artifact of the triggering run
      pull-requests: write   # upsert the comment (and the optional label)
    uses: bhaskargurram-ai/stataudit/.github/workflows/comment.yml@v1.0.0
    with:
      stataudit-ref: v1.0.0
```

It validates `result.json` strictly, re-renders the card from the validated data (the uploaded `card.md` is ignored), and creates or updates one sticky comment, found by a hidden marker. The card ends with an invisible `<!-- stataudit:data {...} -->` block (escaped, size-capped, built only from validated fields) so bots can read the verdict without parsing Markdown.

<details>
<summary><b><code>.github/workflows/stataudit-agent.yml</code> (optional, experimental): propose verified fixes</b></summary>

```yaml
name: stataudit agent

on:
  workflow_run:
    workflows: ["stataudit"]   # must match the `name:` of stataudit.yml
    types: [completed]

permissions: {}

jobs:
  agent:
    if: github.event.workflow_run.event == 'pull_request' && github.event.workflow_run.conclusion != 'cancelled'
    permissions:
      contents: read
      actions: read
      pull-requests: read
    uses: bhaskargurram-ai/stataudit/.github/workflows/agent.yml@v1.0.0
    with:
      stataudit-ref: v1.0.0
      llm-provider: ${{ vars.STATAUDIT_LLM_PROVIDER }}
      llm-model: ${{ vars.STATAUDIT_LLM_MODEL }}
      llm-base-url: ${{ vars.STATAUDIT_LLM_BASE_URL }}
      llm-reasoning-effort: ${{ vars.STATAUDIT_LLM_REASONING_EFFORT }}
      python-version: "3.12"
      install-command: 'pip install -r requirements.txt || true'
      run-for: trusted   # or "everyone" (spends API credits on any fork PR)
    secrets:
      llm-api-key: ${{ secrets.STATAUDIT_LLM_API_KEY }}
```

Add a repository **secret** `STATAUDIT_LLM_API_KEY` and repository **variables** `STATAUDIT_LLM_PROVIDER` and `STATAUDIT_LLM_MODEL` (optional: `STATAUDIT_LLM_BASE_URL`, `STATAUDIT_LLM_REASONING_EFFORT`). The workflow has two jobs: a **propose** job that holds the key, reads the PR's files as data and never installs or runs them, and a **verify** job with no secrets that runs the patched script through the tracer. A fix that verifies is added to the card as a collapsible "suggested fix (verified)" diff.

</details>

[`examples/workflows/`](examples/workflows) contains exactly what `stataudit init` generates, and a test keeps the two in sync.

## Security model

Running PR code safely is the core problem. stataudit splits the work so that capabilities never meet in one job:

<p align="center">
  <img src="docs/assets/security-model.svg" alt="Four jobs and their capabilities. The check job runs on pull_request: it runs PR code with a read-only token and no secrets, and hands on result.json and card.md. The agent propose job holds the LLM API key, reads PR files as data and never runs PR code. The agent verify job runs the patched script with no secrets on a separate VM. The comment job holds a write token, never checks out PR code, validates result.json and re-renders the card. No job both runs PR code and holds a secret or a write token." width="880">
</p>

- **Untrusted code runs only in jobs without secrets.** The check job runs on `pull_request` with `contents: read`, `persist-credentials: false`, timeouts and a scrubbed environment. `stataudit run` only touches files you opted in with `run = [...]`; `lint` and `check-report` never execute anything.
- **The LLM key never meets PR code.** The propose job reads files as data and never installs or runs them. The verify job, on a separate VM, has no secrets. Patches also pass a static denylist before they run, but the denylist is a speed bump, not the boundary: the boundary is that the job that runs them holds nothing worth stealing.
- **The privileged comment job never checks out or executes PR code.** It treats the artifact as hostile: it validates the PR number and the `result.json` schema, re-renders the card itself and escapes all PR-derived text.
- **No `${{ github.event.* }}` in `run:` steps, no `pull_request_target`,** and minimal `permissions:` everywhere. A test in this repository checks these invariants on every workflow.
- **Prompt injection.** The deterministic path has nothing to inject into. Code, data, reports and docstrings reach an LLM only as delimited, untrusted data; the model has no write, exec or network tools; its output is schema-validated; and nothing it writes counts until stataudit has run and re-audited it.

stataudit raises the bar; it is **not** a sandbox. A malicious PR can still tamper with its own results inside the unprivileged job, and `stataudit run` on your laptop runs the script with your permissions, like `python analysis.py` would. Read [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md) for the full threat model.

## How stataudit compares

Different tools answer different questions. This is our best reading of each as of October 2026; corrections are welcome.

| | [statcheck](https://github.com/MicheleNuijten/statcheck)<br><sub>Nuijten & Epskamp; R and web app</sub> | stat-lint<br><sub>[*The Fault in our Stats*, ASE 2025](https://conf.researchr.org/details/ase-2025/ase-2025-papers/226/The-Fault-in-our-Stats)</sub> | [pingouin](https://pingouin-stats.org) / [statsmodels diagnostics](https://www.statsmodels.org/stable/diagnostic.html) | LLM data-analysis agents<br><sub>e.g. ChatGPT data analysis, Julius</sub> | **stataudit**<br><sub>1.0.0</sub> |
|---|---|---|---|---|---|
| Main question | Do the reported numbers agree with each other? | Does the code meet statistical best practice? | Does this data meet this assumption? | What does my data say? | **Does the conclusion survive on the data?** |
| Input | Article text (PDF, HTML, Word) | Python notebooks, statically | Arrays you pass to each function | Data and a prompt | Your script or notebook and its data; also text and code statically |
| Runs your analysis on your data | No | No | You call each function | Writes and runs its own code | **Yes, traces your own run** |
| Checks assumptions per test | No | Checks whether the code checks them | Yes, one call at a time | If asked; varies by prompt | **Yes, automatically, on the data each test received** |
| Computes alternatives, judges the conclusion | No | No | Provides the tests (Welch, robust SEs, …) | Sometimes | **Yes: holds / fragile / does not hold** |
| Recomputes reported p-values from text | **Yes**: t, F, r, χ², z, Q | No | No | No | Yes: t, F, r, χ², z (no Q) |
| Static analysis without data | No | **Yes, dataflow over its own IR (SIR)** | No | No | Yes, simpler single-file AST rules |
| Multiple comparisons | No | Flags missing correction | Provides corrections | Varies | Holm across every traced p-value |
| Uses an LLM | No | No | No | Yes, and it decides | Optional, and it never decides |
| CI / pull requests | Not built in | Research prototype | n/a | Hosted chat | GitHub workflows and a PR comment |
| Language | R (statcheck); text in any field | Python | Python | Mostly Python | Python: `scipy.stats`, statsmodels OLS |

These tools are complementary. statcheck is the established tool for checking published articles and has been validated at scale (a validity study reported accuracy of roughly 96–99.9 % depending on settings); stataudit's `check-report` borrows the idea for analysis write-ups and does less (no Q tests, no tables). stat-lint's static analysis is more thorough than `stataudit lint`, which matches patterns in one file. pingouin and statsmodels give you every diagnostic stataudit uses and more, but you have to know which to call and what to conclude. LLM agents run analyses quickly, but nothing independent checks what they concluded. stataudit's own gaps: it sees only `scipy.stats` and statsmodels OLS calls (no R, no hand-computed tests, no mixed models), it cannot know your study design, and its agent is unbenchmarked.

## Configuration

`stataudit.toml` at the repository root configures CI and the defaults for the CLI. `stataudit init` writes a commented starting point.

<details open>
<summary><b>All keys</b></summary>

```toml
alpha = 0.05                      # significance level used to judge conclusions
fail-on = "fragile"               # fragile | fails | never

# Scripts and notebooks that CI may *execute* with `stataudit run`.
# Opt-in on purpose: nothing runs unless it matches here.
run = ["analysis/**/*.py", "notebooks/**/*.ipynb"]

# Text checked with `stataudit check-report` (Markdown, plain text, HTML).
reports = ["reports/**/*.md"]
```

`lint` runs on every changed `.py` and `.ipynb` file and needs no configuration.

</details>

**Environment variables (agent only)**

| Variable | Purpose |
|---|---|
| `STATAUDIT_LLM_PROVIDER` | `anthropic` (Messages API) or `openai` (any OpenAI-compatible Chat Completions endpoint) |
| `STATAUDIT_LLM_API_KEY` | API key. Never read from config files, and stripped from every subprocess. |
| `STATAUDIT_LLM_MODEL` | Model id. **Required**; there is no default. |
| `STATAUDIT_LLM_BASE_URL` | Optional custom endpoint |
| `STATAUDIT_LLM_REASONING_EFFORT` | Optional reasoning effort, passed through to the provider (for example `none` for `gpt-6-luna` tool calls) |

## CLI reference

```text
stataudit run          SCRIPT.py|NOTEBOOK.ipynb [-- ARGS...] [report options] [--timeout 300] [--show-output]
stataudit check-report FILE [report options]
stataudit lint         FILE... [--json FILE]
stataudit check-data   DATA.csv --test NAME [--value-col C] [--group-col C] [--x-col C] [--y-col C]
                       [--popmean M] [--welch] [--sep S] [report options]
stataudit fix          SCRIPT.py|NOTEBOOK.ipynb [--rounds 2] [--share-data] [--apply]
stataudit explain      REPORT.json
stataudit init         [--agent] [--force]
stataudit ui           [--host 127.0.0.1] [--port 8765]
stataudit mcp          (same as the stataudit-mcp command)
```

| Command | What it does |
|---|---|
| `run` | Executes a script or notebook under the tracer and audits every test it performs. |
| `check-report` | Recomputes p-values for APA-style results in text and flags inconsistencies. Executes nothing. |
| `lint` | Static rules over `.py` / `.ipynb` files. Executes nothing. |
| `check-data` | Runs one test on columns of a CSV and audits it, without a script. |
| `fix` | Experimental agent: proposes a corrected analysis and verifies it by running it. Writes files only with `--apply`. |
| `explain` | Experimental agent: plain-language summary, with every number checked against the findings. |
| `init` | Writes the caller workflows and a commented `stataudit.toml`. |
| `ui` | Local web UI. |
| `mcp` | MCP server over stdio. |

<details>
<summary><b>Report options</b> (<code>run</code>, <code>check-data</code>, <code>check-report</code>)</summary>

| Option | Purpose |
|---|---|
| `--alpha A` | significance level used to judge conclusions (default 0.05) |
| `--html FILE` | write a self-contained HTML report (light and dark) |
| `--md FILE` | write a Markdown report |
| `--json FILE` | write a JSON report |
| `--fail-on fragile\|fails\|never` | exit 1 when the overall status is at least this bad (default `fragile`) |
| `-q`, `--quiet` | don't print the terminal report |
| `--color/--no-color` | force or disable ANSI colours |
| `-v`, `--verbose` | show details for clean results too |

You can also run `python -m stataudit …`.

</details>

## FAQ

<details>
<summary><b>Does ✅ mean my analysis is correct?</b></summary>

No. It means stataudit's checks found no problem that changes the conclusion. It can't know whether your observations are independent, whether you chose the hypothesis before seeing the data, or whether a variable is a confounder. It is not a substitute for a statistician.

</details>

<details>
<summary><b>Does ❌ mean the effect isn't real?</b></summary>

No. It means the data, as analysed, doesn't support the conclusion once its assumption problems are taken into account. In example 01 the right next step is to find out why three users took twice as long, not to rerun tests until one is significant.

</details>

<details>
<summary><b>Do I need an LLM?</b></summary>

No. Tracing, auditing, alternatives, verdicts, `check-report` and `lint` are deterministic. An LLM is only used by `fix` and `explain`, and neither decides a verdict: a proposed fix counts only after stataudit has run and re-audited it, and an explanation is rejected if it contains a number that isn't in the findings.

</details>

<details>
<summary><b>Does my data leave my machine?</b></summary>

Not unless you use the agent. `run`, `check-report`, `lint`, the web UI and the MCP server work locally. `fix` sends the script, the findings and a data summary (column names, dtypes, n) to the LLM provider you configure; raw rows only with `--share-data`.

</details>

<details>
<summary><b>Why does stataudit run my script instead of reading it?</b></summary>

Because assumptions are about data, not code. `ttest_ind(a, b)` is fine or not depending on what `a` and `b` contain after your filtering. When you can't run the analysis, `stataudit lint` gives the weaker, static answer.

</details>

<details>
<summary><b>What about R, Stata or SPSS?</b></summary>

Only Python is traced today. `check-report` works on any text, whatever produced the numbers. R support is on the roadmap.

</details>

## Limitations

- stataudit audits **assumption violations and robustness** of the tests it sees. Independence, study design, the family of tests and pre-registration are questions only you can answer.
- Only calls through `scipy.stats` (plus statsmodels OLS, when installed) are traced. Tests computed by hand, in R, or through other libraries are invisible to it.
- Assumption checks have little power at small n and too much at very large n. stataudit reports evidence (skewness, variance ratios, Cook's distance) next to p-values for that reason.
- Holm-Bonferroni treats every traced p-value as one family. If your design says otherwise, read the multiplicity section with that in mind.
- Multi-dimensional calls (`axis=1` on 2-D arrays) are recorded but not audited.
- `check-report` reads sentences, not tables, and can't recognise deliberately adjusted p-values unless the text says so.
- `lint` is pattern-based and single-file; it misses problems that need dataflow across files and can flag code that is fine.
- The agent is experimental and unbenchmarked. A verified fix passes stataudit's checks; it can still change the question being asked.

## Roadmap

- More tests: logistic and Poisson regression, mixed models, survival analysis (log-rank, Cox)
- R support: tracing `t.test`, `cor.test` and `lm` through an R bootstrap
- `check-report` for tables and Q statistics
- A Jupyter extension that shows verdicts under the cell that ran the test
- Post-hoc procedures (Games-Howell) and FDR control as an alternative to Holm
- Bootstrap confidence intervals for effect sizes
- A published benchmark of the fix agent on analyses with known problems

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). In short: every new check needs a test showing that it flags a bad case and passes a clean one, and every new statistical helper needs a test against an independent calculation.

```bash
pip install -e ".[dev]"
python -m pytest            # or, without pytest: python -m unittest discover -s tests
```

[docs/TESTING.md](docs/TESTING.md) describes every test level, including the agent's scripted fake-LLM tests and the workflow invariant tests. Please report security issues privately through the repository's security advisory form rather than in a public issue.

## License

[MIT](LICENSE) © 2026 Bhaskar Gurram
