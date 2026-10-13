# stataudit

**A second-opinion statistician for AI-generated analyses.** stataudit runs your analysis script, records every statistical test it performs, checks each test's assumptions on the data it actually used, reruns robust alternatives, and tells you whether the conclusion still holds.

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

That is real output from [`examples/01_ttest_outliers.py`](examples/01_ttest_outliers.py), a short analysis of the kind an assistant writes when asked "is the new onboarding flow faster?".

## Why this matters

More and more statistics gets done by asking an AI assistant. The code runs and a p-value comes out, but the conclusion can still be wrong, because the test's assumptions don't fit the data:

- a Pearson correlation produced by two extreme points
- a Student t-test on small, skewed samples with very different variances
- a chi-square test on a table whose expected counts are mostly below 5
- an OLS slope whose standard error ignores heteroskedastic residuals
- twenty metrics tested, three "significant", none corrected

Benchmarks such as StatQA and P-Bench measure how often models choose an inappropriate test. stataudit works on your own analysis instead. It doesn't judge the code. It looks at the data each test actually received and asks whether the result survives the analyses a careful statistician would also have run.

## Install

```bash
pip install stataudit            # core: numpy, scipy, pandas, click, jinja2
pip install "stataudit[web]"     # + Flask for the local web UI
pip install "stataudit[mcp]"     # + the MCP server
```

From a checkout: `pip install -e ".[dev]"`. Requires Python 3.10 or newer. stataudit does **not** depend on statsmodels. If statsmodels is installed, its OLS fits are traced as well.

## Quick start

### CLI

```bash
stataudit run analysis.py                                  # terminal report
stataudit run analysis.py --html report.html --md report.md --json report.json
stataudit run analysis.py --alpha 0.01 -- --input data.csv # args after -- go to your script
stataudit check-data data.csv --test ttest_ind --group-col g --value-col y
stataudit check-data data.csv --test chi2_contingency --x-col plan --y-col reason
```

Exit codes: `0` when every conclusion holds and `1` when any is fragile or does not hold. Use `--fail-on fails` to exit 1 only when a conclusion does not hold, or `--fail-on never` to always exit 0. This lets you put `stataudit run` in CI next to a notebook export or a report script.

### Notebooks and Python

```python
import scipy.stats as stats
import stataudit

with stataudit.recording() as calls:
    stats.ttest_ind(treated, control)
    stats.pearsonr(spend, signups)

stataudit.report(calls)          # renders as Markdown in Jupyter; .to_text(), .to_html(), .to_json()
```

You can also audit data directly:

```python
stataudit.audit_ttest(a, b)                       # Student by default, like scipy
stataudit.audit_pearson(x, y)
stataudit.audit_chi2(table)
stataudit.audit_ols(df[["dose", "age"]], df["response"])
stataudit.audit_call("f_oneway", [g1, g2, g3])    # any supported scipy.stats test by name
stataudit.audit_script("analysis.py")
```

`recording()` also catches names you imported *before* the block (`from scipy.stats import ttest_ind`), because it temporarily rebinds them in the calling module.

### MCP server

`stataudit-mcp` is a Model Context Protocol server running over stdio. It offers two tools:

- `audit_script(path, alpha)` runs a local `.py` file under the tracer (in a subprocess, with a 120-second timeout and no shell) and returns the Markdown report.
- `audit_test(test_name, data_json, alpha)` audits arrays passed as JSON, for example `{"a": [...], "b": [...], "options": {"equal_var": true}}`.

Generic client configuration:

```json
{
  "mcpServers": {
    "stataudit": {
      "command": "stataudit-mcp",
      "args": []
    }
  }
}
```

The server treats its inputs as data. It accepts only known test names and numeric arrays (up to 200,000 values), allows only a short list of scalar options, and runs only the script you name.

### Web UI

```bash
pip install "stataudit[web]"
stataudit ui --port 8765         # http://127.0.0.1:8765
```

Upload a CSV, pick a test and its columns, and read the HTML report. The UI binds to 127.0.0.1 by default and stores nothing.

## What it checks

| Test (scipy.stats) | Checks | Alternatives computed on the same data |
|---|---|---|
| `ttest_ind` | per-group normality (Shapiro-Wilk for 3 ≤ n ≤ 5000, skew, kurtosis); outliers (robust z by MAD > 3.5, Tukey IQR); variance ratio and Levene (median) | Welch's t (if Student and variances differ), Yuen's 20% trimmed t (outliers), Mann-Whitney U with rank-biserial r (outliers or non-normal small n); Cohen's d, Hedges' g |
| `ttest_rel` | normality and outliers of the paired differences | Wilcoxon signed-rank |
| `ttest_1samp` | normality and outliers | Wilcoxon signed-rank against `popmean` |
| `pearsonr` | outliers, normality, leverage and Cook's distance (> 4/n), group deletion of high-leverage points, leave-one-out flip of significance | Spearman; Pearson without the high-leverage points; Pearson without the most influential single point |
| `linregress`, `audit_ols`, statsmodels OLS | residual normality, Breusch-Pagan (Koenker), Cook's distance and leverage, leave-one-out | HC3 robust SE and p-value for each slope; refit without high-influence points |
| `f_oneway` | per-group normality and outliers, Levene | Welch's ANOVA, Kruskal-Wallis |
| `chi2_contingency` | Cochran's rule (any expected count < 1, or > 20% of cells < 5), non-integer "counts", Yates note | Fisher's exact (2×2) or a Monte Carlo permutation p-value with fixed margins and a fixed seed (r×c) |
| `fisher_exact`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `spearmanr`, `shapiro` | info-level notes: ties, zero differences, very small n, smallest attainable p, normality pre-testing | — |
| **Whole script** | number of p-values reported and how many are significant | Holm-Bonferroni adjusted p for every test, and which results survive |
| **Whole script** | the same arrays tested with ≥ 3 different tests ("test shopping"); groups with n < 10 | — |

Every statistical helper (HC3, Breusch-Pagan, Welch's ANOVA, Cook's distance, Holm, Monte Carlo chi-square) is written with numpy and scipy. The test suite checks each one against an independent calculation: direct matrix formulas, brute-force leave-one-out refits, exact hypergeometric enumeration, and hand-worked examples.

## How verdicts work

Each traced p-value gets one of three statuses:

- ✅ **holds**: no assumption problem changes the conclusion. Warnings can still appear, but every recommended alternative agrees.
- ⚠️ **fragile**: some recommended alternatives disagree, *or* a sensitivity analysis (dropping one point, dropping the high-leverage points) changes the conclusion, *or* the result does not survive Holm-Bonferroni across the script.
- ❌ **does not hold**: every recommended alternative reaches the opposite conclusion at α, or there is a critical problem (for example NaNs that make the statistic NaN, or a "contingency table" that isn't made of counts).

"Recommended" alternatives are computed only when an assumption check fails. For example, Welch's t-test is run only when the variances differ. A clean analysis therefore gets no alternatives at all. The overall status is the worst status among the tests, raised to "fragile" when the script tested the same data with three or more different tests.

Here are the bundled examples, each run for real:

| Example | Mistake | Verdict |
|---|---|---|
| `01_ttest_outliers.py` | Student t-test, 3 outliers, variances 16× apart | ❌ Welch p = 0.066, Mann-Whitney p = 0.073 |
| `02_pearson_leverage.py` | correlation created by 2 extreme regions | ❌ r = 0.81 → 0.06 without them; Spearman p = 0.15 |
| `03_chisq_sparse.py` | chi-square on a 3×4 table, 67% of expected counts < 5 | ❌ p = 0.03 → Monte Carlo p = 0.08 |
| `04_many_tests.py` | 15 uncorrected Welch tests | ⚠️ 3 significant, 2 survive Holm |
| `05_good_analysis.py` | Welch t-test on clean data | ✅ holds |

## Limitations

- stataudit audits **assumption violations and robustness** of the tests it sees. It cannot know your study design. Whether observations are independent, whether the tests form one family, whether a variable is a confounder, and whether the hypothesis was fixed before you saw the data are all questions only you can answer.
- It is **not a substitute for a statistician**. A ✅ means "no problems found with these checks", not "this analysis is correct".
- Only calls through `scipy.stats` (plus statsmodels OLS, when installed) are traced. Tests computed by hand, in R, or through other libraries are invisible to it.
- Assumption checks themselves have little power at small n and too much at very large n. stataudit reports evidence (skewness, variance ratios, Cook's distance) next to p-values for that reason.
- Holm-Bonferroni treats every traced p-value as one family. If your design says otherwise, read the multiplicity section with that in mind.
- Multi-dimensional calls (`axis=1` on 2-D arrays) are recorded but not audited.

## Roadmap

- More tests: logistic and Poisson regression, mixed models, survival analysis (log-rank, Cox).
- R support (tracing `t.test`, `cor.test`, `lm` through an R bootstrap).
- A Jupyter extension that shows verdicts inline under the cell that ran the test.
- Post-hoc procedures (Games-Howell) and FDR control as an alternative to Holm.
- Bootstrap confidence intervals for effect sizes.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). In short: every new check needs a test showing that it flags a bad case and passes a clean one, and every new statistical helper needs a test against an independent calculation.

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -v    # or: pytest
```

## License

MIT © 2026 Bhaskar Gurram
