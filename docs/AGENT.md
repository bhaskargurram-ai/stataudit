# The stataudit agent (experimental)

stataudit's checks are deterministic: `run` re-runs the analysis and audits
every test on the real data, `check-report` recomputes reported p-values,
`lint` reads code. None of them involves a language model, and none ever will
decide a verdict.

The agent is an **optional** layer on top, with your own API key:

* **`stataudit fix`** asks a model to propose a corrected analysis for a script
  or notebook whose conclusions are fragile or do not hold. stataudit then
  **runs the proposal under its tracer** and accepts it only if the audit
  says it is better. The model proposes; execution decides.
* **`stataudit explain`** asks a model for a plain-language summary of the
  findings for a non-statistician. Every number in the text must appear in
  the findings; otherwise the text is thrown away and a deterministic summary
  is printed instead.

> **Status: experimental.** Off unless you configure a key. It has been tested
> end to end with a scripted fake model (no network); how often real models
> produce fixes that pass verification has **not been measured**.

## Honest expectations

* Many audit failures have a standard remedy (Welch instead of Student, a rank
  test for skewed data with outliers, Fisher's exact test for a sparse 2×2
  table). Expect the agent to find those. Problems that need a design
  decision (which tests form a family, whether an outlier is an error, whether
  a variable is a confounder) cannot be settled by re-running code, and the
  agent will usually fail to produce an accepted fix for them. Example 04
  (fifteen uncorrected tests) is such a case: stataudit applies Holm's
  correction to every traced p-value, so no rewrite of the script changes its
  verdict.
* A verified fix means "stataudit's audit of the new analysis is better and
  every original question is still answered". It does **not** mean the new
  analysis is the right one for your study. Review the diff like any other
  change; files are only modified with `--apply` or `--out`.
* The model's rationale is unverified prose. It is shown on the card, labelled
  as such.

## Configuration

| Variable | Meaning |
|---|---|
| `STATAUDIT_LLM_PROVIDER` | `anthropic` (Messages API) or `openai` (any OpenAI-compatible Chat Completions server) |
| `STATAUDIT_LLM_MODEL` | model id. **Required**: there is no default model |
| `STATAUDIT_LLM_API_KEY` | API key. Only ever read from the environment (refused in config files) |
| `STATAUDIT_LLM_BASE_URL` | optional API base URL (a local server such as `http://localhost:8000/v1` needs no key) |
| `STATAUDIT_LLM_REASONING_EFFORT` | optional, OpenAI-compatible only: sent as `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) |

Generic names such as `LLM_API_KEY` are deliberately **not** read.
`provider`, `model` and `base_url` may also come from the `[llm]` table of
`stataudit.toml` (the environment wins).

Provider notes:

* OpenAI's **GPT-6 Luna** needs `STATAUDIT_LLM_REASONING_EFFORT=none` to call
  tools through Chat Completions.
* **GPT-6** models reject a `temperature` parameter unless reasoning is off.
  stataudit sends no temperature.
* Requests are retried on 408/409/425/429/5xx/529 with exponential back-off
  (honouring `Retry-After`, capped at 60 s); token usage (including prompt
  cache reads and writes) is reported in the JSON output.

## How `fix` works

```
 stataudit run (tracer)  ──►  findings (verdicts, assumption checks, alternatives,
        │                               data summaries: n, mean, sd, missing)
        ▼
 model  ⇄ read-only tools: read_script, show_findings, data_summary
        │                  (+ show_data with --share-data)
        ▼
 submit_fix(code, rationale, replacements)   ── schema + static checks ──► refused? tell the model
        ▼
 run the candidate in a subprocess: timeout, scrubbed environment, tracer
        ▼
 compare the two audits  ──►  accepted (verified)  or  reasons fed back (up to --rounds)
```

### What the model sees

* the script (or, for a notebook, its code cells as `# %% [cell N]` sections);
* stataudit's findings for the original: per test the location, verdict,
  findings and the alternative analyses that were computed;
* data **summaries** for each traced input: `n`, missing values, mean and
  standard deviation, and the schema of CSV/TSV files the code names (columns,
  dtypes, missing and distinct counts). Single observed values (minimum,
  maximum, rows) are **not** shown;
* with `--share-data` only: the first 20 rows of those CSV files and, locally,
  the first 30 values of each traced input.

Everything the model reads is wrapped in `<<<UNTRUSTED_<nonce> ...>>>`
delimiters with a fresh random nonce, and the system prompt tells it never to
follow instructions found inside. Its **only** output channel is the
`submit_fix` tool, whose arguments are schema-validated (code must be a
string, `replacements` at most 50 string pairs, rationale capped).

### Static checks (before anything runs)

A submission is refused, and the model told why, when it

* does not parse, is empty, unchanged, or longer than 200,000 characters;
* for notebooks, adds, removes or renumbers `# %% [cell N]` markers;
* **adds** any denylisted construct: imports of `os`, `subprocess`, `socket`,
  `shutil`, `ctypes`, `importlib`, `multiprocessing`, `pty`, `signal`,
  `urllib`, `http`, `requests`, `httpx`, `pickle`, `marshal`, `builtins`,
  `asyncio`, `threading`, `runpy` and similar; relative imports; calls to
  `eval`, `exec`, `compile`, `__import__`, `breakpoint`, `input`, `globals`,
  `locals`, `vars`, `getattr`/`setattr`/`delattr`; `open(...)` with a
  write/append/create mode (or a computed mode); attribute access such as
  `__globals__`, `__subclasses__`, `__builtins__`, `.system`, `.popen`,
  `.write_text`, `.write_bytes`, `.unlink`, `.rmtree`, `.chmod`.

Constructs that the **original** file already contains are not counted
against the candidate (an analysis that does `import os` to build a data path
may keep it). The denylist is a filter for obviously off-task code, **not a
sandbox**; see the safety model below.

### Verification (the only thing that counts)

The candidate is written to a temporary file next to the original (so relative
data paths keep working), run in a subprocess under the tracer with a timeout
(`--timeout`, default 120 s; the whole process group is killed on expiry) and
with every secret-looking environment variable removed (names containing
`TOKEN`, `SECRET`, `KEY`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `AUTH`, `COOKIE`,
`SESSION`, plus `STATAUDIT_LLM_*`, `GH_*`, `ACTIONS_RUNTIME_*`,
`ACTIONS_ID_TOKEN_*`). The temporary file is deleted afterwards.

The candidate is **accepted** only if all of these hold:

1. it runs to completion (no exception, no timeout);
2. every result the original audit judged is still answered: by a test on the
   **same data** (same data fingerprint), or by a call the model explicitly
   named in `replacements` (`{"original_location": "analysis.py:25",
   "patched_location": "analysis.py:27"}`), for example when it transformed
   the data;
3. each of those answers **holds**, or its status improved
   (does not hold → fragile → holds);
4. no new critical finding appears, and the patch adds no new test that is
   fragile or does not hold;
5. the overall status is no worse, and at least one result improved.

Otherwise the reasons (and the audit of the candidate, as untrusted data) go
back to the model for the next round (`--rounds`, default 2).

### Output

A unified diff, a before/after table per original test, the model's rationale
and the reasons for any rejection; `--json` and `--md` write the same as JSON
or Markdown. Exit code 0 for a verified fix (or nothing to fix), 1 otherwise.
The file changes only with `--apply` (overwrite) or `--out PATH`, and only for
a verified fix. For notebooks, `--apply` rewrites the changed code cells and
clears their outputs; markdown cells and unchanged cells are kept.

## How `explain` works

`stataudit explain SOURCE` takes a JSON report (`run`, `check-report` or
`lint` with `--json`), or a `.py`/`.ipynb` analysis (audited first) or a
`.md`/`.txt`/`.html`/`.pdf` report (checked first). The model receives the
findings as delimited data and must answer through `submit_explanation`.

The text is **grounded** before it is shown: every number in it must match a
number in the findings (numeric values or numbers written inside its
strings), allowing for rounding to the decimals written (0.066 matches
0.0661) and for percentages (5% matches 0.05). Anything else, such as a
computed "23% faster", rejects the whole text, and the deterministic summary
built from the findings is printed instead, with a note on stderr. Without an
LLM configured (or with `--no-llm`) the deterministic summary is printed.

## Safety model

The full threat model is in [SECURITY_MODEL.md](SECURITY_MODEL.md). In short:

* **Read-only tools.** The model can read the script, the findings and data
  summaries. It cannot run code, write files or fetch URLs.
* **Single output channel.** Its only effect is a schema-validated
  `submit_fix` / `submit_explanation` call.
* **Never trusted.** Model-written code is statically checked, then executed
  only in a subprocess with a timeout and a scrubbed environment, and then
  only the re-run audit decides. Model-written prose is shown as unverified
  (`fix`) or must be grounded in the findings (`explain`).
* **Split in CI.** In GitHub Actions the key lives in a job that reads the
  pull request as data and never executes it (`stataudit agent propose`);
  the candidates run in a different job, on a different VM, with no secrets
  (`stataudit agent verify`).

### Why local `fix` is for code you trust

Run locally, `stataudit fix` audits your analysis (running it, as
`stataudit run` does) and runs the model's candidate on your machine. The
candidate's environment is scrubbed and the denylist blocks the obvious
escape hatches, but this is **not** a sandbox: Python code can read any file
your user can read. Use local `fix` on your own analyses with a provider you
trust; for pull requests from others, use the split CI workflow.

## Usage

### Local

```bash
export STATAUDIT_LLM_PROVIDER=anthropic        # or openai
export STATAUDIT_LLM_MODEL=<model id>
export STATAUDIT_LLM_API_KEY=...
# export STATAUDIT_LLM_REASONING_EFFORT=none   # OpenAI GPT-6 Luna

stataudit fix analysis.py                      # diff + before/after table, nothing written
stataudit fix analysis.py --apply              # overwrite with the fix, only if verified
stataudit fix notebook.ipynb --out fixed.ipynb --json fix.json
stataudit explain analysis.py                  # audit, then explain
stataudit explain report.json --no-llm         # deterministic summary only
```

### Split mode (what CI does)

```bash
# step 1 (holds the key, executes nothing): needs the findings of an earlier run
stataudit run analysis.py --json findings.json --fail-on never
stataudit fix analysis.py --propose-only --findings findings.json --out-dir cand/
# step 2 (no key; runs the original and the candidate)
stataudit fix analysis.py --verify-only cand/ --json fix.json
```

`--verify-only` re-audits the original itself rather than trusting the
findings file, and refuses a candidate proposed for a different version of
the file (SHA-256 of the original is recorded in `cand/manifest.json`).

### GitHub Actions

`stataudit init --agent` writes `.github/workflows/stataudit-agent.yml`,
which calls the reusable `agent.yml`:

| Job | Holds | Runs PR code? | Does |
|---|---|---|---|
| `gate` | read-only token (`actions`, `pull-requests`) | never (no checkout) | downloads the check artifact; continues only if a key and model are configured, an analysis is fragile or does not hold, the PR's current head is the checked commit, the PR is open, and the author is an owner/member/collaborator or a maintainer added the `stataudit:agent` label (or `run-for: everyone`) |
| `propose` | the LLM key | never (checks the PR out into `pr/` as data) | `stataudit agent propose`: reads `[agent]` from `stataudit.toml` at the **base** commit (off unless `enabled = true`), proposes up to `max_fixes` candidates |
| `verify` | nothing (`contents: read`, no secrets, no cache) | yes | installs the analyses' dependencies, `stataudit agent verify`: re-runs each original and candidate, merges the outcome into result.json |
| `comment` (caller) | `pull-requests: write` | never | re-renders and upserts the card with a "Suggested fix (verified)" section and its diff |

Setup:

```bash
stataudit init --agent
gh secret set STATAUDIT_LLM_API_KEY
gh variable set STATAUDIT_LLM_PROVIDER --body anthropic     # or openai
gh variable set STATAUDIT_LLM_MODEL --body <model id>
# optional: STATAUDIT_LLM_BASE_URL, STATAUDIT_LLM_REASONING_EFFORT (none for GPT-6 Luna)
```

and keep `[agent] enabled = true` in `stataudit.toml` on the default branch.
In CI there is one proposal per analysis (no refinement rounds: feeding the
verification outcome back would need the key in the job that runs PR code).

## Cost controls

* `max_tool_calls` (default 12) per proposal; after that the model is forced
  to submit (twice at most), then the attempt ends.
* `--rounds` (default 2) locally; one round in CI.
* `[agent] max_fixes` (default 3) analyses per pull request in CI.
* `explain` makes exactly one request.
* The gate skips forks by strangers unless a maintainer labels the PR.

Token usage is reported (`usage` in the JSON output, `usage` per proposal in
`proposals.json`). stataudit does not compute prices.

## Limitations

* Only analyses that `stataudit run` can trace can be verified (scipy.stats
  tests and statsmodels OLS). A fix that moves the analysis to a library
  stataudit does not trace is "no longer answered" and is rejected.
* Matching an original test to its replacement uses the data fingerprint or
  the model's explicit `replacements`; a fix that changes the data (for
  example a log transform) is only accepted when the model names the
  replacement.
* The success rate on real analyses and real models has not been measured.
* The static denylist can have false positives (the model is told why and can
  resubmit) and false negatives (it is not a sandbox).
