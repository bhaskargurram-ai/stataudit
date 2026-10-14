"""Command-line interface.

Commands: ``run``, ``check-data``, ``check-report``, ``lint``, ``fix``, ``explain``, ``ci``,
``summary``, ``comment``, ``agent propose``, ``agent verify``, ``init``, ``ui``, ``mcp``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from . import __version__
from .models import Report, Status

_FAIL_CHOICES = ("fragile", "fails", "never")


def _exit_code(report: Report, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    if report.status is Status.FAILS:
        return 1
    if report.status is Status.FRAGILE and fail_on == "fragile":
        return 1
    return 0


def _write(path: str | None, content: str, what: str, quiet: bool) -> None:
    if path:
        Path(path).write_text(content, encoding="utf-8")
        if not quiet:
            click.echo(f"{what} written to {path}", err=False)


def _emit(
    report: Report,
    html: str | None,
    md: str | None,
    json_out: str | None,
    quiet: bool,
    color: bool | None,
    verbose: bool = False,
) -> None:
    if not quiet:
        use_color = sys.stdout.isatty() if color is None else color
        click.echo(report.to_text(color=use_color, verbose=verbose), nl=False)
    for path, content, what in (
        (html, report.to_html, "HTML report"),
        (md, report.to_markdown, "Markdown report"),
        (json_out, report.to_json, "JSON report"),
    ):
        if path:
            _write(path, content(), what, quiet)


_ALPHA = click.option("--alpha", type=click.FloatRange(0, 1, min_open=True, max_open=True), default=0.05,
                      show_default=True, help="Significance level used to judge conclusions.")
_common_opts = [
    _ALPHA,
    click.option("--html", "html", type=click.Path(dir_okay=False), help="Write a self-contained HTML report."),
    click.option("--md", "md", type=click.Path(dir_okay=False), help="Write a Markdown report."),
    click.option("--json", "json_out", type=click.Path(dir_okay=False), help="Write a JSON report."),
    click.option("--fail-on", type=click.Choice(_FAIL_CHOICES), default="fragile", show_default=True,
                 help="Exit with code 1 when the overall status is at least this bad."),
    click.option("-q", "--quiet", is_flag=True, help="Do not print the terminal report."),
    click.option("--color/--no-color", default=None, help="Force or disable ANSI colours."),
    click.option("-v", "--verbose", is_flag=True, help="Show details for clean results too."),
]


def common(f):  # type: ignore[no-untyped-def]
    for opt in reversed(_common_opts):
        f = opt(f)
    return f


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="stataudit")
def main() -> None:
    """stataudit: check AI-written statistics with evidence instead of trust.

    Re-runs your analysis and audits every statistical test on the actual data
    (run), recomputes the p-values written in reports (check-report), and
    checks analysis code statically (lint). An optional agent proposes fixes
    that stataudit verifies by re-running them (fix).
    """


# --------------------------------------------------------------------------- run / check-data


@main.command()
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
@common
@click.option("--timeout", type=float, default=300, show_default=True, help="Seconds before the script is stopped.")
@click.option("--show-output", is_flag=True, help="Echo the script's own stdout/stderr.")
def run(
    script: str,
    script_args: tuple[str, ...],
    alpha: float,
    html: str | None,
    md: str | None,
    json_out: str | None,
    fail_on: str,
    quiet: bool,
    color: bool | None,
    verbose: bool,
    timeout: float,
    show_output: bool,
) -> None:
    """Run SCRIPT (a .py file or an .ipynb notebook) under the tracer and audit every statistical test.

    Arguments after ``--`` are passed to the script:

        stataudit run analysis.py --html report.html -- --input data.csv

    Notebooks run their code cells in order (IPython magics and shell lines are
    skipped with a note); call sites are reported as ``cell N, line M``.
    """
    from .audits import audit_records
    from .models import Finding, Severity

    notes: list[str] = []
    if script.lower().endswith(".ipynb"):
        from .notebook import NotebookError, trace_notebook

        if script_args:
            raise click.UsageError("notebooks do not take arguments")
        try:
            nt = trace_notebook(script, timeout=timeout)
        except NotebookError as exc:
            raise click.UsageError(str(exc)) from exc
        tr, notes = nt.trace, nt.notes
    else:
        from .trace import trace_script

        tr = trace_script(script, args=list(script_args), timeout=timeout)
    if show_output and not quiet:
        if tr.stdout:
            click.echo(tr.stdout, nl=False)
        if tr.stderr:
            click.echo(tr.stderr, nl=False, err=True)
    meta: dict[str, Any] = {"returncode": tr.returncode, "timed_out": tr.timed_out, "calls": len(tr.records)}
    if notes:
        meta["notes"] = notes
    report = audit_records(tr.records, alpha=alpha, source=script, script_error=tr.error, meta=meta)
    if notes:
        report.script_findings.append(Finding(
            "notebook_syntax", Severity.INFO, f"{len(notes)} IPython-only line(s) were skipped or rewritten before running",
            "; ".join(notes[:10]) + (" ..." if len(notes) > 10 else ""), {"notes": notes[:100]}))
    _emit(report, html, md, json_out, quiet, color, verbose)
    sys.exit(_exit_code(report, fail_on))


@main.command("check-data")
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.option("--test", "test", required=True,
              type=click.Choice(["ttest_ind", "mannwhitneyu", "f_oneway", "kruskal", "ttest_rel", "wilcoxon",
                                 "pearsonr", "spearmanr", "linregress", "chi2_contingency", "fisher_exact",
                                 "ttest_1samp"]),
              help="Which test to run and audit.")
@click.option("--value-col", help="Numeric outcome column (group tests, one-sample).")
@click.option("--group-col", help="Grouping column (group tests).")
@click.option("--x-col", help="First column (paired, correlation, regression, contingency).")
@click.option("--y-col", help="Second column (paired, correlation, regression, contingency).")
@click.option("--popmean", type=float, default=0.0, show_default=True, help="Null mean for ttest_1samp.")
@click.option("--welch", is_flag=True, help="Use Welch's t-test (equal_var=False) for ttest_ind.")
@click.option("--sep", default=None, help="CSV separator (default: auto-detect).")
@common
def check_data(
    data: str,
    test: str,
    value_col: str | None,
    group_col: str | None,
    x_col: str | None,
    y_col: str | None,
    popmean: float,
    welch: bool,
    sep: str | None,
    alpha: float,
    html: str | None,
    md: str | None,
    json_out: str | None,
    fail_on: str,
    quiet: bool,
    color: bool | None,
    verbose: bool,
) -> None:
    """Run TEST on columns of a CSV file and audit it, without writing a script.

        stataudit check-data data.csv --test ttest_ind --group-col g --value-col y
    """
    import pandas as pd

    from .api import audit_data

    try:
        df = pd.read_csv(data, sep=sep, engine="python" if sep is None else "c")
        report = audit_data(
            df, test, value_col=value_col, group_col=group_col, x_col=x_col, y_col=y_col,
            alpha=alpha, popmean=popmean, equal_var=not welch, source=data,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    _emit(report, html, md, json_out, quiet, color, verbose)
    sys.exit(_exit_code(report, fail_on))


# --------------------------------------------------------------------------- check-report


@main.command("check-report")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@_ALPHA
@click.option("--html", "html", type=click.Path(dir_okay=False), help="Write an HTML report (one file: the first input).")
@click.option("--md", "md", type=click.Path(dir_okay=False), help="Write a Markdown report.")
@click.option("--json", "json_out", type=click.Path(dir_okay=False), help="Write a JSON report.")
@click.option("--fail-on", type=click.Choice(["inconsistent", "gross", "never"]), default="inconsistent",
              show_default=True, help="Exit 1 on any inconsistency (or impossible value), only on gross ones, or never.")
@click.option("-q", "--quiet", is_flag=True, help="Do not print the terminal report.")
@click.option("--color/--no-color", default=None, help="Force or disable ANSI colours.")
@click.option("-v", "--verbose", is_flag=True, help="List consistent results too.")
def check_report(files: tuple[str, ...], alpha: float, html: str | None, md: str | None, json_out: str | None,
                 fail_on: str, quiet: bool, color: bool | None, verbose: bool) -> None:
    """Recompute the p-values of APA-style results in Markdown, text, HTML or PDF FILES.

    Finds results such as ``t(28) = 2.20, p = .036``, ``F(2, 45) = 3.81, p = .03``,
    ``r(48) = .32, p = .02``, ``χ²(1, N = 90) = 4.1, p < .05`` and ``z = 2.3, p = .02``,
    recomputes p from the statistic and degrees of freedom (allowing for rounding),
    and flags inconsistent results, gross inconsistencies (the significance
    decision flips at --alpha) and impossible values. PDFs need ``pdftotext``.
    """
    from .reportcheck import GROSS, IMPOSSIBLE, INCONSISTENT, check_file

    checks = []
    for f in files:
        try:
            checks.append(check_file(f, alpha=alpha))
        except (ValueError, OSError) as exc:
            raise click.UsageError(f"{f}: {exc}") from exc
    use_color = sys.stdout.isatty() if color is None else color
    if not quiet:
        for rc in checks:
            click.echo(rc.to_text(color=use_color, verbose=verbose))
    if md:
        _write(md, "\n".join(rc.to_markdown() for rc in checks), "Markdown report", quiet)
    if html:
        _write(html, checks[0].to_html(), "HTML report", quiet)
    if json_out:
        payload: Any = checks[0].to_dict() if len(checks) == 1 else [rc.to_dict() for rc in checks]
        _write(json_out, json.dumps(payload, indent=2, ensure_ascii=False), "JSON report", quiet)
    bad = {INCONSISTENT, GROSS, IMPOSSIBLE} if fail_on == "inconsistent" else {GROSS, IMPOSSIBLE}
    failed = fail_on != "never" and any(r.status in bad for rc in checks for r in rc.results)
    sys.exit(1 if failed else 0)


# --------------------------------------------------------------------------- lint


@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--select", default="", help="Comma-separated rule ids to run (default: all).")
@click.option("--ignore", default="", help="Comma-separated rule ids to skip.")
@click.option("--md", "md", type=click.Path(dir_okay=False), help="Write a Markdown report.")
@click.option("--json", "json_out", type=click.Path(dir_okay=False), help="Write a JSON report.")
@click.option("--fail-on", type=click.Choice(["warn", "info", "never"]), default="warn", show_default=True,
              help="Exit 1 when there is a finding of at least this severity.")
@click.option("-q", "--quiet", is_flag=True, help="Do not print the terminal report.")
@click.option("-v", "--verbose", is_flag=True, help="Explain each rule.")
@click.option("--list-rules", is_flag=True, is_eager=True, expose_value=False, help="List the rules and exit.",
              callback=lambda ctx, p, v: _list_rules(ctx) if v else None)
def lint(files: tuple[str, ...], select: str, ignore: str, md: str | None, json_out: str | None, fail_on: str,
         quiet: bool, verbose: bool) -> None:
    """Check analysis code statically (FILES: .py or .ipynb). Nothing is executed and no data is needed.

    Rules: SA101 uncorrected multiple tests, SA102 Student's t-test without a
    variance check, SA103 Pearson without a plot/outlier check, SA104
    chi-square with small expected counts, SA105 test shopping, SA106 selecting
    results by p < 0.05 across many tests, SA107 normality pre-testing.
    Suppress one with ``# stataudit: ignore[SA102]`` on the reported line.
    """
    from .lint import RULES, lint_file
    from .notebook import NotebookError

    sel = [s.strip() for s in select.split(",") if s.strip()] or None
    ign = [s.strip() for s in ignore.split(",") if s.strip()]
    unknown = [r for r in (sel or []) + ign if r.upper() not in RULES]
    if unknown:
        raise click.UsageError(f"unknown rule id(s): {', '.join(unknown)}")
    reports = []
    for f in files:
        try:
            reports.append(lint_file(f, select=sel, ignore=ign))
        except (NotebookError, OSError) as exc:
            raise click.UsageError(f"{f}: {exc}") from exc
    if not quiet:
        for rep in reports:
            click.echo(rep.to_text(verbose=verbose))
    if md:
        _write(md, "\n".join(r.to_markdown() for r in reports), "Markdown report", quiet)
    if json_out:
        payload: Any = reports[0].to_dict() if len(reports) == 1 else [r.to_dict() for r in reports]
        _write(json_out, json.dumps(payload, indent=2, ensure_ascii=False), "JSON report", quiet)
    levels = {"warn": ("warn",), "info": ("warn", "info"), "never": ()}[fail_on]
    sys.exit(1 if any(f.severity in levels for r in reports for f in r.findings) else 0)


def _list_rules(ctx: click.Context) -> None:
    from .lint import RULES

    for r in RULES.values():
        click.echo(f"{r.id}  {r.name}  [{r.severity}]\n    {r.summary}\n    fix: {r.fix}\n")
    ctx.exit(0)


# --------------------------------------------------------------------------- agent: fix / explain


def _llm_factory(config_path: str | None = None):  # type: ignore[no-untyped-def]
    from .agent.common import llm_from_env
    from .config import ConfigError, load_config

    try:
        cfg = load_config(".", path=config_path)
    except ConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    return lambda: llm_from_env(llm_cfg=cfg.llm), cfg


@main.command()
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option("--rounds", type=click.IntRange(1, 5), default=2, show_default=True,
              help="Proposals the model may make; each rejected one is fed back with the reasons.")
@click.option("--apply", "apply_", is_flag=True, help="Overwrite SCRIPT with the fix (only if it was verified).")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), help="Write the corrected file here (only if verified).")
@click.option("--json", "json_out", type=click.Path(dir_okay=False), help="Write the fix result as JSON.")
@click.option("--md", "md", type=click.Path(dir_okay=False), help="Write the fix result as Markdown.")
@click.option("--share-data", is_flag=True, help="Let the model see raw data values (first rows/values). Off by default: "
                                                 "it only sees summaries (n, missing, mean, sd; CSV column names and dtypes).")
@click.option("--timeout", type=float, default=120, show_default=True, help="Seconds per analysis run.")
@click.option("--max-tool-calls", type=click.IntRange(1, 50), default=12, show_default=True)
@_ALPHA
@click.option("--config", "config_path", type=click.Path(dir_okay=False), help="Config file ([llm] fallbacks).")
@click.option("--propose-only", is_flag=True, help="CI step 1: ask the model for a candidate using --findings; "
                                                   "executes nothing. Writes --out-dir.")
@click.option("--findings", type=click.Path(exists=True, dir_okay=False), help="JSON from `stataudit run --json` "
                                                                                "(for --propose-only).")
@click.option("--out-dir", type=click.Path(file_okay=False), help="Candidate directory (for --propose-only).")
@click.option("--verify-only", "verify_dir", type=click.Path(exists=True, file_okay=False),
              help="CI step 2: verify the candidate in this directory (runs the analysis; no LLM).")
@click.option("-q", "--quiet", is_flag=True)
def fix(script: str, rounds: int, apply_: bool, out_path: str | None, json_out: str | None, md: str | None,
        share_data: bool, timeout: float, max_tool_calls: int, alpha: float, config_path: str | None,
        propose_only: bool, findings: str | None, out_dir: str | None, verify_dir: str | None, quiet: bool) -> None:
    """EXPERIMENTAL: have an LLM propose a corrected analysis, and accept it only if stataudit verifies it.

    stataudit audits SCRIPT (.py or .ipynb), gives the model read-only tools
    (the script, the findings, data summaries), runs the proposed version in a
    subprocess (timeout, scrubbed environment, static denylist) under the
    tracer, and accepts it only if every original question is still answered,
    those answers now hold (or improved), and nothing new breaks. Prints a diff
    and a before/after table; files change only with --apply or --out.

    Needs STATAUDIT_LLM_PROVIDER, STATAUDIT_LLM_MODEL and STATAUDIT_LLM_API_KEY.
    Local mode runs the model's code on this machine (in a subprocess); see docs/AGENT.md.
    """
    from .agent import fix as fx
    from .agent.llm import LLMConfigError, LLMToolError

    if propose_only and verify_dir:
        raise click.UsageError("--propose-only and --verify-only are separate steps")
    log = (lambda s: None) if quiet else (lambda s: click.echo(s, err=True))
    try:
        if verify_dir:
            res = fx.verify_only(script, verify_dir, alpha=alpha, timeout=timeout)
        else:
            factory, cfg = _llm_factory(config_path)
            if propose_only:
                if not findings or not out_dir:
                    raise click.UsageError("--propose-only needs --findings and --out-dir")
                data = json.loads(Path(findings).read_text(encoding="utf-8"))
                res = fx.propose_only(script, data, factory, out_dir, share_data=share_data,
                                      max_tool_calls=max_tool_calls, log=log)
            else:
                res = fx.run_fix(script, factory, rounds=rounds, alpha=alpha, timeout=timeout, share_data=share_data,
                                 max_tool_calls=max_tool_calls, log=log)
    except LLMConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    except (fx.FixError, LLMToolError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not quiet:
        click.echo(res.to_text(), nl=False)
    if json_out:
        _write(json_out, json.dumps(res.to_dict(), indent=2, ensure_ascii=False), "JSON", quiet)
    if md:
        _write(md, res.to_markdown(), "Markdown", quiet)
    if res.accepted and res.candidate:
        target = fx.load_target(script)
        content = target.patched_file_text(res.candidate.code)
        if out_path:
            _write(out_path, content, "Corrected file", quiet)
        if apply_:
            Path(script).write_text(content, encoding="utf-8")
            if not quiet:
                click.echo(f"Applied the verified fix to {script}")
    elif (apply_ or out_path) and not quiet:
        click.echo("Nothing written: no verified fix.", err=True)
    sys.exit(0 if res.status in (fx.VERIFIED, fx.NOTHING_TO_FIX, fx.PROPOSED) else 1)


@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), help="Write the explanation (Markdown) here.")
@click.option("--json", "json_out", type=click.Path(dir_okay=False), help="Write the result as JSON.")
@click.option("--no-llm", is_flag=True, help="Only print the deterministic summary.")
@_ALPHA
@click.option("--config", "config_path", type=click.Path(dir_okay=False), help="Config file ([llm] fallbacks).")
def explain(source: str, out_path: str | None, json_out: str | None, no_llm: bool, alpha: float,
            config_path: str | None) -> None:
    """EXPERIMENTAL: explain findings in plain language for a non-statistician.

    SOURCE is a JSON report (from run, check-report or lint with --json), or a
    .py/.ipynb analysis (audited first) or a .md/.txt/.html/.pdf report
    (checked first). Every number in the LLM's text must appear in the
    findings; otherwise the text is rejected and a deterministic summary is
    printed instead. Without STATAUDIT_LLM_* settings the deterministic
    summary is printed.
    """
    from .agent.explain import explain as do_explain

    findings = _load_findings(source, alpha)
    factory = None
    if not no_llm:
        factory, _cfg = _llm_factory(config_path)
    res = do_explain(findings, factory, log=lambda s: click.echo(s, err=True))
    click.echo(res.text)
    if res.source != "llm":
        for n in res.notes:
            click.echo(f"(deterministic summary: {n})", err=True)
    if out_path:
        Path(out_path).write_text(res.text + "\n", encoding="utf-8")
    if json_out:
        Path(json_out).write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_findings(source: str, alpha: float) -> dict[str, Any]:
    p = Path(source)
    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise click.UsageError(f"{source} is not valid JSON") from exc
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict) or data.get("tool") != "stataudit":
            raise click.UsageError(f"{source} is not a stataudit JSON report")
        return data
    if suffix in (".py", ".ipynb"):
        from .api import audit_script

        return audit_script(p, alpha=alpha).to_dict()
    from .reportcheck import check_file

    try:
        return check_file(p, alpha=alpha).to_dict()
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


# --------------------------------------------------------------------------- CI / GitHub


@main.command("ci")
@click.option("--repo", "repo_dir", default=".", show_default=True, type=click.Path(file_okay=False, exists=True))
@click.option("--base", help="Base commit: changed files are diffed against it, and the config is read from it.")
@click.option("--head", help="Head commit (default HEAD).")
@click.option("--files", "files", multiple=True, help="Check these paths instead of a git diff (repeatable).")
@click.option("--all", "all_files", is_flag=True, help="Lint/check every tracked file, not just changed ones.")
@click.option("--config", "config_path", help="Config file (default: stataudit.toml, at --base when given).")
@click.option("--out-dir", default="stataudit-out", show_default=True, type=click.Path(file_okay=False),
              help="Where result.json, card.md, pr_number and head_sha are written.")
@click.option("--fail-on", type=click.Choice(["config", "never", "warn", "fail"]), default="config", show_default=True,
              help="Exit 1 when the overall status is at least this bad (config: fail_on from stataudit.toml).")
@click.option("--pr", "pr_number", type=click.IntRange(1), help="Pull request number (default: from $GITHUB_EVENT_PATH).")
@click.option("-q", "--quiet", is_flag=True)
def ci_cmd(repo_dir: str, base: str | None, head: str | None, files: tuple[str, ...], all_files: bool,
           config_path: str | None, out_dir: str, fail_on: str, pr_number: int | None, quiet: bool) -> None:
    """Check what a pull request changed: lint .py/.ipynb, check-report .md/.txt/.html, re-run `run` analyses.

    Writes OUT_DIR/result.json and OUT_DIR/card.md (the GitHub card). Used by
    the reusable workflow check.yml; also works locally:

        stataudit ci --base origin/main
    """
    from .ci import exit_code, run_ci, write_outputs
    from .ci_result import ResultValidationError
    from .config import ConfigError
    from .github_api import pr_number_from_event

    if pr_number is None:
        pr_number = pr_number_from_event()
    slug = os.environ.get("GITHUB_REPOSITORY") or None
    try:
        result = run_ci(repo_dir, base=base, head=head, files=list(files) if files else None, all_files=all_files,
                        config_path=config_path, pr_number=pr_number, repo_slug=slug,
                        log=(lambda s: None) if quiet else (lambda s: click.echo(s, err=True)))
        rj, card = write_outputs(result, out_dir)
    except (ConfigError, RuntimeError, ResultValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not quiet:
        click.echo(card.read_text(encoding="utf-8"))
        click.echo(f"result: {rj}  card: {card}", err=True)
    eff = result["fail_on"] if fail_on == "config" else fail_on
    sys.exit(exit_code(result["status"], eff))


@main.command("summary")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True, dir_okay=False), help="result.json to render.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
              help="Append the card here (default: $GITHUB_STEP_SUMMARY, else stdout).")
def summary_cmd(json_path: str, out_path: str | None) -> None:
    """Render the card from result.json into the job summary (validates it first; needs no token)."""
    from .ci import write_summary
    from .ci_result import ResultValidationError

    try:
        dest = write_summary(json_path, out_path)
    except (ValueError, ResultValidationError, OSError) as exc:
        raise click.ClickException(f"could not write summary: {exc}") from exc
    if dest:
        click.echo(f"card appended to {dest}", err=True)


@main.command("comment")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--pr", "pr_number", help="Pull request number (default: from result.json or the event).")
@click.option("--repo", "repo_slug", help="owner/name (default: $GITHUB_REPOSITORY).")
@click.option("--bot-login", help="Only reuse an existing card comment written by this login.")
@click.option("--expect-head-sha", help="Refuse unless result.json AND the PR's current head match this commit.")
@click.option("--check-pr-files/--no-check-pr-files", default=True, show_default=True,
              help="Warn on the card when the PR changes workflows or the stataudit config.")
@click.option("--labels", is_flag=True, help="Also set a `stataudit: pass|warnings|problems` label.")
@click.option("--dry-run", is_flag=True, help="Print the card instead of posting it.")
def comment_cmd(json_path: str, pr_number: str | None, repo_slug: str | None, bot_login: str | None,
                expect_head_sha: str | None, check_pr_files: bool, labels: bool, dry_run: bool) -> None:
    """Post or update the stataudit card on a pull request (for the privileged workflow_run job).

    result.json is treated as untrusted: it is validated and the card is
    re-rendered from it (card.md is never posted). Needs GITHUB_TOKEN with
    pull-requests: write.
    """
    from .card import harness_warning, render_card
    from .ci_result import MAX_RESULT_BYTES, ResultValidationError, validate_result
    from .github_api import GitHubError, env_client, harness_changes, pr_number_from_event, validate_number

    p = Path(json_path)
    if p.stat().st_size > MAX_RESULT_BYTES:
        raise click.ClickException("result.json is too large")
    try:
        result = validate_result(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, ResultValidationError) as exc:
        raise click.ClickException(f"invalid result.json: {exc}") from exc
    try:
        if pr_number is not None:
            number = validate_number(pr_number)
        elif result.get("pr_number"):
            number = validate_number(result["pr_number"])
        else:
            n = pr_number_from_event()
            if n is None:
                raise click.ClickException("PR number not given (--pr) and not found in result.json or the event")
            number = n
    except GitHubError as exc:
        raise click.ClickException(str(exc)) from exc
    card = render_card(result)
    if dry_run:
        click.echo(card)
        return
    env = dict(os.environ)
    if repo_slug:
        env["GITHUB_REPOSITORY"] = repo_slug
    if not env.get("GITHUB_TOKEN"):
        raise click.ClickException("GITHUB_TOKEN is not set")
    try:
        client = env_client(env)
        if expect_head_sha:
            expected = expect_head_sha.strip().lower()
            if result["head"] != expected:
                raise click.ClickException("result.json head commit does not match --expect-head-sha")
            pr = client.get_pull(number)
            if str((pr.get("head") or {}).get("sha", "")).lower() != expected:
                raise click.ClickException(f"PR #{number} head does not match --expect-head-sha "
                                           "(PR updated, or wrong PR number)")
        if check_pr_files:
            changed = harness_changes(client.list_pull_files(number))
            if changed:
                card = render_card(result, trusted_warnings=[harness_warning(changed)])
        action, cid = client.upsert_card_comment(number, card, bot_login=bot_login)
    except GitHubError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{action} stataudit card comment {cid} on #{number}")
    if labels:
        try:
            added, removed = client.sync_verdict_label(number, result["status"])
        except GitHubError as exc:
            click.echo(f"warning: could not update labels: {exc}", err=True)
        else:
            click.echo(f"labels: added {added or 'none'}, removed {removed or 'none'}")


@main.group()
def agent() -> None:
    """EXPERIMENTAL: the agent's CI steps (propose with the key, verify without it)."""


@agent.command("propose")
@click.option("--result", "result_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="result.json of the check job (validated).")
@click.option("--repo-dir", default=".", show_default=True, type=click.Path(file_okay=False, exists=True),
              help="Pull request checkout, read as data only.")
@click.option("--base", help="Base commit: the [agent] config is read from it.")
@click.option("--out-dir", required=True, type=click.Path(file_okay=False), help="Where candidates are written.")
def agent_propose_cmd(result_path: str, repo_dir: str, base: str | None, out_dir: str) -> None:
    """Propose fixes for analyses that are fragile or do not hold. Holds the key; executes nothing from the PR."""
    from .agent.ci import agent_propose
    from .agent.common import llm_from_env
    from .agent.llm import LLMConfigError
    from .ci_result import ResultValidationError
    from .config import ConfigError, load_config

    try:
        cfg = load_config(repo_dir, rev=base)
        summary = agent_propose(result_path, repo_dir, out_dir, lambda: llm_from_env(llm_cfg=cfg.llm), base=base,
                                config=cfg, log=lambda s: click.echo(s, err=True))
    except LLMConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    except (ConfigError, ResultValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for p in summary["proposals"]:
        click.echo(f"{p['path']}: {p['status']}")
    for n in summary["notes"]:
        click.echo(f"note: {n}")


@agent.command("verify")
@click.option("--result", "result_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--candidates", required=True, type=click.Path(file_okay=False, exists=True))
@click.option("--repo-dir", default=".", show_default=True, type=click.Path(file_okay=False, exists=True))
@click.option("--json", "json_out", required=True, type=click.Path(dir_okay=False), help="Updated result.json.")
@click.option("--out", "card_out", type=click.Path(dir_okay=False), help="Rendered card.")
@click.option("--timeout", type=float, default=300, show_default=True)
def agent_verify_cmd(result_path: str, candidates: str, repo_dir: str, json_out: str, card_out: str | None,
                     timeout: float) -> None:
    """Verify candidates by re-running the analyses (no LLM; use in a job without secrets)."""
    from .agent.ci import agent_verify
    from .card import render_card
    from .ci_result import ResultValidationError, validate_result

    try:
        result = agent_verify(result_path, candidates, repo_dir, timeout=timeout, log=lambda s: click.echo(s, err=True))
    except (ResultValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    Path(json_out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    if card_out:
        Path(card_out).write_text(render_card(validate_result(result)), encoding="utf-8")
    for f in result["agent"]["fixes"]:
        click.echo(f"{f['path']}: {f['status']}")


@main.command()
@click.option("--repo", "repo_dir", default=".", show_default=True, type=click.Path(file_okay=False, exists=True))
@click.option("--agent", "with_agent", is_flag=True, help="Also write the experimental agent workflow.")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
@click.option("--ref", help="Tag or commit of stataudit the workflows use (default: this version's tag).")
@click.option("--install-command", help="Installs the analyses' dependencies in CI (default: detected).")
def init(repo_dir: str, with_agent: bool, force: bool, ref: str | None, install_command: str | None) -> None:
    """Write the GitHub workflows and a commented stataudit.toml into a repository."""
    from .init_cmd import InitError, init_repo

    try:
        res = init_repo(repo_dir, agent=with_agent, force=force, ref=ref, install_command=install_command)
    except InitError as exc:
        raise click.ClickException(str(exc)) from exc
    for f in res.written:
        click.echo(f"wrote {f}")
    click.echo("\nNext steps:")
    for i, s in enumerate(res.next_steps, 1):
        click.echo(f"  {i}. {s}")


# --------------------------------------------------------------------------- ui / mcp


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind (local only by default).")
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--debug", is_flag=True, hidden=True)
def ui(host: str, port: int, debug: bool) -> None:
    """Start the local web UI (upload a CSV, pick a test, read the report)."""
    from .web import create_app

    if host not in ("127.0.0.1", "localhost", "::1"):
        click.echo(f"warning: binding to {host} exposes the UI to your network", err=True)
    click.echo(f"stataudit UI on http://{host}:{port}  (Ctrl+C to stop)")
    create_app().run(host=host, port=port, debug=debug)


@main.command()
def mcp() -> None:
    """Run the MCP server over stdio (same as the ``stataudit-mcp`` command)."""
    from .mcp_server import main as mcp_main

    mcp_main()


if __name__ == "__main__":  # pragma: no cover
    main()
