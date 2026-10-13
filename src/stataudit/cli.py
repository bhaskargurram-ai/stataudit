"""Command-line interface: ``stataudit run``, ``stataudit check-data``, ``stataudit ui``."""

from __future__ import annotations

import sys
from pathlib import Path

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
        (html, report.to_html, "HTML"),
        (md, report.to_markdown, "Markdown"),
        (json_out, report.to_json, "JSON"),
    ):
        if path:
            Path(path).write_text(content(), encoding="utf-8")
            if not quiet:
                click.echo(f"{what} report written to {path}")


_common_opts = [
    click.option("--alpha", type=click.FloatRange(0, 1, min_open=True, max_open=True), default=0.05, show_default=True,
                 help="Significance level used to judge conclusions."),
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
    """stataudit: a second-opinion statistician for AI-generated analyses.

    Runs your analysis, records every statistical test it performs, checks
    the assumptions on the actual data and tells you whether each conclusion
    holds.
    """


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
    """Run SCRIPT under the tracer and audit every statistical test it performs.

    Arguments after ``--`` are passed to the script:

        stataudit run analysis.py --html report.html -- --input data.csv
    """
    from .audits import audit_records
    from .trace import trace_script

    tr = trace_script(script, args=list(script_args), timeout=timeout)
    if show_output and not quiet:
        if tr.stdout:
            click.echo(tr.stdout, nl=False)
        if tr.stderr:
            click.echo(tr.stderr, nl=False, err=True)
    report = audit_records(
        tr.records,
        alpha=alpha,
        source=script,
        script_error=tr.error,
        meta={"returncode": tr.returncode, "timed_out": tr.timed_out, "calls": len(tr.records)},
    )
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
