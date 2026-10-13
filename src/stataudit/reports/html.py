"""Self-contained HTML report rendered with Jinja2 (no external assets)."""

from __future__ import annotations

from importlib import resources

import jinja2

from ..audits.common import fmt_num, fmt_p
from ..models import Report, Role, Severity, Status
from ..verdict import STATUS_ICON

_HEADLINE = {
    Status.HOLDS: "Conclusions hold",
    Status.FRAGILE: "Some conclusions are fragile",
    Status.FAILS: "At least one conclusion does not hold",
    Status.NOT_AUDITED: "Nothing to audit",
}


def _env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FunctionLoader(
            lambda name: resources.files("stataudit.reports").joinpath("templates", name).read_text(encoding="utf-8")
        ),
        autoescape=jinja2.select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["p"] = fmt_p
    env.filters["num"] = fmt_num
    env.globals.update(
        icon=STATUS_ICON, headline=_HEADLINE, Role=Role, Severity=Severity, Status=Status
    )
    return env


def render_html(report: Report, title: str | None = None, embedded: bool = False) -> str:
    """Render a standalone HTML page (or, with ``embedded=True``, just the report body)."""
    tpl = _env().get_template("report.html.j2")
    return tpl.render(report=report, title=title or "stataudit report", embedded=embedded)
