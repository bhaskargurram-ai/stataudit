"""A small local web UI: upload a CSV, choose a test and columns, read the report.

Start it with ``stataudit ui``. It binds to 127.0.0.1 by default, keeps
nothing on disk and limits uploads to :data:`MAX_UPLOAD_BYTES`.
"""

from __future__ import annotations

import io
from typing import Any

import jinja2
from flask import Flask, request

from .api import DATA_TESTS, audit_data
from .reports.html import _env, render_html

__all__ = ["create_app", "MAX_UPLOAD_BYTES"]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>stataudit</title><style>{% include "_style.css" %}</style></head>
<body><main>
<h1>stataudit</h1>
<p class="muted">Upload a CSV, pick a test and the columns it should use. The test is run on your data and audited
locally; nothing is stored.</p>
<section class="card">
<form class="upload" method="post" action="/audit" enctype="multipart/form-data">
  <label>CSV file <input type="file" name="file" accept=".csv,text/csv" required></label>
  <label>Test <select name="test">
    {% for t, desc in tests.items() %}<option value="{{ t }}" {% if form.test == t %}selected{% endif %}>{{ t }} — {{ desc }}</option>{% endfor %}
  </select></label>
  <label>Value column (group tests, one-sample) <input name="value_col" value="{{ form.value_col or '' }}"></label>
  <label>Group column (group tests) <input name="group_col" value="{{ form.group_col or '' }}"></label>
  <label>X column (paired / correlation / regression / contingency) <input name="x_col" value="{{ form.x_col or '' }}"></label>
  <label>Y column <input name="y_col" value="{{ form.y_col or '' }}"></label>
  <label>Alpha <input name="alpha" value="{{ form.alpha or '0.05' }}" inputmode="decimal"></label>
  <label><span><input type="checkbox" name="welch" value="1" {% if form.welch %}checked{% endif %}> Welch (unequal variances)</span></label>
  <button type="submit">Audit</button>
</form>
{% if columns %}<p class="muted">Columns in your file: {{ columns|join(', ') }}</p>{% endif %}
</section>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
{% if report_html %}{{ report_html|safe }}{% endif %}
<footer>stataudit checks assumptions and robustness. It cannot know your study design and is not a substitute
for a statistician.</footer>
</main></body></html>"""


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    if config:
        app.config.update(config)
    env: jinja2.Environment = _env()
    page = env.from_string(_PAGE)

    def render(status: int = 200, **ctx: Any) -> tuple[str, int]:
        ctx.setdefault("form", {})
        return page.render(tests=DATA_TESTS, **ctx), status

    @app.get("/")
    def index() -> tuple[str, int]:
        return render()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/audit")
    def audit() -> tuple[str, int]:
        import pandas as pd

        form = {k: (request.form.get(k) or "").strip() for k in ("test", "value_col", "group_col", "x_col", "y_col", "alpha", "welch")}
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return render(400, form=form, error="Please choose a CSV file.")
        try:
            raw = upload.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                return render(413, form=form, error="File too large.")
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        except Exception as exc:
            return render(400, form=form, error=f"Could not read the CSV: {exc}")
        columns = [str(c) for c in df.columns]
        try:
            alpha = float(form["alpha"] or 0.05)
            if not 0 < alpha < 1:
                raise ValueError("alpha must be between 0 and 1")
            rep = audit_data(
                df,
                form["test"],
                value_col=form["value_col"] or None,
                group_col=form["group_col"] or None,
                x_col=form["x_col"] or None,
                y_col=form["y_col"] or None,
                alpha=alpha,
                equal_var=not form["welch"],
                source=upload.filename,
            )
        except ValueError as exc:
            return render(400, form=form, columns=columns, error=str(exc))
        return render(200, form=form, columns=columns, report_html=render_html(rep, embedded=True))

    @app.errorhandler(413)
    def too_large(_: Any) -> tuple[str, int]:
        return render(413, error="File too large.")

    return app
