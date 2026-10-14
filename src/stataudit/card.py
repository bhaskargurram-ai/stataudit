"""Render the stataudit card (Markdown for a PR comment / job summary) from a *validated* result.

Everything in the result can originate from the pull request -- file paths,
code snippets, verdict texts built around them, and the agent's rationale and
diff. Before it is placed in the card it is:

* stripped of control characters and truncated;
* HTML-escaped (``<``, ``>``, ``&``), so no raw HTML, comments or the card
  marker can be injected;
* Markdown that could create links, images or formatting is backslash-escaped;
* defused for ``@mentions``, ``#123`` references and bare URLs (a zero-width
  joiner is inserted);
* pipe-escaped inside tables, with newlines collapsed;
* code (the agent's diff) goes in a fenced block whose fence is longer than
  any backtick run inside it.

The card ends with a machine-readable ``<!-- stataudit:data {...} -->`` block
(:mod:`stataudit.carddata`).
"""

from __future__ import annotations

import re
from typing import Any

from .carddata import card_data_block

__all__ = ["CARD_MARKER", "MAX_CARD_CHARS", "render_card", "escape_md", "code", "harness_warning"]

CARD_MARKER = "<!-- stataudit:card -->"
#: GitHub's comment limit is 65536 characters; stay well below it.
MAX_CARD_CHARS = 60_000
MAX_ROWS = 25
MAX_DIFF_CHARS = 12_000

ZWJ = "\u200d"
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u202a-\u202e\u2066-\u2069]")
_MD_SPECIAL_RE = re.compile(r"([\\`*_\[\]#!~|])")
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9])")
_URL_RE = re.compile(r"(?i)\b(https?|ftp|file|javascript|data)(:)")
_WWW_RE = re.compile(r"(?i)\bwww\.")

HEADLINES = {
    "pass": "✅ No statistical problems found",
    "warn": "⚠️ Some results need a second look",
    "fail": "❌ A conclusion does not hold or a reported result is wrong",
}
RUN_ICONS = {"holds": "✅", "fragile": "⚠️", "does_not_hold": "❌", "not_audited": "➖", "error": "❔"}
RUN_LABELS = {"holds": "holds", "fragile": "fragile", "does_not_hold": "does not hold", "not_audited": "not audited",
              "error": "error"}
CHECK_ICONS = {"consistent": "✅", "inconsistent": "⚠️", "gross": "❌", "impossible": "❌", "error": "❔"}
CHECK_LABELS = {"consistent": "consistent", "inconsistent": "inconsistent", "gross": "gross inconsistency",
                "impossible": "impossible value", "error": "error"}
FIX_HEAD = {
    "verified": "✅ Suggested fix (verified by re-running the audit)",
    "rejected": "❌ Proposed fix rejected by verification",
    "no_submission": "➖ No fix proposed",
    "nothing_to_fix": "➖ Nothing to fix",
    "proposed": "📝 Proposed fix (not verified)",
    "error": "❔ Fix attempt failed",
}


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _clean(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    s = _CONTROL_RE.sub(" ", s.replace("\r", " "))
    s = re.sub(r"\s+", " ", s).strip()
    return _truncate(s, limit)


def _defuse(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = _MENTION_RE.sub("@" + ZWJ, s)
    s = _URL_RE.sub(lambda m: m.group(1) + ZWJ + m.group(2), s)
    s = _WWW_RE.sub(lambda m: m.group(0)[:3] + ZWJ + ".", s)
    return s


def escape_md(text: Any, limit: int = 300) -> str:
    """Escape untrusted text for Markdown prose or a table cell."""
    s = _clean(text, limit)
    s = _MD_SPECIAL_RE.sub(r"\\\1", s)
    s = _defuse(s)
    s = re.sub(r"\\#(?=\d)", "#" + ZWJ, s)
    return s


def _no_comments(s: str) -> str:
    """Break up HTML comment delimiters so no copy of the card marker or data block appears in content."""
    return s.replace("<!--", "<!-" + ZWJ + "-").replace("-->", "-" + ZWJ + "->")


def code(text: Any, limit: int = 160) -> str:
    """Untrusted text as an inline code span that is safe inside a table cell."""
    s = _no_comments(_clean(text, limit).replace("`", "'"))
    # Code spans are inert on GitHub, but copies of the text (notifications, e-mail) may not be.
    s = _MENTION_RE.sub("@" + ZWJ, s)
    s = _URL_RE.sub(lambda m: m.group(1) + ZWJ + m.group(2), s)
    if not s:
        return "` `"
    s = s.replace("|", "\\|")
    return f"`{s}`"


def _fmt_p(p: Any) -> str:
    if not isinstance(p, (int, float)):
        return "—"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}".rstrip("0").rstrip(".") or "0"


def _fenced(text: str, lang: str = "") -> list[str]:
    body = _no_comments(_CONTROL_RE.sub(" ", text.replace("\r", "")))
    if len(body) > MAX_DIFF_CHARS:
        body = body[:MAX_DIFF_CHARS] + "\n… (diff truncated)"
    runs = re.findall(r"`{3,}|~{3,}", body)
    fence = "`" * max(3, max((len(r) for r in runs), default=0) + 1)
    return [f"{fence}{lang}", body.rstrip("\n"), fence]


def harness_warning(paths: list[str]) -> str:
    shown = ", ".join(code(p, 80) for p in paths[:5]) + (" …" if len(paths) > 5 else "")
    return (f"This PR changes files that control how this card is produced ({shown}). The checks ran with the "
            "PR's version of them; review those changes before trusting the card.")


def _summary_line(r: dict[str, Any]) -> str:
    parts = []
    if r["runs"]:
        parts.append(f"{len(r['runs'])} analysis run(s)")
    if r["reports"]:
        parts.append(f"{len(r['reports'])} report(s) checked")
    if r["lint"]:
        parts.append(f"{len(r['lint'])} file(s) linted")
    return " · ".join(parts) if parts else "no matching files changed"


def _runs_section(r: dict[str, Any]) -> list[str]:
    if not r["runs"]:
        return []
    out = ["", "### Analyses re-run on the data", "",
           "| | File | Tests | Holds | Fragile | Does not hold |", "|---|---|---|---|---|---|"]
    for run in r["runs"][:MAX_ROWS]:
        c = run["counts"]
        out.append(f"| {RUN_ICONS[run['status']]} | {code(run['path'], 100)} | {len(run['results'])} | {c['holds']} | "
                   f"{c['fragile']} | {c['does_not_hold']} |")
    problems = [(run, res) for run in r["runs"] for res in run["results"] if res["status"] in ("fragile", "does_not_hold")]
    errors = [run for run in r["runs"] if run.get("error")]
    if problems:
        out += ["", "<details open><summary>Conclusions that are fragile or do not hold</summary>", ""]
        for run, res in problems[:MAX_ROWS]:
            out.append(f"- {RUN_ICONS[res['status']]} {code(res['location'], 120)} **{escape_md(res['test'], 80)}** "
                       f"(p = {_fmt_p(res['reported_p'])}): {escape_md(res['verdict'], 600)}")
        if len(problems) > MAX_ROWS:
            out.append(f"- … and {len(problems) - MAX_ROWS} more")
        out += ["", "</details>"]
    for run in errors[:10]:
        out.append(f"\n> ❔ {code(run['path'], 100)} did not run to completion: {escape_md(run['error'], 300)}")
    return out


def _reports_section(r: dict[str, Any]) -> list[str]:
    if not r["reports"]:
        return []
    out = ["", "### Reported results recomputed (check-report)", ""]
    rows = [(rep, res) for rep in r["reports"] for res in rep["results"] if res["status"] != "consistent"]
    total = sum(len(rep["results"]) for rep in r["reports"])
    ok = total - len(rows)
    out.append(f"{total} APA-style result(s) found in {len(r['reports'])} file(s); {ok} consistent.")
    if rows:
        out += ["", "| | Where | Reported | Recomputed p | Problem |", "|---|---|---|---|---|"]
        for rep, res in rows[:MAX_ROWS]:
            out.append(f"| {CHECK_ICONS[res['status']]} | {code(rep['path'] + ' ' + res['location'], 120)} | "
                       f"{code(res['label'], 120)} | {_fmt_p(res['computed_p'])} | "
                       f"{escape_md(CHECK_LABELS[res['status']] + ': ' + res['message'], 300)} |")
        if len(rows) > MAX_ROWS:
            out.append(f"\n… and {len(rows) - MAX_ROWS} more")
    for rep in r["reports"]:
        if rep.get("error"):
            out.append(f"\n> ❔ {code(rep['path'], 100)}: {escape_md(rep['error'], 300)}")
    return out


def _lint_section(r: dict[str, Any]) -> list[str]:
    if not r["lint"]:
        return []
    rows = [(f, x) for f in r["lint"] for x in f["findings"]]
    warn = sum(x["severity"] == "warn" for _, x in rows)
    out = ["", "### Static checks (lint, no code run)", "",
           f"{len(r['lint'])} file(s) checked; {warn} warning(s), {len(rows) - warn} note(s)."]
    if rows:
        out += ["", "| Where | Rule | Finding |", "|---|---|---|"]
        for f, x in rows[:MAX_ROWS]:
            where = f["path"] + (f" cell {x['cell']}" if x.get("cell") else "") + f":{x['line']}"
            icon = "⚠️" if x["severity"] == "warn" else "ℹ️"
            out.append(f"| {code(where, 120)} | {icon} {x['rule']} | {escape_md(x['message'], 300)} |")
        if len(rows) > MAX_ROWS:
            out.append(f"\n… and {len(rows) - MAX_ROWS} more")
    for f in r["lint"]:
        if f.get("error"):
            out.append(f"\n> ❔ {code(f['path'], 100)}: {escape_md(f['error'], 300)}")
    return out


def _agent_section(r: dict[str, Any]) -> list[str]:
    a = r.get("agent")
    if not a:
        return []
    out = ["", "### Agent (experimental)", ""]
    if not a["fixes"]:
        out.append(f"No fix was attempted ({escape_md(a['status'], 40)}).")
    for fx in a["fixes"]:
        head = FIX_HEAD.get(fx["status"], fx["status"])
        change = ""
        if fx.get("before_status") and fx.get("after_status"):
            change = f" ({escape_md(fx['before_status'], 20)} → {escape_md(fx['after_status'], 20)})"
        out += [f"<details{' open' if fx['status'] == 'verified' else ''}><summary>{head}: "
                f"{escape_md(fx['path'], 100)}{change}</summary>", ""]
        if fx.get("rationale"):
            out.append("**Model's rationale** (unverified prose): " + escape_md(fx["rationale"], 1500))
            out.append("")
        if fx["rows"]:
            out += ["| Original | Before | After | Outcome |", "|---|---|---|---|"]
            for row in fx["rows"][:MAX_ROWS]:
                out.append(f"| {code(row.get('location') or '(new)', 100)} | "
                           f"{escape_md((row.get('before_test') or '—') + ' (' + (row.get('before_status') or '—') + ')', 120)} | "
                           f"{escape_md((row.get('after_test') or '—') + ' (' + (row.get('after_status') or 'dropped') + ')', 120)} | "
                           f"{escape_md(row.get('outcome') or '', 30)} |")
            out.append("")
        for reason in fx.get("reasons") or []:
            out.append(f"- ✗ {escape_md(reason, 400)}")
        if fx.get("diff") and fx["status"] in ("verified", "rejected", "proposed"):
            out += ["", *_fenced(fx["diff"], "diff")]
        if fx.get("model"):
            out.append(f"\n<sub>model: {escape_md(fx['model'], 80)}. Only stataudit's re-run decides whether a fix is "
                       "accepted; apply it yourself after review.</sub>")
        out += ["", "</details>", ""]
    for n in a.get("notes") or []:
        out.append(f"- {escape_md(n, 400)}")
    return out


def render_card(result: dict[str, Any], trusted_warnings: list[str] | None = None, _short: bool = False) -> str:
    """Render the card from a **validated** result (see :func:`stataudit.ci_result.validate_result`)."""
    head = result.get("head")
    meta = [f"stataudit {escape_md(result.get('tool_version'), 20)}"]
    if head:
        meta.append(f"head `{head[:12]}`")
    meta.append(f"α = {result['alpha']:g}")
    lines = [CARD_MARKER, f"## stataudit: {HEADLINES[result['status']]}", "", f"<sub>{' · '.join(meta)}</sub>", "",
             _summary_line(result)]
    for w in trusted_warnings or []:
        lines += ["", f"> ⚠️ {w}"]
    body = _runs_section(result) + _reports_section(result) + _lint_section(result) + _agent_section(result)
    for n in result.get("notes") or []:
        body.append(f"\n- note: {escape_md(n, 300)}")
    lines += body
    lines += ["", "<sub>stataudit re-runs the analysis on the real data, checks the tests' assumptions and recomputes "
              "reported p-values. It cannot know your study design and is not a substitute for a statistician.</sub>"]
    text = "\n".join(lines)
    block = card_data_block(result, len(trusted_warnings or []))
    if len(text) + len(block) + 2 > MAX_CARD_CHARS and result.get("agent") and not _short:
        # Diffs are the bulky part: drop them first (the artifact keeps them).
        slim = dict(result)
        slim["agent"] = dict(result["agent"])
        slim["agent"]["fixes"] = [{**f, "diff": ""} for f in result["agent"]["fixes"]]
        return render_card(slim, trusted_warnings, _short=True)
    if len(text) + len(block) + 2 > MAX_CARD_CHARS:
        cut = text[: MAX_CARD_CHARS - len(block) - 200].rsplit("\n", 1)[0]
        if cut.count("<details") > cut.count("</details>"):
            cut += "\n\n</details>"
        text = cut + "\n\n… (card truncated; the full result is in the job summary artifact)"
    return text.rstrip() + "\n\n" + block + "\n"
