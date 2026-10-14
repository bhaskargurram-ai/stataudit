"""``stataudit ci``: check the files a pull request changes and write result.json + card.md.

* **lint** every changed ``.py`` / ``.ipynb`` matching ``lint`` globs (static, nothing runs);
* **check-report** every changed ``.md`` / ``.txt`` / ``.html`` (and ``.pdf`` when ``pdftotext``
  exists) matching ``reports`` globs;
* **run** every file matching the ``run`` globs (opt-in: these execute, so they
  only run in the unprivileged job). They run whether or not they changed,
  because a change to data or to an imported module can change their result.

Configuration comes from ``stataudit.toml`` at the pull request's **base**
commit when ``--base`` is given.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from .ci_result import new_result, overall_status
from .config import Config, glob_match, load_config

__all__ = ["run_ci", "changed_files", "tracked_files", "agent_report", "write_outputs", "exit_code"]

LINT_EXT = (".py", ".ipynb")
REPORT_EXT = (".md", ".markdown", ".txt", ".html", ".htm", ".rst")
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", "build", "dist", ".stataudit-tool"}


def _git(repo: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def changed_files(repo: str | Path, base: str, head: str = "HEAD") -> list[str]:
    """Files added, copied, modified or renamed between the merge base of ``base`` and ``head``."""
    root = Path(repo)
    out = _git(root, "diff", "--name-only", "--diff-filter=ACMR", "-z", f"{base}...{head}")
    if out is None:
        out = _git(root, "diff", "--name-only", "--diff-filter=ACMR", "-z", base, head)
    if out is None:
        raise RuntimeError(f"git diff {base}...{head} failed (is the base commit fetched? use fetch-depth: 0)")
    return sorted({p for p in out.split("\0") if p})


def tracked_files(repo: str | Path) -> list[str]:
    root = Path(repo)
    out = _git(root, "ls-files", "-z")
    if out is not None:
        return sorted({p for p in out.split("\0") if p})
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            files.append(Path(dirpath, f).relative_to(root).as_posix())
    return sorted(files)


def _safe(root: Path, rel: str) -> Path | None:
    """``root/rel`` if it is a regular file inside ``root`` (symlinks may not point outside)."""
    if not rel or rel.startswith("/") or ".." in rel.split("/"):
        return None
    p = root / rel
    try:
        real = p.resolve()
        root_r = root.resolve()
    except OSError:
        return None
    if not real.is_file() or not (root_r in real.parents):
        return None
    return p


def _match(path: str, globs: Iterable[str]) -> bool:
    return any(glob_match(path, g) for g in globs)


def agent_report(rep: dict[str, Any]) -> dict[str, Any]:
    """The parts of a run report the agent needs later (no raw data), size-capped."""
    from .agent.fix import compact_findings

    out = compact_findings(rep)
    by_loc = {r.get("location"): r for r in rep.get("results") or [] if isinstance(r, dict)}
    for r in out["results"]:
        full = by_loc.get(r["location"]) or {}
        r["sample_sizes"] = full.get("sample_sizes")
        r["inputs"] = full.get("inputs")
    while len(json.dumps(out)) > 300_000 and out["results"]:
        out["results"].pop()
        out["truncated"] = True
    return out


def _lint_entry(root: Path, rel: str, cfg: Config) -> dict[str, Any]:
    from .lint import lint_file

    p = _safe(root, rel)
    if p is None:
        return {"path": rel, "status": "error", "error": "not a regular file inside the repository", "findings": []}
    try:
        rep = lint_file(p, ignore=cfg.lint_ignore)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        return {"path": rel, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500], "findings": []}
    status = rep.status
    if rep.notes and not rep.findings and rep.tests_found == 0 and any("could not parse" in n for n in rep.notes):
        status = "error"
    return {
        "path": rel, "status": status, "error": "; ".join(rep.notes)[:500] if status == "error" else None,
        "tests_found": rep.tests_found,
        "findings": [{"rule": f.rule, "line": f.line, "cell": f.cell, "severity": f.severity, "message": f.message[:500]}
                     for f in rep.findings[:50]],
    }


def _report_entry(root: Path, rel: str, cfg: Config) -> dict[str, Any]:
    from .reportcheck import check_file

    p = _safe(root, rel)
    if p is None:
        return {"path": rel, "status": "error", "error": "not a regular file inside the repository", "results": []}
    try:
        rc = check_file(p, alpha=cfg.alpha)
    except Exception as exc:  # noqa: BLE001
        return {"path": rel, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500], "results": []}
    return {
        "path": rel, "status": rc.status, "error": None, "counts": rc.counts(),
        "results": [{"location": r.location[:60], "label": r.label[:200], "computed_p": r.computed_p,
                     "status": r.status, "message": r.message[:500]} for r in rc.results[:200]],
    }


def _run_entry(root: Path, rel: str, cfg: Config) -> dict[str, Any]:
    from .api import audit_script

    p = _safe(root, rel)
    if p is None:
        return {"path": rel, "status": "error", "error": "not a regular file inside the repository", "results": []}
    try:
        rep = audit_script(p.resolve(), alpha=cfg.alpha, timeout=cfg.timeout)
    except Exception as exc:  # noqa: BLE001
        return {"path": rel, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:2000], "results": []}
    d = rep.to_dict()
    d["source"] = rel
    results = [{"location": r.location[:200], "test": r.test_label[:100], "target": (r.target or "")[:200],
                "reported_p": r.reported_p, "status": r.status.value, "verdict": r.verdict[:1500]}
               for r in rep.results[:200]]
    return {"path": rel, "status": rep.status.value, "error": (rep.script_error or "")[-2000:] or None,
            "counts": {k: v for k, v in rep.counts().items()}, "results": results, "report": agent_report(d)}


def run_ci(
    repo: str | Path = ".",
    base: str | None = None,
    head: str | None = None,
    files: list[str] | None = None,
    all_files: bool = False,
    config: Config | None = None,
    config_path: str | None = None,
    pr_number: int | None = None,
    repo_slug: str | None = None,
    log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    """Run the configured checks and return a result document (validate it before rendering)."""
    root = Path(repo)
    cfg = config or load_config(root, path=config_path, rev=base)
    notes: list[str] = []
    tracked = tracked_files(root)
    if files is not None:
        norm = (f.replace("\\", "/") for f in files)
        changed = sorted({f[2:] if f.startswith("./") else f for f in norm})
    elif base and not all_files:
        changed = changed_files(root, base, head or "HEAD")
    else:
        changed = tracked
    pdf_ok = shutil.which("pdftotext") is not None
    report_ext = REPORT_EXT + ((".pdf",) if pdf_ok else ())

    def pick(paths: list[str], exts: tuple[str, ...], globs: list[str], what: str) -> list[str]:
        bad = [p for p in paths if p.startswith("/") or ".." in p.split("/") or "\\" in p or len(p) > 300]
        if bad:
            notes.append(f"{what}: ignored {len(bad)} path(s) outside the repository")
        sel = [p for p in paths if p not in bad and p.lower().endswith(exts) and _match(p, globs)
               and not p.startswith(".stataudit-tool/")]
        if len(sel) > cfg.max_files:
            notes.append(f"{what}: only the first {cfg.max_files} of {len(sel)} files were checked (max_files)")
            sel = sel[: cfg.max_files]
        return sel

    lint_t = pick(changed, LINT_EXT, cfg.lint, "lint")
    report_t = pick(changed, report_ext, cfg.reports, "check-report")
    run_t = pick(tracked, LINT_EXT, cfg.run, "run") if cfg.run else []
    lint, reports, runs = [], [], []
    for rel in lint_t:
        log(f"lint {rel}")
        lint.append(_lint_entry(root, rel, cfg))
    for rel in report_t:
        log(f"check-report {rel}")
        reports.append(_report_entry(root, rel, cfg))
    for rel in run_t:
        log(f"run {rel}")
        runs.append(_run_entry(root, rel, cfg))
    doc = new_result(
        repo=repo_slug, pr_number=pr_number, head=head if head and _SHA_RE.match(head) else None,
        base=base if base and _SHA_RE.match(base) else None,
        alpha=cfg.alpha, fail_on=cfg.fail_on, config_source=cfg.source[:200], lint=lint, runs=runs, reports=reports,
        agent=None, notes=notes,
    )
    doc["status"] = overall_status(doc)
    return doc


def exit_code(status: str, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    if fail_on == "warn":
        return 1 if status in ("warn", "fail") else 0
    return 1 if status == "fail" else 0


def write_outputs(result: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Validate ``result``, then write ``result.json`` and ``card.md`` into ``out_dir``."""
    from .card import render_card
    from .ci_result import validate_result

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    valid = validate_result(result)
    rj = d / "result.json"
    rj.write_text(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    card = d / "card.md"
    card.write_text(render_card(valid), encoding="utf-8")
    if result.get("pr_number"):
        (d / "pr_number").write_text(f"{result['pr_number']}\n", encoding="utf-8")
    if result.get("head"):
        (d / "head_sha").write_text(f"{result['head']}\n", encoding="utf-8")
    return rj, card


def write_summary(json_path: str | Path, out_path: str | Path | None = None) -> str | None:
    """Validate result.json, render the card and append it to ``out_path`` / ``$GITHUB_STEP_SUMMARY`` (or return it)."""
    from .card import render_card
    from .ci_result import MAX_RESULT_BYTES, validate_result

    p = Path(json_path)
    if p.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError("result.json is too large")
    card = render_card(validate_result(json.loads(p.read_text(encoding="utf-8"))))
    dest = out_path or os.environ.get("GITHUB_STEP_SUMMARY")
    if not dest:
        print(card)
        return None
    with open(dest, "a", encoding="utf-8") as fh:
        fh.write(card + "\n")
    return str(dest)
