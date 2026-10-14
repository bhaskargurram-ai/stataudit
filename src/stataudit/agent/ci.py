"""The agent's two CI steps: ``stataudit agent propose`` and ``stataudit agent verify``.

* ``propose`` runs in the job that holds the LLM key. It reads the pull request
  checkout **as data** (the analysis source, CSV headers) and the validated
  result.json of the unprivileged check job; it never imports or executes
  anything from the pull request. Output: one candidate directory per analysis.
* ``verify`` runs in a job **without secrets**. It re-runs each original
  analysis and its candidate under the tracer (scrubbed environment, timeout)
  and merges the outcome into result.json, from which the comment job renders
  the card.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from ..ci_result import MAX_RESULT_BYTES, validate_result
from ..config import Config, load_config
from .fix import FixError, FixResult, propose_only, verify_only
from .llm import ToolLLM

__all__ = ["agent_propose", "agent_verify", "PROPOSALS"]

PROPOSALS = "proposals.json"
_NEEDS_FIX = ("fragile", "does_not_hold")


def _load_result(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.stat().st_size > MAX_RESULT_BYTES:
        raise FixError("result.json is too large")
    return validate_result(json.loads(p.read_text(encoding="utf-8")))


def _inside(root: Path, rel: str) -> Path | None:
    if not rel or rel.startswith("/") or ".." in rel.split("/"):
        return None
    p = root / rel
    try:
        real, root_r = p.resolve(), root.resolve()
    except OSError:
        return None
    if not real.is_file() or root_r not in real.parents:
        return None
    return p


def agent_propose(result_path: str | Path, repo_dir: str | Path, out_dir: str | Path,
                  llm_factory: Callable[[], ToolLLM], base: str | None = None, config: Config | None = None,
                  log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """Propose a fix for each analysis that is fragile or does not hold. Executes nothing from the PR."""
    root = Path(repo_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = _load_result(result_path)
    cfg = config or load_config(root, rev=base)
    summary: dict[str, Any] = {"schema": 1, "proposals": [], "notes": []}
    if not cfg.agent_enabled:
        summary["notes"].append("the agent is disabled ([agent] enabled = false in stataudit.toml at the base commit)")
    else:
        todo = [r for r in result["runs"] if r["status"] in _NEEDS_FIX and r.get("report")][: cfg.agent_max_fixes]
        if not todo:
            summary["notes"].append("no analysis needs a fix")
        for i, run in enumerate(todo):
            entry: dict[str, Any] = {"index": i, "path": run["path"], "status": "error", "notes": []}
            path = _inside(root, run["path"])
            if path is None:
                entry["notes"].append("not a regular file inside the checkout")
            else:
                log(f"agent: proposing a fix for {run['path']}")
                try:
                    fr = propose_only(path, run["report"], llm_factory, out / str(i), share_data=cfg.agent_share_data,
                                      max_tool_calls=cfg.agent_max_tool_calls, root=root, log=log)
                    entry["status"] = fr.status
                    entry["notes"] = fr.notes[:5]
                    entry["model"] = fr.model
                    entry["usage"] = fr.usage.to_dict()
                except FixError as exc:
                    entry["notes"].append(str(exc))
            summary["proposals"].append(entry)
    (out / PROPOSALS).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _fix_entry(rel: str, fr: FixResult) -> dict[str, Any]:
    v = fr.verification
    rows = []
    for r in (v.rows if v else [])[:50]:
        rows.append({k: (str(r[k])[:200] if r.get(k) is not None else None)
                     for k in ("location", "before_test", "before_status", "after_location", "after_test", "after_status",
                               "outcome")})
    return {
        "path": rel,
        "status": fr.status,
        "rationale": (fr.candidate.rationale if fr.candidate else "")[:2000],
        "diff": fr.diff[:20000],
        "before_status": fr.before_status[:20],
        "after_status": (v.after_status if v else "")[:20],
        "rows": rows,
        "reasons": [s[:500] for s in (v.reasons if v else fr.notes)][:20],
        "model": fr.model[:100],
    }


def agent_verify(result_path: str | Path, candidates_dir: str | Path, repo_dir: str | Path = ".",
                 timeout: float | None = None, log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """Verify every candidate (runs the analyses; no LLM, no secrets needed) and merge into the result."""
    root = Path(repo_dir)
    cdir = Path(candidates_dir)
    result = _load_result(result_path)
    raw = json.loads(Path(result_path).read_text(encoding="utf-8"))
    run_paths = {r["path"] for r in result["runs"]}
    agent: dict[str, Any] = {"status": "skipped", "notes": [], "fixes": []}
    props_path = cdir / PROPOSALS
    props: dict[str, Any] = {}
    if props_path.is_file() and props_path.stat().st_size < 1_000_000:
        try:
            props = json.loads(props_path.read_text(encoding="utf-8"))
        except ValueError:
            agent["notes"].append("proposals.json is not valid JSON")
    if not isinstance(props, dict):
        props = {}
    for n in (props.get("notes") or [])[:5]:
        if isinstance(n, str):
            agent["notes"].append(n[:300])
    for p in (props.get("proposals") or [])[:10]:
        if not isinstance(p, dict) or not isinstance(p.get("path"), str) or not isinstance(p.get("index"), int):
            continue
        rel = p["path"]
        if rel not in run_paths:
            agent["notes"].append(f"ignored a proposal for {rel[:100]!r}: not an analysis from this run")
            continue
        if p.get("status") != "proposed":
            fx = {"path": rel, "status": p.get("status") if p.get("status") in ("no_submission", "nothing_to_fix", "error")
                  else "error", "rationale": "", "diff": "", "before_status": "", "after_status": "", "rows": [],
                  "reasons": [str(x)[:500] for x in (p.get("notes") or [])[:5]], "model": str(p.get("model", ""))[:100]}
            agent["fixes"].append(fx)
            continue
        path = _inside(root, rel)
        cand = cdir / str(p["index"])
        if path is None or not re.fullmatch(r"\d{1,2}", str(p["index"])) or not cand.is_dir():
            agent["notes"].append(f"candidate for {rel[:100]!r} is missing")
            continue
        log(f"agent: verifying the candidate for {rel}")
        try:
            fr = verify_only(path, cand, alpha=result["alpha"], timeout=timeout or 300)
        except FixError as exc:
            agent["fixes"].append({"path": rel, "status": "error", "rationale": "", "diff": "", "before_status": "",
                                   "after_status": "", "rows": [], "reasons": [str(exc)[:500]], "model": ""})
            continue
        agent["fixes"].append(_fix_entry(rel, fr))
    statuses = [f["status"] for f in agent["fixes"]]
    for s in ("verified", "rejected", "no_submission", "nothing_to_fix", "error"):
        if s in statuses:
            agent["status"] = s
            break
    raw["agent"] = agent
    validate_result(raw)  # fail loudly here rather than in the comment job
    return raw
