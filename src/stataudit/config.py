"""``stataudit.toml`` (or ``[tool.stataudit]`` in pyproject.toml): settings for ``stataudit ci`` and the agent.

Every key is optional::

    alpha = 0.05
    fail_on = "fail"                    # never | warn | fail   (exit code of `stataudit ci`)
    run = ["analysis/**/*.py"]          # executed under the tracer in CI (opt-in: this runs code)
    reports = ["reports/**/*.md"]       # checked with check-report when changed
    lint = ["**/*.py", "**/*.ipynb"]    # linted when changed
    lint_ignore = []                    # rule ids, e.g. ["SA107"]
    timeout = 300                       # seconds per executed analysis
    max_files = 50                      # per category, per run

    [agent]                             # `stataudit fix` in CI (stataudit-agent.yml)
    enabled = false
    max_tool_calls = 12
    max_fixes = 3                       # analyses per PR
    share_data = false                  # never send raw rows to the LLM unless true

    [llm]                               # fallbacks; the key only comes from STATAUDIT_LLM_API_KEY
    provider = ""                       # anthropic | openai
    model = ""
    base_url = ""

In CI the configuration is read from the pull request's **base** commit, so a
pull request cannot weaken its own check.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

__all__ = ["Config", "ConfigError", "load_config", "parse_config", "glob_match", "DEFAULT_REPORT_GLOBS"]

FAIL_ON = ("never", "warn", "fail")
DEFAULT_LINT_GLOBS = ["**/*.py", "**/*.ipynb"]
DEFAULT_REPORT_GLOBS = ["**/*.md", "**/*.txt", "**/*.html", "**/*.htm"]
CONFIG_NAMES = ("stataudit.toml", ".stataudit.toml")


class ConfigError(ValueError):
    """Invalid configuration."""


@dataclass
class Config:
    alpha: float = 0.05
    fail_on: str = "fail"
    run: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=lambda: list(DEFAULT_REPORT_GLOBS))
    lint: list[str] = field(default_factory=lambda: list(DEFAULT_LINT_GLOBS))
    lint_ignore: list[str] = field(default_factory=list)
    timeout: float = 300.0
    max_files: int = 50
    agent_enabled: bool = False
    agent_max_tool_calls: int = 12
    agent_max_fixes: int = 3
    agent_share_data: bool = False
    llm: dict[str, str] = field(default_factory=dict)
    source: str = "defaults"


def _strs(v: Any, key: str) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ConfigError(f"{key} must be a list of strings")
    return [x.strip() for x in v if x.strip()]


def parse_config(data: dict[str, Any], source: str = "") -> Config:
    if not isinstance(data, dict):
        raise ConfigError("configuration must be a table")
    c = Config(source=source or "defaults")
    known = {"alpha", "fail_on", "fail-on", "run", "reports", "lint", "lint_ignore", "timeout", "max_files", "agent", "llm"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in {source or 'config'}: {', '.join(unknown)}")
    if "alpha" in data:
        a = data["alpha"]
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 < float(a) < 1:
            raise ConfigError("alpha must be a number between 0 and 1")
        c.alpha = float(a)
    fo = data.get("fail_on", data.get("fail-on"))
    if fo is not None:
        if fo not in FAIL_ON:
            raise ConfigError(f"fail_on must be one of {', '.join(FAIL_ON)}")
        c.fail_on = fo
    if "run" in data:
        c.run = _strs(data["run"], "run")
    if "reports" in data:
        c.reports = _strs(data["reports"], "reports")
    if "lint" in data:
        c.lint = _strs(data["lint"], "lint")
    if "lint_ignore" in data:
        c.lint_ignore = [s.upper() for s in _strs(data["lint_ignore"], "lint_ignore")]
    if "timeout" in data:
        t = data["timeout"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 1 <= float(t) <= 7200:
            raise ConfigError("timeout must be between 1 and 7200 seconds")
        c.timeout = float(t)
    if "max_files" in data:
        m = data["max_files"]
        if isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 1000:
            raise ConfigError("max_files must be an integer between 1 and 1000")
        c.max_files = m
    agent = data.get("agent") or {}
    if not isinstance(agent, dict):
        raise ConfigError("[agent] must be a table")
    for k, v in agent.items():
        if k == "enabled":
            c.agent_enabled = bool(v)
        elif k == "share_data":
            c.agent_share_data = bool(v)
        elif k in ("max_tool_calls", "max_fixes"):
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= (50 if k == "max_tool_calls" else 20):
                raise ConfigError(f"[agent] {k} is out of range")
            setattr(c, f"agent_{k}", v)
        else:
            raise ConfigError(f"unknown key [agent] {k}")
    llm = data.get("llm") or {}
    if not isinstance(llm, dict):
        raise ConfigError("[llm] must be a table")
    for k, v in llm.items():
        if k not in ("provider", "model", "base_url"):
            raise ConfigError(f"unknown key [llm] {k} (the API key is only read from STATAUDIT_LLM_API_KEY)")
        if not isinstance(v, str):
            raise ConfigError(f"[llm] {k} must be a string")
        c.llm[k] = v
    return c


def _from_text(text: str, name: str, pyproject: bool | None = None) -> Config:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{name}: {exc}") from exc
    # Decide by the file name, not the label: labels for files read from git look like "pyproject.toml@<sha>".
    if pyproject is None:
        pyproject = Path(name).name == "pyproject.toml"
    if pyproject:
        data = (data.get("tool") or {}).get("stataudit") or {}
        if not data:
            return Config(source="defaults")
    return parse_config(data, name)


def _git_show(repo: Path, rev: str, path: str) -> str | None:
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._/-]{0,99}", rev) or ".." in rev:
        raise ConfigError(f"invalid revision {rev!r}")
    try:
        proc = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, timeout=30,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def load_config(repo: str | Path = ".", path: str | None = None, rev: str | None = None) -> Config:
    """Load ``path``, else ``stataudit.toml`` / ``.stataudit.toml`` / ``[tool.stataudit]`` in pyproject.toml.

    With ``rev``, files are read from that git revision (e.g. the PR's base commit) instead of the working tree.
    """
    root = Path(repo)
    names = [path] if path else [*CONFIG_NAMES, "pyproject.toml"]
    for name in names:
        assert name is not None
        if rev:
            text = _git_show(root, rev, name)
            label = f"{name}@{rev[:12]}"
        else:
            p = Path(name) if Path(name).is_absolute() else root / name
            text = p.read_text(encoding="utf-8") if p.is_file() else None
            label = name
        if text is None:
            if path:
                raise ConfigError(f"configuration file not found: {name}" + (f" at {rev}" if rev else ""))
            continue
        cfg = _from_text(text, label, pyproject=Path(name).name == "pyproject.toml")
        if cfg.source != "defaults" or name != "pyproject.toml":
            return cfg
    return Config()


def glob_match(path: str, pattern: str) -> bool:
    """Match a POSIX relative path against a glob where ``**`` spans directories."""
    return re.fullmatch(_glob_re(pattern), path) is not None


def _glob_re(pattern: str) -> str:
    pattern = pattern.strip().lstrip("./") if pattern.startswith("./") else pattern.strip()
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif ch == "*":
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return "".join(out)
