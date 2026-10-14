"""Guard rails for running code an LLM wrote: a static denylist and a scrubbed environment.

Neither is a sandbox. The denylist is a cheap filter that refuses obviously
off-task code *before* it runs (process, network, file-system and dynamic
import/eval machinery that a statistical analysis does not need). The real
protection is where the code runs: in a subprocess with a timeout, with
secret-looking environment variables removed, and, in CI, in a job that holds
no secrets at all (see docs/SECURITY_MODEL.md).

Constructs that the *original* script already contains are not counted
against the candidate: an analysis that does ``import os`` to build a data
path may keep doing so. Only what the model added is judged.
"""

from __future__ import annotations

import ast
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "DENY_MODULES",
    "DENY_CALLS",
    "DENY_ATTRS",
    "MAX_CODE_CHARS",
    "StaticCheck",
    "static_check",
    "denylist_hits",
    "scrub_env",
    "is_sensitive_env_name",
]

MAX_CODE_CHARS = 200_000

#: Importing any of these (or a submodule) is refused.
DENY_MODULES = frozenset({
    "os", "subprocess", "socket", "shutil", "ctypes", "importlib", "multiprocessing", "pty", "signal", "urllib",
    "http", "requests", "httpx", "aiohttp", "urllib3", "ftplib", "smtplib", "telnetlib", "poplib", "imaplib",
    "webbrowser", "socketserver", "xmlrpc", "pickle", "marshal", "builtins", "asyncio", "threading", "_thread",
    "code", "codeop", "runpy", "sysconfig", "zipimport",
})
#: Calling any of these names is refused.
DENY_CALLS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint", "input", "globals", "locals", "vars",
                        "getattr", "setattr", "delattr", "memoryview"})
#: Attribute names that are refused anywhere (escape hatches and file-system writes).
DENY_ATTRS = frozenset({
    "__builtins__", "__subclasses__", "__globals__", "__code__", "__loader__", "__import__", "__dict__",
    "__getattribute__", "__reduce__", "__reduce_ex__", "system", "popen", "write_text", "write_bytes", "unlink",
    "rmdir", "rmtree", "chmod", "chown", "symlink_to", "hardlink_to", "touch",
})
_WRITE_MODES = set("wax+")

#: Substrings that mark an environment variable as sensitive (matched case-insensitively on the name).
SENSITIVE_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "COOKIE", "SESSION")
#: Environment variables always removed, whatever their name.
ALWAYS_DROP = ("STATAUDIT_LLM_", "ACTIONS_RUNTIME_", "ACTIONS_ID_TOKEN_", "ACTIONS_CACHE_URL", "GH_", "GITHUB_TOKEN")


def is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(m in upper for m in SENSITIVE_MARKERS) or upper.startswith(ALWAYS_DROP)


def scrub_env(environ: Mapping[str, str] | None = None, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``environ`` without secret-looking variables (``extra`` is added afterwards)."""
    src = os.environ if environ is None else environ
    out = {k: v for k, v in src.items() if not is_sensitive_env_name(k)}
    if extra:
        out.update(extra)
    return out


def _mode_writes(call: ast.Call) -> bool:
    mode: ast.expr | None = call.args[1] if len(call.args) >= 2 else None
    for k in call.keywords:
        if k.arg == "mode":
            mode = k.value
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return bool(set(mode.value) & _WRITE_MODES)
    return True  # a computed mode could be anything


def denylist_hits(code: str) -> Counter[str]:
    """Denylisted constructs in ``code`` (a multiset of short descriptions). Raises SyntaxError."""
    tree = ast.parse(code)
    hits: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in DENY_MODULES or a.name in DENY_MODULES:
                    hits[f"import {a.name}"] += 1
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                hits["relative import"] += 1
            elif mod.split(".")[0] in DENY_MODULES:
                hits[f"from {mod} import ..."] += 1
            for a in node.names:
                if a.name in DENY_CALLS or a.name in ("system", "popen"):
                    hits[f"from {mod} import {a.name}"] += 1
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if isinstance(f, ast.Name) and name in DENY_CALLS:
                hits[f"call {name}()"] += 1
            if name == "open" and _mode_writes(node):
                hits["open() for writing"] += 1
        elif isinstance(node, ast.Attribute) and node.attr in DENY_ATTRS:
            hits[f"attribute .{node.attr}"] += 1
        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "__loader__", "__spec__"):
            hits[f"name {node.id}"] += 1
    return hits


@dataclass
class StaticCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)


def static_check(code: str, original: str | None = None) -> StaticCheck:
    """Refuse candidate code that does not parse, is too large, or adds denylisted constructs.

    ``original`` is the code the candidate replaces; constructs it already
    contains (same description, up to the same count) are allowed.
    """
    errors: list[str] = []
    if not isinstance(code, str) or not code.strip():
        return StaticCheck(False, ["the code is empty"])
    if len(code) > MAX_CODE_CHARS:
        return StaticCheck(False, [f"the code is longer than {MAX_CODE_CHARS} characters"])
    try:
        hits = denylist_hits(code)
    except SyntaxError as exc:
        return StaticCheck(False, [f"the code does not parse: {exc.msg} (line {exc.lineno})"])
    allowed: Counter[str] = Counter()
    if original:
        try:
            allowed = denylist_hits(original)
        except SyntaxError:
            allowed = Counter()
    new = hits - allowed
    for what, n in sorted(new.items()):
        errors.append(f"not allowed in a proposed analysis: {what}" + (f" (x{n})" if n > 1 else ""))
    return StaticCheck(not errors, errors)
