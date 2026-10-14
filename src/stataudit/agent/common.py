"""Helpers shared by the agent commands: untrusted-data wrapping, redaction, the LLM factory."""

from __future__ import annotations

import os
import re
import secrets
from typing import Any, Mapping, Sequence

from .llm import LLMConfigError, ToolLLM, client_from_env

__all__ = ["new_nonce", "untrusted", "redact", "llm_from_env", "clip"]

_SECRET_PATTERNS = [
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
]


def new_nonce() -> str:
    return secrets.token_hex(8)


def untrusted(source: str, text: str, nonce: str) -> str:
    """Delimit ``text`` as data the model must not take instructions from."""
    src = re.sub(r"[^A-Za-z0-9_.:/@ -]", "_", source)[:120]
    body = str(text).replace(f"UNTRUSTED_{nonce}", "UNTRUSTED_xxxx")
    return f"<<<UNTRUSTED_{nonce} source={src}>>>\n{body}\n<<<END_UNTRUSTED_{nonce}>>>"


def redact(text: str, secrets_: Sequence[str] = ()) -> str:
    """Remove known secret values and common token formats from ``text``."""
    for s in secrets_:
        if s and len(s) >= 6:
            text = text.replace(s, "[REDACTED]")
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n…[{len(text) - limit} more characters not shown]"


def llm_from_env(env: Mapping[str, str] | None = None, llm_cfg: Mapping[str, Any] | None = None,
                 transport: Any = None, max_output_tokens: int = 8192) -> ToolLLM:
    """Build the LLM client from ``STATAUDIT_LLM_*`` (``[llm]`` in stataudit.toml supplies fallbacks)."""
    cfg = dict(llm_cfg or {})
    return client_from_env(
        os.environ if env is None else env,
        provider=str(cfg.get("provider") or ""),
        model=str(cfg.get("model") or ""),
        base_url=str(cfg.get("base_url") or ""),
        max_output_tokens=max_output_tokens,
        transport=transport,
    )


__all__ += ["LLMConfigError"]
