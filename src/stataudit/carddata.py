"""Machine-readable data block at the end of the stataudit card.

The card ends with one HTML comment::

    <!-- stataudit:data {"v":1,"status":"fail","runs":[...],...} -->

GitHub does not render it, but bots, dashboards and end-to-end tests that
read the comment through the API can get the statuses and counts without
scraping Markdown.

Safety (same design as proofcheck's card data block):

* built only from fields of a *validated* result (:func:`stataudit.ci_result.validate_result`);
* compact, ASCII-only JSON in which every ``-``, ``<``, ``>``, ``&`` and ``@``
  is written as a ``\\u`` escape, so nothing inside it can end the comment
  (``-->``, ``--!>``), open markup or carry a live @mention; JSON parsers read
  the escapes back transparently;
* paths are cleaned and cut to 80 characters, lists are capped, and the block
  never exceeds :data:`MAX_DATA_CHARS` (per-file lists are dropped first);
* :func:`parse_card_data` only accepts the block when it is the *last* thing
  in the card, so a look-alike placed earlier by PR-controlled text is ignored.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["DATA_PREFIX", "DATA_SUFFIX", "MAX_DATA_CHARS", "build_card_data", "encode_card_data", "card_data_block",
           "parse_card_data"]

DATA_PREFIX = "<!-- stataudit:data "
DATA_SUFFIX = " -->"
DATA_VERSION = 1
MAX_DATA_CHARS = 4000
MAX_ITEMS = 20
MAX_PATH = 80

_ESCAPED = "-<>&@"
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f   ‪-‮⁦-⁩]")
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_TRAILING_RE = re.compile(r"(?:\A|\n)<!-- stataudit:data (\{[^\n]*\}) -->\s*\Z")


def _text(value: Any, limit: int) -> str:
    s = _CONTROL_RE.sub(" ", "" if value is None else str(value))
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def build_card_data(result: dict[str, Any], trusted_warnings: int = 0) -> dict[str, Any]:
    data: dict[str, Any] = {
        "v": DATA_VERSION,
        "tool": _text(result.get("tool_version"), 20),
        "pr": result.get("pr_number") if _count(result.get("pr_number")) else None,
        "head": result.get("head") if isinstance(result.get("head"), str) and _SHA_RE.match(result["head"]) else None,
        "status": _text(result.get("status"), 10),
        # as a string: a float such as 1e-05 would put a "-" outside a JSON string
        "alpha": f"{result['alpha']:g}" if isinstance(result.get("alpha"), (int, float)) else None,
    }
    runs = result.get("runs") or []
    data["runs"] = [{"path": _text(r.get("path"), MAX_PATH), "status": _text(r.get("status"), 20),
                     **{k: _count((r.get("counts") or {}).get(k)) for k in ("holds", "fragile", "does_not_hold")}}
                    for r in runs[:MAX_ITEMS]]
    reports = result.get("reports") or []
    data["reports"] = [{"path": _text(r.get("path"), MAX_PATH), "status": _text(r.get("status"), 20),
                        **{k: _count((r.get("counts") or {}).get(k)) for k in ("consistent", "inconsistent", "gross", "impossible")}}
                       for r in reports[:MAX_ITEMS]]
    lint = result.get("lint") or []
    findings = [f for x in lint for f in (x.get("findings") or [])]
    data["lint"] = {"files": len(lint), "warn": sum(f.get("severity") == "warn" for f in findings),
                    "info": sum(f.get("severity") == "info" for f in findings),
                    "rules": sorted({_text(f.get("rule"), 8) for f in findings})}
    agent = result.get("agent")
    if agent:
        fixes = agent.get("fixes") or []
        data["agent"] = {"status": _text(agent.get("status"), 20), "fixes": len(fixes),
                         "verified": sum(f.get("status") == "verified" for f in fixes)}
    else:
        data["agent"] = None
    data["warnings"] = _count(trusted_warnings)
    return data


def encode_card_data(data: dict[str, Any]) -> str:
    """Compact JSON that is safe inside an HTML comment (no ``-``, ``<``, ``>``, ``&``, ``@``)."""
    s = json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    # Only non-negative integers occur outside strings, so every "-" is inside a string.
    for ch in _ESCAPED:
        s = s.replace(ch, f"\\u{ord(ch):04x}")
    return s


def card_data_block(result: dict[str, Any], trusted_warnings: int = 0) -> str:
    data = build_card_data(result, trusted_warnings)
    budget = MAX_DATA_CHARS - len(DATA_PREFIX) - len(DATA_SUFFIX)
    payload = encode_card_data(data)
    for key in ("runs", "reports"):
        while len(payload) > budget and data[key]:
            data[key].pop()
            data[f"{key}_truncated"] = True
            payload = encode_card_data(data)
    if len(payload) > budget:  # pragma: no cover - fixed-size fields only
        data = {k: data[k] for k in ("v", "head", "status")}
        payload = encode_card_data(data)
    return DATA_PREFIX + payload + DATA_SUFFIX


def parse_card_data(markdown: Any) -> dict[str, Any] | None:
    """The data block of a card, or None (only a block at the very end counts)."""
    if not isinstance(markdown, str):
        return None
    m = _TRAILING_RE.search(markdown[-(MAX_DATA_CHARS + 64):])
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("v"), int):
        return None
    return data
