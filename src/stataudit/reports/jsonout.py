"""JSON output."""

from __future__ import annotations

import json

from ..models import Report


def render_json(report: Report, indent: int = 2) -> str:
    """Serialize a report to JSON (NaN/inf become null)."""
    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)
