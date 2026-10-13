"""stataudit — a second-opinion statistician for AI-generated analyses."""

from __future__ import annotations

__version__ = "0.1.0"

from .trace import recording, trace_script

__all__ = ["__version__", "recording", "trace_script"]
