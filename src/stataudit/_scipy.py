"""Access to the *original* scipy.stats functions.

While a trace is active, ``scipy.stats`` attributes are replaced by recording
wrappers. The auditors must never record their own calls, so they always go
through :func:`orig`, which unwraps a patched function.
"""

from __future__ import annotations

from typing import Any, Callable

import scipy.stats as _stats

ORIGINAL_ATTR = "__stataudit_original__"


def orig(name: str) -> Callable[..., Any]:
    """Return the unpatched ``scipy.stats.<name>`` function."""
    func = getattr(_stats, name)
    return getattr(func, ORIGINAL_ATTR, func)


class _OriginalStats:
    """Attribute proxy: ``ostats.ttest_ind`` is always the unpatched function."""

    def __getattr__(self, name: str) -> Any:
        value = getattr(_stats, name)
        return getattr(value, ORIGINAL_ATTR, value)


#: Use ``from stataudit._scipy import ostats`` inside auditors instead of ``scipy.stats``.
ostats = _OriginalStats()
