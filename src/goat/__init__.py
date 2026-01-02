"""
GOAT Attention package.

Public API:
- GoatAttention
"""

from __future__ import annotations

from ._version import __version__

from typing import TYPE_CHECKING, Any

__all__ = ["GoatAttention", "__version__"]

if TYPE_CHECKING:  # pragma: no cover
    from .attention import GoatAttention


def __getattr__(name: str) -> Any:
    """
    Lazy import the torch-backed implementation so `import goat` can succeed even
    in environments where PyTorch is not correctly installed/configured.
    """
    if name != "GoatAttention":
        raise AttributeError(name)

    try:
        from .attention import GoatAttention as _GoatAttention
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "Failed to import PyTorch-backed GOAT attention.\n"
            "This usually means PyTorch is missing or misconfigured.\n"
            "If you're on CPU-only, install a CPU build of torch (see pytorch.org).\n"
        ) from e

    return _GoatAttention



