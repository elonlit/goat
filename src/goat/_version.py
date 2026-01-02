from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _safe_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        # Editable installs or running from source without installation.
        return "0.0.0"


__version__ = _safe_version("goat-attention")


