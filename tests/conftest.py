from __future__ import annotations

import sys
from pathlib import Path


# Allow `pytest` runs from a fresh checkout without requiring an editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


