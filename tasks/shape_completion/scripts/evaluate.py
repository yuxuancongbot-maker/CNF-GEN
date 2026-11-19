#!/usr/bin/env python3
"""Thin entry point for the six-method held-out shape-completion evaluator.

All business logic lives in ``gcnf_shape_completion.evaluate``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcnf_shape_completion.evaluate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
