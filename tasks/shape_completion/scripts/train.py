#!/usr/bin/env python3
"""Thin entry point for shape-completion preflight / training.

All business logic lives in ``gcnf_shape_completion.cli``.  Run from any
working directory; the task package is located relative to this script unless
it has been installed with ``pip install -e tasks/shape_completion``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcnf_shape_completion.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
