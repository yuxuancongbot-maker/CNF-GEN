"""Centralized configuration for the learned-basis Hybrid-CNF v2 experiment.

This module resolves every external resource (train/eval datasets, v2 eval
references, the Wan 1.3B model and Control LoRA, the VideoX-Fun checkout, and
the frozen RGB evaluator) plus the output directory. Precedence, highest first:

1. environment variables prefixed ``CNF_V2_``;
2. a JSON config file (path from ``CNF_V2_CONFIG``, otherwise
   ``configs/paths.json`` next to this experiment);
3. built-in defaults (the output directory only; external resources have no
   machine-specific default and resolve to ``None``).

There are no hardcoded absolute paths to the original research workspace, so a
collaborator can point these at their own copies. Use :func:`assert_configured`
(which the entry points call) to fail with a clear message listing what is
missing. Importing this module never requires the external resources to exist,
so pure CPU numerics tests can run without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV_PREFIX = "CNF_V2_"
CONFIG_ENV = "CNF_V2_CONFIG"

PKG = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PKG.parent.parent

_OUT_DEFAULT = "outputs/learned_basis_hybrid_cnf_v2"


class ConfigError(RuntimeError):
    """Raised when a required external resource is not configured."""


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value if value else None


def _config_file() -> dict[str, Any]:
    path = os.environ.get(CONFIG_ENV)
    candidates = [Path(path) if path else None, EXPERIMENT_ROOT / "configs" / "paths.json"]
    for cand in candidates:
        if cand is not None and cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {}


_CFG = _config_file()


def _get(name: str) -> str | None:
    value = _env(name)
    if value is not None:
        return value
    value = _CFG.get(name)
    if value is not None:
        return str(value)
    return None


def _path(name: str, default: str | None = None) -> Path | None:
    value = _get(name)
    if value is None:
        return Path(default).expanduser() if default is not None else None
    return Path(value).expanduser()


# Output directory (always has a default, relative to the current working dir).
OUT = Path(_get("OUT") or _OUT_DEFAULT).expanduser()

# Report output path.
DOC_REPORT = _path("REPORT", default=str(OUT / "learned_basis_hybrid_cnf_report.md"))

# External resources. ``None`` means "not configured yet".
TRAIN_DS = _path("TRAIN_DS")
EVAL_DS = _path("EVAL_DS")
V2_COND = _path("V2_COND")
WAN_MODEL = _path("WAN_MODEL")
WAN_LORA = _path("WAN_LORA")
VIDEOX = _path("VIDEOX")
EVALUATOR_PT = _path("EVALUATOR_PT")
EVALUATOR_SPEC = _path("EVALUATOR_SPEC")
EVALUATOR_CAL = _path("EVALUATOR_CAL")
BASE_1_3B = _path("BASE_1_3B")

# Derived paths used by the pipeline.
V2_REFS_DIR = V2_COND / "references" if V2_COND is not None else None
V2_TARGETS_DIR = V2_COND / "targets" if V2_COND is not None else None

# Used only by scripts/prepare_v2_references.py to reuse scenes 0..7 of the
# legacy 8-scene v2 references when regenerating the 16-scene v2 references.
OLD_V2_COND = _path("OLD_V2_COND")

_RESOURCE_NAMES = [
    "TRAIN_DS",
    "EVAL_DS",
    "V2_COND",
    "WAN_MODEL",
    "WAN_LORA",
    "VIDEOX",
    "EVALUATOR_PT",
    "EVALUATOR_SPEC",
    "EVALUATOR_CAL",
    "BASE_1_3B",
]


def resolved() -> dict[str, str | None]:
    """Snapshot of the resolved configuration, for protocol/report records."""
    out: dict[str, str | None] = {
        "OUT": str(OUT),
        "DOC_REPORT": str(DOC_REPORT),
        "PKG": str(PKG),
        "experiment_root": str(EXPERIMENT_ROOT),
    }
    for name in _RESOURCE_NAMES:
        out[name] = str(globals()[name]) if globals()[name] is not None else None
    out["V2_REFS_DIR"] = str(V2_REFS_DIR) if V2_REFS_DIR is not None else None
    out["V2_TARGETS_DIR"] = str(V2_TARGETS_DIR) if V2_TARGETS_DIR is not None else None
    out["OLD_V2_COND"] = str(OLD_V2_COND) if OLD_V2_COND is not None else None
    return out


def missing(require_wan: bool = True) -> list[str]:
    """Names of required resources that are not yet configured."""
    names = list(_RESOURCE_NAMES)
    if not require_wan:
        names = [n for n in names if n not in {"WAN_MODEL", "WAN_LORA", "VIDEOX", "BASE_1_3B"}]
    return [n for n in names if globals()[n] is None]


def assert_configured(require_wan: bool = True) -> None:
    """Raise :class:`ConfigError` listing any unconfigured required resources."""
    miss = missing(require_wan=require_wan)
    if miss:
        names = ", ".join(miss)
        raise ConfigError(
            "Missing configuration for required resources: " + names + ". "
            "Set them via environment variables (e.g. CNF_V2_" + miss[0] + "=...) "
            "or a JSON config file (CNF_V2_CONFIG=configs/paths.json). "
            "See experiments/learned_basis_hybrid_cnf/README.md."
        )


__all__ = [
    "ConfigError",
    "ENV_PREFIX",
    "PKG",
    "EXPERIMENT_ROOT",
    "OUT",
    "DOC_REPORT",
    "TRAIN_DS",
    "EVAL_DS",
    "V2_COND",
    "V2_REFS_DIR",
    "V2_TARGETS_DIR",
    "WAN_MODEL",
    "WAN_LORA",
    "VIDEOX",
    "EVALUATOR_PT",
    "EVALUATOR_SPEC",
    "EVALUATOR_CAL",
    "BASE_1_3B",
    "OLD_V2_COND",
    "resolved",
    "missing",
    "assert_configured",
]
