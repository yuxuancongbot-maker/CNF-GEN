"""Regression tests for the centralized path configuration."""

from __future__ import annotations

import unittest
from pathlib import Path

from learned_basis_hybrid_cnf import config


class TestConfig(unittest.TestCase):
    def test_output_directory_always_resolves(self):
        self.assertIsInstance(config.OUT, Path)
        self.assertTrue(str(config.OUT))

    def test_resolved_contains_all_resource_keys(self):
        r = config.resolved()
        for key in [
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
        ]:
            self.assertIn(key, r)

    def test_missing_wan_is_superset_of_missing_cpu(self):
        miss_cpu = set(config.missing(require_wan=False))
        miss_wan = set(config.missing(require_wan=True))
        self.assertTrue(miss_cpu.issubset(miss_wan))

    def test_wan_only_resources_are_excluded_without_wan(self):
        miss_cpu = config.missing(require_wan=False)
        for name in ("WAN_MODEL", "WAN_LORA", "VIDEOX", "BASE_1_3B"):
            self.assertNotIn(name, miss_cpu)

    def test_assert_configured_raises_when_resources_unset(self):
        if config.missing(require_wan=True):
            with self.assertRaises(config.ConfigError):
                config.assert_configured(require_wan=True)


if __name__ == "__main__":
    unittest.main()
