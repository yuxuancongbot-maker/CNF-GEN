"""Regression tests for the collaboration boundaries and experiment contracts."""

from __future__ import annotations

import copy
import json
import unittest

from jsonschema import Draft202012Validator

from scripts.check_repository import ROOT, check_imports, check_modules, check_repository


class RepositoryTests(unittest.TestCase):
    def test_repository_structure(self):
        errors, modules = check_repository()
        self.assertEqual(errors, [])
        self.assertEqual(len(modules), 7)

    def test_module_manifests_follow_schema(self):
        schema = json.loads((ROOT / "schemas/module.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        manifests = [ROOT / "module.json", *sorted((ROOT / "tasks").glob("*/module.json"))]
        for path in manifests:
            with self.subTest(path=path):
                validator.validate(json.loads(path.read_text()))

    def test_task_cannot_depend_on_other_task_code(self):
        _, modules = check_repository()
        changed = copy.deepcopy(modules)
        task = next(item for item in changed if item["id"] == "video_generation")
        task["dependencies"].append("action_generation")
        self.assertTrue(any("code dependencies" in error for error in check_modules(changed)))

    def test_duplicate_work_item_is_rejected(self):
        _, modules = check_repository()
        changed = copy.deepcopy(modules)
        changed[-1]["work_items"].append("CORE-001")
        self.assertIn("duplicate work item CORE-001", check_modules(changed))

    def test_core_cannot_import_task(self):
        errors = check_imports(
            "from gcnf_action_generation.model import Model", "core", {"gcnf_action_generation"}
        )
        self.assertTrue(errors)

    def test_task_can_import_core_and_own_package(self):
        source = "import gcnf\nfrom gcnf_action_generation import model\n"
        self.assertEqual(check_imports(source, "action_generation", {"gcnf_action_generation"}), [])

    def test_video_cannot_import_action_internals(self):
        errors = check_imports(
            "import gcnf_action_generation.model",
            "video_generation",
            {"gcnf_action_generation", "gcnf_video_generation"},
        )
        self.assertTrue(errors)

    def test_run_template_is_explicitly_incomplete(self):
        schema = json.loads((ROOT / "schemas/run.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        sample = json.loads((ROOT / "schemas/examples/run.json").read_text())
        validator = Draft202012Validator(schema)
        validator.validate(sample)
        self.assertEqual(sample["status"], "incomplete")
        self.assertEqual(sample["metrics"], [])
        sample["status"] = "succeeded"
        self.assertFalse(validator.is_valid(sample))

    def test_metric_requires_units_and_evaluation_stage(self):
        schema = json.loads((ROOT / "schemas/run.schema.json").read_text())
        sample = json.loads((ROOT / "schemas/examples/run.json").read_text())
        sample["metrics"] = [{"name": "constraint_residual", "value": 0}]
        self.assertFalse(Draft202012Validator(schema).is_valid(sample))


if __name__ == "__main__":
    unittest.main()
