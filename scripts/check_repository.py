"""Check module ownership metadata, package boundaries, and local documentation links."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ROOT = (
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "TASKS.md",
    "pyproject.toml",
    "schemas/module.schema.json",
    "schemas/run.schema.json",
    ".github/workflows/ci.yml",
)
REQUIRED_MODULE = ("README.md", "TASKS.md", "pyproject.toml")


def check_modules(modules: list[dict]) -> list[str]:
    """Reject ambiguous identifiers, invalid dependencies, and overlapping work items."""
    errors = []
    identifiers = [module.get("id") for module in modules]
    packages = [module.get("package") for module in modules]
    for label, values in (("module id", identifiers), ("package", packages)):
        if len(values) != len(set(values)):
            errors.append(f"duplicate {label}")
    if identifiers.count("core") != 1:
        errors.append("exactly one core module is required")
    work_items = set()
    for module in modules:
        name = module.get("id", "<missing>")
        dependencies = module.get("dependencies", [])
        if not isinstance(dependencies, list):
            errors.append(f"{name}: dependencies must be a list")
            continue
        if name == "core" and dependencies:
            errors.append("core: must not depend on tasks")
        if name != "core" and dependencies != ["core"]:
            errors.append(f"{name}: code dependencies must be ['core']")
        for dependency in dependencies + module.get("upstream_artifacts", []):
            if dependency not in identifiers or dependency == name:
                errors.append(f"{name}: unknown or self dependency {dependency}")
        for item in module.get("work_items", []):
            if item in work_items:
                errors.append(f"duplicate work item {item}")
            work_items.add(item)
    return errors


def check_imports(source: str, module_id: str, task_packages: set[str]) -> list[str]:
    """Check static imports; this is a layering check, not a runtime security boundary."""
    errors = []
    tree = ast.parse(source)
    own_package = f"gcnf_{module_id}"
    for node in ast.walk(tree):
        imports = []
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports = [node.module]
        for name in imports:
            top_level = name.split(".")[0]
            if top_level == "tasks" or (
                top_level in task_packages and (module_id == "core" or top_level != own_package)
            ):
                errors.append(f"{module_id}: forbidden task import {name}")
    return errors


def check_links(root: Path, paths: list[Path]) -> list[str]:
    errors = []
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", 1)[0]
            if not target or re.match(r"[a-zA-Z][\w+.-]*:", target):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                errors.append(f"{path.relative_to(root)}: invalid local link {target}")
    return errors


def check_repository(root: Path = ROOT) -> tuple[list[str], list[dict]]:
    errors = []
    for name in REQUIRED_ROOT:
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    directories = [root] + sorted(path for path in (root / "tasks").iterdir() if path.is_dir())
    modules = []
    locations = []
    docs = []
    for directory in directories:
        manifest = directory / "module.json"
        try:
            module = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(module, dict):
                raise ValueError("module metadata must be an object")
        except (OSError, ValueError) as exc:
            errors.append(f"{manifest.relative_to(root)}: {exc}")
            continue
        modules.append(module)
        locations.append((directory, module))
        for name in REQUIRED_MODULE:
            if not (directory / name).is_file():
                errors.append(f"{directory.relative_to(root)}: missing {name}")
        if directory != root and directory.name != module.get("id"):
            errors.append(f"{directory.name}: manifest id does not match directory")
        package_dir = directory / module.get("package_dir", "")
        if not package_dir.resolve().is_relative_to(directory.resolve()):
            errors.append(f"{module.get('id')}: package directory escapes module")
        elif not (package_dir / "__init__.py").is_file():
            errors.append(f"{module.get('id')}: missing package __init__.py")
        task_file = directory / "TASKS.md"
        if task_file.is_file():
            content = task_file.read_text(encoding="utf-8")
            for item in module.get("work_items", []):
                if item not in content:
                    errors.append(f"{module.get('id')}: undocumented work item {item}")
        if directory == root:
            docs.extend(directory.glob("*.md"))
        else:
            docs.extend(directory.rglob("*.md"))
            for name in ("configs", "scripts", "tests"):
                if not (directory / name).is_dir():
                    errors.append(f"{directory.name}: missing {name}/")
    errors.extend(check_modules(modules))
    task_packages = {module["package"] for module in modules if module.get("id") != "core"}
    for directory, module in locations:
        for path in (directory / module.get("package_dir", "")).rglob("*.py"):
            try:
                errors.extend(
                    f"{path.relative_to(root)}: {error}"
                    for error in check_imports(
                        path.read_text(encoding="utf-8"), module.get("id"), task_packages
                    )
                )
            except SyntaxError as exc:
                errors.append(f"{path.relative_to(root)}: {exc}")
    docs.extend((root / "docs").rglob("*.md"))
    errors.extend(check_links(root, docs))
    return errors, modules


def main() -> int:
    errors, modules = check_repository()
    for module in modules:
        owner = module.get("owner") or "unassigned"
        print(f"{module['id']:20} {module['priority']} {module['state']:12} {owner}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"OK: {len(modules)} modules; metadata, static imports, and local links checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
