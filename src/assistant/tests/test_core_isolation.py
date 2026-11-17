"""The core is extracted into portable-slm unchanged, so it must not depend on the editor."""

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
FORBIDDEN = ("src.assistant.host", "src.routers", "src.reviewer", "main", "src.DataUtils", "src.job_")


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_core_imports_nothing_from_the_editor():
    offenders = [
        f"{p.name}: {mod}"
        for p in sorted(CORE.glob("*.py"))
        for mod in imported_modules(p)
        if mod == "main" or any(mod.startswith(f) for f in FORBIDDEN if f != "main")
    ]
    assert offenders == []


def test_core_only_imports_itself_within_src():
    for p in CORE.glob("*.py"):
        for mod in imported_modules(p):
            if mod.startswith("src."):
                assert mod.startswith("src.assistant.core"), f"{p.name} imports {mod}"
