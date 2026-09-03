from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from frayid.io import write_json
from frayid.v2.contracts import reject_sealed_capability

ROUTE_CALL_MARKERS = {
    "detach",
    "no_grad",
    "cpu",
    "cuda",
    "to",
    "item",
    "numpy",
    "backward",
    "grad",
    "render",
    "extract",
    "rasterize",
    "interpolate",
}


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Call):
        return f"{_call_name(function)}.__call__"
    if isinstance(function, ast.Subscript) and isinstance(function.value, ast.Name):
        return f"{function.value.id}[]"
    return "dynamic"


REQUIRED_V2_PATHS = {
    "optimizer": ("src/frayid/v2/qualification.py", "qualify_outer_field"),
    "evidence": ("src/frayid/v2/evidence.py", "build_confidence_aware_visual_hull"),
    "renderer": ("src/frayid/v2/field.py", "render_rays"),
    "extraction": ("src/frayid/v2/field.py", "extract_field"),
    "topology": ("src/frayid/v2/topology.py", "certify_surface"),
    "evaluation": ("src/frayid/v2/evaluation.py", "bidirectional_chamfer"),
    "checkpoint": ("src/frayid/v2/checkpoint.py", "capture_checkpoint"),
    "g02_optimizer": ("src/frayid/v2/g02_shortcut_resistant.py", "qualify_g02_local"),
    "g02_checkpoint": (
        "src/frayid/v2/g02_shortcut_resistant.py",
        "_capture_g02_checkpoint",
    ),
    "g02_evaluation": (
        "src/frayid/v2/g02_shortcut_resistant.py",
        "run_g02_public_benchmark",
    ),
}


def audit_source_tree(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    reject_sealed_capability([source_root])
    files: list[dict[str, Any]] = []
    unresolved_dynamic_calls = 0
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root.parent.parent)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions: list[dict[str, Any]] = []
        route_calls: list[dict[str, Any]] = []
        sealed_references: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.append(
                    {
                        "kind": type(node).__name__,
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                    }
                )
            elif isinstance(node, ast.Call):
                name = _call_name(node)
                if name == "dynamic":
                    unresolved_dynamic_calls += 1
                if any(marker in name.lower() for marker in ROUTE_CALL_MARKERS):
                    route_calls.append({"name": name, "line": node.lineno})
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "sealed" in node.value.lower()
            ):
                sealed_references.append(node.lineno)
        files.append(
            {
                "path": str(relative),
                "sha256": _source_digest(path),
                "definitions": definitions,
                "route_calls": route_calls,
                "sealed_reference_lines": sorted(set(sealed_references)),
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    source_tree_digest = hashlib.sha256()
    for item in files:
        source_tree_digest.update(str(item["path"]).encode())
        source_tree_digest.update(str(item["sha256"]).encode())
    definition_index = {
        (item["path"], definition["name"]) for item in files for definition in item["definitions"]
    }
    required_path_coverage = {
        route: (path, definition) in definition_index
        for route, (path, definition) in REQUIRED_V2_PATHS.items()
    }
    blockers = [
        f"missing_required_{route}"
        for route, covered in required_path_coverage.items()
        if not covered
    ]
    if unresolved_dynamic_calls:
        blockers.append("unresolved_dynamic_calls")
    return {
        "schema_version": "frayid_v2_source_audit.v1",
        "status": "pass" if not blockers else "fail",
        "source_commit": commit,
        "source_tree_sha256": source_tree_digest.hexdigest(),
        "git_worktree_dirty": bool(git_status),
        "source_root": str(source_root),
        "file_count": len(files),
        "unresolved_dynamic_call_count": unresolved_dynamic_calls,
        "required_path_coverage": required_path_coverage,
        "blockers": blockers,
        "scientific_dispatch_blockers": (["source_worktree_not_clean"] if git_status else []),
        "files": files,
        "limitations": [
            "static_call_inventory_requires_runtime_gradient_probe_before_scientific_use",
            "sealed references are inventory only and never executable capability",
        ],
    }


def write_source_audit(source_root: Path, output_path: Path) -> Path:
    reject_sealed_capability([output_path])
    return write_json(output_path, audit_source_tree(source_root))
