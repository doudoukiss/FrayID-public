"""Run the preregistered public P2 exact-refinement correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_post_v1_e12_public_gate as e12
import run_post_v1_e14_public_gate as e14
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import (
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.io import write_json
from frayid.refinement_certificate import (
    RefinementProvenance,
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)
from frayid.source_exclusion_carrier import uniform_conforming_subdivide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS_ID = "postv1_p2_exact_refinement_certificate_r01"
REPORT_SCHEMA = "post_v1_p2_public_exact_refinement_gate.v1"
MAX_SECONDS = 30 * 60
MAXIMUM_FINAL_FACES = 20_000


def _provenance_bytes(refinement: RefinementProvenance) -> bytes:
    header = json.dumps(
        {
            "denominator": refinement.denominator,
            "faces_shape": list(refinement.faces.shape),
            "parents_shape": list(refinement.parent_face_indices.shape),
            "barycentrics_shape": list(refinement.barycentric_numerators.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return b"\n".join(
        (
            header,
            np.ascontiguousarray(refinement.faces, dtype="<i8").tobytes(),
            np.ascontiguousarray(refinement.parent_face_indices, dtype="<i8").tobytes(),
            np.ascontiguousarray(refinement.barycentric_numerators, dtype="<i8").tobytes(),
        )
    )


def _run_fixture(
    fixture: GenusCarrierFidelityFixture,
    *,
    constructor: Path,
    auditor: Path,
    root: Path,
) -> dict[str, Any]:
    fixture_root = root / fixture.name
    fixture_root.mkdir(parents=True, exist_ok=False)
    source_path = fixture_root / "source.e6mesh"
    e12._write_fixture(source_path, fixture)
    grid = e14._grid_for_fixture(fixture)
    repetitions: list[dict[str, Any]] = []
    complete_bytes: list[list[bytes]] = []
    blockers: list[str] = []

    for repetition in range(2):
        parent_path = fixture_root / f"parent_{repetition}.e10mesh"
        constructed = subprocess.run(
            [str(constructor), str(source_path), str(parent_path)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_SECONDS,
        )
        if constructed.returncode != 0 or not parent_path.is_file():
            blockers.append(f"repetition_{repetition}:constructor_failure")
            repetitions.append(
                {
                    "repetition": repetition,
                    "status": "fail",
                    "diagnostic": (constructed.stdout + constructed.stderr).strip(),
                }
            )
            continue
        parent_vertices, parent_faces = read_e10_mesh(parent_path)
        level_reports: list[dict[str, Any]] = []
        serialized: list[bytes] = []
        for rounds in range(3):
            refinement = subdivide_with_exact_provenance(
                parent_vertices, parent_faces, rounds=rounds
            )
            frozen_vertices, frozen_faces = uniform_conforming_subdivide(
                parent_vertices, parent_faces, rounds=rounds
            )
            matches_frozen = bool(
                np.array_equal(refinement.vertices, frozen_vertices)
                and np.array_equal(refinement.faces, frozen_faces)
            )
            certificate = certify_exact_dyadic_refinement(
                parent_vertices,
                parent_faces,
                refinement,
                parent_grid=grid,
                rounds=rounds,
            )
            mesh_path = fixture_root / f"round_{rounds}_{repetition}.e10mesh"
            e12._write_mesh(mesh_path, refinement.vertices, refinement.faces)
            audit_path = fixture_root / f"round_{rounds}_{repetition}_audit.json"
            audit, diagnostic = e14._audit(auditor, source_path, mesh_path, audit_path)
            level_blockers: list[str] = []
            if audit.get("status") != "pass":
                level_blockers.append("exact_audit")
            if certificate.status != "pass":
                level_blockers.append("exact_refinement_certificate")
            if not matches_frozen:
                level_blockers.append("frozen_refinement_mismatch")
            if len(refinement.faces) > MAXIMUM_FINAL_FACES:
                level_blockers.append("face_cap")
            blockers.extend(
                f"repetition_{repetition}:round_{rounds}:{value}" for value in level_blockers
            )
            certificate_record = certificate.report()
            serialized.extend(
                (
                    mesh_path.read_bytes(),
                    _provenance_bytes(refinement),
                    json.dumps(certificate_record, sort_keys=True, separators=(",", ":")).encode(
                        "ascii"
                    ),
                )
            )
            level_reports.append(
                {
                    "rounds": rounds,
                    "vertex_count": len(refinement.vertices),
                    "face_count": len(refinement.faces),
                    "matches_frozen_refinement": matches_frozen,
                    "certificate": certificate_record,
                    "exact_audit": audit,
                    "exact_diagnostic": diagnostic,
                    "status": "pass" if not level_blockers else "fail",
                    "blockers": level_blockers,
                }
            )
        complete_bytes.append(serialized)
        repetitions.append(
            {
                "repetition": repetition,
                "status": (
                    "pass" if all(level["status"] == "pass" for level in level_reports) else "fail"
                ),
                "levels": level_reports,
            }
        )

    deterministic = bool(
        len(complete_bytes) == 2
        and len(complete_bytes[0]) == 9
        and complete_bytes[0] == complete_bytes[1]
    )
    if complete_bytes and not deterministic:
        blockers.append("nondeterministic_mesh_or_certificate_serialization")
    return {
        **fixture.as_public_record(),
        "status": "pass" if not blockers else "fail",
        "repetitions": repetitions,
        "deterministic_mesh_and_certificate_bytes": deterministic,
        "round_2_bundle_sha256": (
            hashlib.sha256(b"".join(complete_bytes[0][-3:])).hexdigest() if complete_bytes else None
        ),
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = e14.build_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-p2-public-") as temporary_name:
        root = Path(temporary_name)
        fixtures = [
            _run_fixture(fixture, constructor=constructor, auditor=auditor, root=root)
            for fixture in public_genus_fidelity_fixtures()
        ]
    blockers = [f"fixture:{fixture['name']}" for fixture in fixtures if fixture["status"] != "pass"]
    return {
        "schema_version": REPORT_SCHEMA,
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_exact_refinement_identity_certificate",
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_geometry_only",
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "fixtures": fixtures,
        "blockers": blockers,
        "elapsed_seconds": time.monotonic() - started,
        "execution_counters": {
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "modal_invocations": 0,
            "sealed_test_accesses": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"immutable P2 report exists: {arguments.output}")
    report = run_public_gate()
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
