"""Audit R01 source, licensed inputs, public example, and prebuilt runtime binding.

This command performs only local, read-only evidence checks and writes one
immutable JSON report. It does not download, build a runtime, open a camera,
launch a GPU worker, or begin SelfRecon training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frayid.selfrecon_reference import audit_reference_binding, load_reference_spec
from frayid.v3.contracts import write_immutable_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv3_r01_selfrecon_reference_reproduction_r01"
RUN_ID = "registered-20260903-r01"
DEFAULT_SPEC = PROJECT_ROOT / "configs/reconstruction/selfrecon_reference_public_v1.yaml"
OUTPUT_ROOT = Path("outputs/post_v3") / EXPERIMENT_ID / RUN_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evidence-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_reference_spec(args.spec)
    report = audit_reference_binding(spec=spec, evidence_root=args.evidence_root)
    output = write_immutable_json(
        OUTPUT_ROOT, Path("qualification/local_binding_audit.json"), report
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
