"""Run the public P0 exact next-step replay gate on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frayid.io import write_json
from frayid.replay_gate import run_replay_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_replay_gate(arguments.device)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"immutable output already exists: {arguments.output}")
        write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
