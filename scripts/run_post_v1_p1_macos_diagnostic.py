"""Run the P1 deterministic CPU contract and record unavailable CUDA gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path

import torch

from frayid.camera import make_intrinsics
from frayid.io import write_json
from frayid.renderer_determinism import cpu_reference_trace, first_bitwise_tensor_difference

EXPERIMENT_ID = "postv1_p1_opaque_renderer_deterministic_execution"


def run_diagnostic() -> dict[str, object]:
    vertices = torch.tensor(
        [[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [0.0, 0.4, 2.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    intrinsics = make_intrinsics(40.0, (15.0, 17.0))
    reference = cpu_reference_trace(
        vertices,
        faces,
        intrinsics,
        (32, 36),
        source_image_size=(80, 120),
    )
    first_difference = None
    completed_repeats = 0
    for repeat in range(100):
        candidate = cpu_reference_trace(
            vertices,
            faces,
            intrinsics,
            (32, 36),
            source_image_size=(80, 120),
        )
        difference = first_bitwise_tensor_difference(reference, candidate)
        if difference is not None:
            first_difference = {
                "repeat": repeat,
                "tensor": difference.name,
                "first_flat_index": difference.first_flat_index,
                "maximum_absolute_difference": difference.maximum_absolute_difference,
            }
            break
        completed_repeats += 1
    cuda_available = torch.cuda.is_available()
    nvdiffrast_available = importlib.util.find_spec("nvdiffrast") is not None
    return {
        "schema_version": "post_v1_p1_macos_diagnostic.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "blocked_no_cuda_host",
        "platform": platform.system().lower(),
        "cpu_forward_contract": {
            "status": "pass" if first_difference is None else "fail",
            "repeat_count": completed_repeats,
            "first_bitwise_difference": first_difference,
            "off_centre_source_dimensions": [80, 120],
            "render_dimensions": [32, 36],
        },
        "trace_interface": {
            "implemented": True,
            "stages": [
                "clip_vertices",
                "raster",
                "raster_derivatives",
                "point_sampled_coverage",
                "interpolated_normals",
                "antialiased_coverage",
                "antialiased_normals",
                "geometry_gradient",
                "interpolated_attribute_gradient",
                "final_parameter_gradient",
            ],
        },
        "cuda_runtime": {
            "available": cuda_available,
            "nvdiffrast_importable": nvdiffrast_available,
            "same_process_forward_backward_repeats": 0,
            "checkpoint_v2_next_step_repeats": 0,
            "state": "not_run_no_cuda_host",
        },
        "atomic_reduction_hypothesis": "not_tested_without_cuda_trace",
        "renderer_modified": False,
        "e8_reopened": False,
        "development_evaluations": 0,
        "sealed_test_accesses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_diagnostic()
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"immutable P1 report exists: {arguments.output}")
        write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["cpu_forward_contract"]["status"] != "pass":  # type: ignore[index]
        raise SystemExit(1)


if __name__ == "__main__":
    main()
