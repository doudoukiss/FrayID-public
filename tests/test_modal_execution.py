from __future__ import annotations

import json
from pathlib import Path

import pytest

from frayid.modal_execution import (
    RunBinding,
    claim_attempt,
    image_definition_sha256,
    write_attempt_event,
)


def _binding() -> RunBinding:
    image_sha = image_definition_sha256({"name": "frayid-cuda", "torch": "2.7.1"})
    return RunBinding(
        schema_version="frayid_modal_run_binding.v1",
        run_id="p1-public-20260901-r01",
        git_commit="a" * 40,
        config_sha256="b" * 64,
        input_hashes={"public_fixture": "c" * 64},
        image_name="frayid-cuda",
        image_definition_sha256=image_sha,
        resource_type="L40S",
        random_seed=20260831,
    )


def test_run_binding_round_trip_and_expected_contract() -> None:
    binding = _binding()
    restored = RunBinding.from_json(binding.to_json())
    assert restored == binding
    restored.validate(
        expected_image_definition_sha256=binding.image_definition_sha256,
        expected_resource_type="L40S",
    )
    with pytest.raises(ValueError, match="resource type"):
        restored.validate(expected_resource_type="A100")


def test_attempt_claim_refuses_reschedule_and_output_overwrite(tmp_path: Path) -> None:
    binding = _binding()
    attempts = tmp_path / "attempts"
    outputs = tmp_path / "runs"
    output = claim_attempt(binding, attempt_root=attempts, output_root=outputs)
    assert output == outputs / binding.run_id
    claim = json.loads((attempts / binding.run_id / "claimed.json").read_text())
    assert claim["binding"]["git_commit"] == binding.git_commit
    with pytest.raises(FileExistsError, match="automatic continuation"):
        claim_attempt(binding, attempt_root=attempts, output_root=outputs)


def test_attempt_events_are_append_only(tmp_path: Path) -> None:
    binding = _binding()
    attempts = tmp_path / "attempts"
    outputs = tmp_path / "runs"
    claim_attempt(binding, attempt_root=attempts, output_root=outputs)
    event = write_attempt_event(
        binding,
        attempt_root=attempts,
        event_name="completed",
        payload={"status": "pass"},
    )
    assert event.is_file()
    with pytest.raises(FileExistsError):
        write_attempt_event(
            binding,
            attempt_root=attempts,
            event_name="completed",
            payload={"status": "pass"},
        )


def test_modal_entrypoints_package_shared_layer_for_remote_import() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "modal_post_v1_e11_genus_carrier.py",
        "modal_post_v1_e11_fidelity_gate.py",
        "modal_post_v1_p1_determinism.py",
    ):
        source_path = root / "scripts" / name
        if not source_path.is_file():
            pytest.skip("operator-only Modal entrypoints are excluded from the public snapshot")
        source = source_path.read_text(encoding="utf-8")
        assert '"/root/modal_execution_layer.py"' in source
        assert source.index('"/root/modal_execution_layer.py"') < source.index("@app.function")


def test_p1_claims_and_commits_before_scientific_imports() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/modal_post_v1_p1_determinism.py"
    if not source_path.is_file():
        pytest.skip("operator-only Modal entrypoint is excluded from the public snapshot")
    source = source_path.read_text(encoding="utf-8")
    claim = source.index("output_root = claim_attempt(")
    commit = source.index("immutable_output_volume.commit()", claim)
    renderer_import = source.index("from frayid.renderer_determinism", commit)
    assert claim < commit < renderer_import


def test_pinned_modal_images_include_transitive_runtime_imports() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/modal_execution_layer.py"
    if not source_path.is_file():
        pytest.skip("operator-only Modal image definition is excluded from the public snapshot")
    source = source_path.read_text(encoding="utf-8")
    assert '"pydantic==2.11.7"' in source
    assert '"scikit-image==0.25.2"' in source


def test_modal_images_smoke_import_scientific_modules_before_function_start() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = {
        "modal_post_v1_e11_genus_carrier.py": "import frayid.interface_field",
        "modal_post_v1_e11_fidelity_gate.py": "import frayid.genus_carrier",
        "modal_post_v1_p1_determinism.py": "import frayid.renderer_determinism",
    }
    for name, required_import in checks.items():
        source_path = root / "scripts" / name
        if not source_path.is_file():
            pytest.skip("operator-only Modal entrypoints are excluded from the public snapshot")
        source = source_path.read_text(encoding="utf-8")
        assert required_import in source
        assert source.index(required_import) < source.index("@app.function")


def test_e11_fidelity_modal_runner_requires_local_pass_activation() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "scripts/modal_post_v1_e11_fidelity_gate.py"
    config_path = root / "configs/execution/modal_unified_v1.yaml"
    if not source_path.is_file():
        pytest.skip("operator-only Modal entrypoint is excluded from the public snapshot")
    source = source_path.read_text(encoding="utf-8")
    config = config_path.read_text(encoding="utf-8")
    assert "fidelity_modal_replication_permitted" in source
    assert "fidelity_modal_replication_permitted: false" in config
