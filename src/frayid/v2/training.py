from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class IntegratedEvidenceStage(StrEnum):
    MASK_NORMAL = "mask_normal"
    TRACK_FEATURE_VISIBILITY = "track_feature_visibility"
    BOUNDED_APPEARANCE = "bounded_appearance"


STAGE_SEQUENCE = tuple(IntegratedEvidenceStage)


class ComponentGateState(BaseModel):
    t01_passed: bool = False
    g01_passed: bool = False
    n01_passed: bool = False
    l01_passed: bool = False
    historical_records_immutable: bool = True
    sealed_test_accesses: int = Field(default=0, ge=0, le=0)

    @property
    def h01_eligible(self) -> bool:
        return self.t01_passed and self.g01_passed and self.n01_passed and self.l01_passed


class IntegratedStageController:
    def __init__(self, component_gates: ComponentGateState) -> None:
        if not component_gates.h01_eligible:
            raise ValueError("H01 requires passing T01, G01, N01, and L01 component gates")
        self.component_gates = component_gates
        self.current_stage = IntegratedEvidenceStage.MASK_NORMAL
        self.last_passing_stage: IntegratedEvidenceStage | None = None

    def record_stage_result(
        self,
        *,
        geometry_preserved: bool,
        capacity_stress_passed: bool,
    ) -> IntegratedEvidenceStage | None:
        if geometry_preserved and capacity_stress_passed:
            self.last_passing_stage = self.current_stage
            index = STAGE_SEQUENCE.index(self.current_stage)
            if index + 1 < len(STAGE_SEQUENCE):
                self.current_stage = STAGE_SEQUENCE[index + 1]
            return self.last_passing_stage
        return self.last_passing_stage
