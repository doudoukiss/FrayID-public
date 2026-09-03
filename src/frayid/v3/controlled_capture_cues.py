from __future__ import annotations

import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.signal import fftconvolve  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability

EXPERIMENT_ID: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = (
    "postv3_v01_controlled_recapture_evidence_master_r01"
)
Direction = Literal["clockwise", "counter_clockwise"]
SAMPLE_RATE: Literal[48000] = 48_000
DETECTION_SAMPLE_RATE = 16_000
PRE_ROLL_SECONDS = 3.0
HOLD_PERIOD_SECONDS = 4.0
STABLE_SECONDS = 2.5
CUE_DURATION_SECONDS = 148.0
SYNC_EVENT_SECONDS = (1.0, 74.0, 147.0)
SYNC_FREQUENCIES = {
    "clockwise": (1013.0, 1423.0, 1877.0),
    "counter_clockwise": (1097.0, 1511.0, 1999.0),
}
_SYNC_TONE_DURATION_SECONDS = 0.4
_MINIMUM_TONE_SNR_DB = 12.0
_MAXIMUM_CLOCK_SCALE_ERROR = 0.005
_MAXIMUM_SYNC_RESIDUAL_MS = 8.0


class StrictCueModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CueEvent(StrictCueModel):
    cue_seconds: float = Field(ge=0.0)
    frequency_hz: float = Field(gt=0.0)


class CueHold(StrictCueModel):
    angle_degrees: int = Field(ge=0, le=350, multiple_of=10)
    stable_start_cue_seconds: float = Field(ge=0.0)
    stable_end_cue_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _minimum_duration(self) -> CueHold:
        if self.stable_end_cue_seconds - self.stable_start_cue_seconds < 2.0:
            raise ValueError("controlled cue holds must be stable for at least two seconds")
        return self


class DirectionCue(StrictCueModel):
    direction: Direction
    audio_path: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    holds: list[CueHold]
    synchronization_events: list[CueEvent]

    @model_validator(mode="after")
    def _complete_direction(self) -> DirectionCue:
        if len(self.holds) != 36:
            raise ValueError("a controlled cue needs 36 holds")
        expected = (
            list(range(0, 360, 10)) if self.direction == "clockwise" else [0, *range(350, 0, -10)]
        )
        if [hold.angle_degrees for hold in self.holds] != expected:
            raise ValueError("controlled cue angles do not match its direction")
        if len(self.synchronization_events) != 3:
            raise ValueError("a controlled cue needs three unique synchronization events")
        return self


class ControlledCaptureCueManifest(StrictCueModel):
    schema_version: Literal["frayid_v3_controlled_capture_cues.v1"] = (
        "frayid_v3_controlled_capture_cues.v1"
    )
    experiment_id: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = EXPERIMENT_ID
    status: Literal["planning_only_not_evidence"] = "planning_only_not_evidence"
    sample_rate_hz: Literal[48000] = SAMPLE_RATE
    duration_seconds: float = Field(default=CUE_DURATION_SECONDS, ge=148.0, le=148.0)
    pre_roll_seconds: float = Field(default=PRE_ROLL_SECONDS, ge=3.0, le=3.0)
    hold_period_seconds: float = Field(default=HOLD_PERIOD_SECONDS, ge=4.0, le=4.0)
    stable_seconds: float = Field(default=STABLE_SECONDS, ge=2.5, le=2.5)
    directions: list[DirectionCue]
    html_path: str = Field(min_length=1)
    html_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_audio_replay: Literal[True] = True
    scientific_result_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _both_directions(self) -> ControlledCaptureCueManifest:
        names = [item.direction for item in self.directions]
        if len(names) != 2 or set(names) != {"clockwise", "counter_clockwise"}:
            raise ValueError("controlled cue manifest requires both directions")
        return self


def _direction_angles(direction: Direction) -> list[int]:
    return list(range(0, 360, 10)) if direction == "clockwise" else [0, *range(350, 0, -10)]


def _holds(direction: Direction) -> list[CueHold]:
    return [
        CueHold(
            angle_degrees=angle,
            stable_start_cue_seconds=PRE_ROLL_SECONDS + index * HOLD_PERIOD_SECONDS,
            stable_end_cue_seconds=(
                PRE_ROLL_SECONDS + index * HOLD_PERIOD_SECONDS + STABLE_SECONDS
            ),
        )
        for index, angle in enumerate(_direction_angles(direction))
    ]


def _add_tone(
    samples: np.ndarray,
    *,
    center_seconds: float,
    frequency_hz: float,
    duration_seconds: float,
    amplitude: float,
    sample_rate: int,
) -> None:
    count = round(duration_seconds * sample_rate)
    start = round(center_seconds * sample_rate) - count // 2
    end = start + count
    if start < 0 or end > len(samples):
        raise ValueError("cue tone lies outside the audio timeline")
    time = np.arange(count, dtype=np.float64) / sample_rate
    envelope = np.hanning(count)
    samples[start:end] += amplitude * envelope * np.sin(2.0 * np.pi * frequency_hz * time)


def _direction_audio(direction: Direction) -> np.ndarray:
    samples = np.zeros(round(CUE_DURATION_SECONDS * SAMPLE_RATE), dtype=np.float64)
    for hold in _holds(direction):
        _add_tone(
            samples,
            center_seconds=hold.stable_start_cue_seconds + 0.08,
            frequency_hz=700.0,
            duration_seconds=0.12,
            amplitude=0.20,
            sample_rate=SAMPLE_RATE,
        )
        _add_tone(
            samples,
            center_seconds=hold.stable_end_cue_seconds - 0.08,
            frequency_hz=520.0,
            duration_seconds=0.12,
            amplitude=0.16,
            sample_rate=SAMPLE_RATE,
        )
    for seconds, frequency in zip(
        SYNC_EVENT_SECONDS,
        SYNC_FREQUENCIES[direction],
        strict=True,
    ):
        _add_tone(
            samples,
            center_seconds=seconds,
            frequency_hz=frequency,
            duration_seconds=_SYNC_TONE_DURATION_SECONDS,
            amplitude=0.65,
            sample_rate=SAMPLE_RATE,
        )
    return np.asarray(np.clip(samples, -0.98, 0.98) * 32767.0, dtype="<i2")


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _cue_html() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FrayID V01 controlled capture cue</title>
<style>
html,body{height:100%;margin:0;background:#0b0d10;color:#f8fafc;font-family:system-ui,sans-serif}
main{height:100%;display:grid;place-items:center;text-align:center}.panel{width:min(92vw,1100px)}
h1{font-size:clamp(2rem,6vw,5rem);margin:.2em}.angle{font-size:clamp(5rem,24vw,16rem);font-weight:800}
.phase{font-size:clamp(2rem,7vw,5rem);font-weight:700}.hold{color:#65e572}.turn{color:#ffd166}
button{font-size:1.4rem;padding:.8rem 1.2rem;margin:.5rem;border:2px solid #f8fafc;background:#18202b;color:#fff}
</style></head><body><main><div class="panel"><h1 id="direction">V01 capture cue</h1>
<div id="angle" class="angle">READY</div><div id="phase" class="phase">Start Mac recording first</div>
<button onclick="startCue('clockwise')">Start clockwise</button>
<button onclick="startCue('counter_clockwise')">Start counter-clockwise</button></div></main>
<audio id="clockwise" src="clockwise_cue.wav" preload="auto"></audio>
<audio id="counter_clockwise" src="counter_clockwise_cue.wav" preload="auto"></audio>
<script>
const pre=3, period=4, stable=2.5, count=36;
function angleAt(direction,index){return direction==='clockwise'?(index*10)%360:(360-index*10)%360}
function startCue(direction){for(const name of ['clockwise','counter_clockwise']){const a=document.getElementById(name);a.pause();a.currentTime=0}
 const audio=document.getElementById(direction);audio.play();document.documentElement.requestFullscreen?.();tick(direction,audio)}
function tick(direction,audio){const t=audio.currentTime, angle=document.getElementById('angle'), phase=document.getElementById('phase');
 document.getElementById('direction').textContent=direction.replace('_',' ').toUpperCase();
 if(t<pre){angle.textContent=Math.max(1,Math.ceil(pre-t));phase.textContent='PRE-ROLL — HOLD 000°';phase.className='phase hold'}
 else{const elapsed=t-pre,index=Math.floor(elapsed/period),within=elapsed-index*period;
  if(index>=count){angle.textContent='DONE';phase.textContent='Keep recording through final sync tone';phase.className='phase hold'}
  else if(within<stable){angle.textContent=String(angleAt(direction,index)).padStart(3,'0')+'°';phase.textContent='HOLD STILL';phase.className='phase hold'}
  else{angle.textContent=String(angleAt(direction,Math.min(index+1,count-1))).padStart(3,'0')+'°';phase.textContent='TURN TO NEXT MARK';phase.className='phase turn'}}
 if(!audio.paused&&!audio.ended)requestAnimationFrame(()=>tick(direction,audio))}
</script></body></html>
"""


def create_controlled_capture_cue_kit(output_root: Path) -> Path:
    """Create write-once direction cues and an operator display."""
    reject_sealed_capability([output_root])
    if output_root.exists():
        raise FileExistsError(f"controlled cue kit is immutable: {output_root}")
    partials = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if partials:
        raise FileExistsError("a prior partial cue build must be audited separately")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    direction_records: list[DirectionCue] = []
    for direction in ("clockwise", "counter_clockwise"):
        samples = _direction_audio(direction)
        replay = _direction_audio(direction)
        if not np.array_equal(samples, replay):
            raise RuntimeError(f"controlled {direction} cue did not replay exactly")
        filename = f"{direction}_cue.wav"
        stage_path = stage / filename
        _write_wav(stage_path, samples)
        direction_records.append(
            DirectionCue(
                direction=direction,
                audio_path=str(output_root / filename),
                audio_sha256=sha256_file(stage_path),
                holds=_holds(direction),
                synchronization_events=[
                    CueEvent(cue_seconds=seconds, frequency_hz=frequency)
                    for seconds, frequency in zip(
                        SYNC_EVENT_SECONDS,
                        SYNC_FREQUENCIES[direction],
                        strict=True,
                    )
                ],
            )
        )
    html_stage = stage / "capture_cue.html"
    html_stage.write_text(_cue_html(), encoding="utf-8")
    manifest = ControlledCaptureCueManifest(
        directions=direction_records,
        html_path=str(output_root / html_stage.name),
        html_sha256=sha256_file(html_stage),
    )
    write_json(stage / "cue_manifest.json", manifest.model_dump(mode="json"))
    os.rename(stage, output_root)
    return output_root / "cue_manifest.json"


def _decode_audio(path: Path, *, sample_rate: int) -> np.ndarray:
    reject_sealed_capability([path])
    if not path.is_file():
        raise FileNotFoundError(f"controlled cue source is missing: {path}")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if len(audio) < sample_rate or not np.all(np.isfinite(audio)):
        raise ValueError(f"controlled cue audio is missing or invalid: {path}")
    peak = float(np.max(np.abs(audio)))
    if peak <= 1.0e-6:
        raise ValueError(f"controlled cue audio is silent: {path}")
    return audio / peak


def _detect_tone(
    audio: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
) -> dict[str, float]:
    count = round(_SYNC_TONE_DURATION_SECONDS * sample_rate)
    relative = np.arange(count, dtype=np.float64) / sample_rate
    window = np.hanning(count)
    sine = window * np.sin(2.0 * np.pi * frequency_hz * relative)
    cosine = window * np.cos(2.0 * np.pi * frequency_hz * relative)
    sine_score = fftconvolve(audio, sine[::-1], mode="same")
    cosine_score = fftconvolve(audio, cosine[::-1], mode="same")
    score = np.hypot(sine_score, cosine_score)
    peak_index = int(np.argmax(score))
    fractional = 0.0
    if 0 < peak_index < len(score) - 1:
        left, center, right = score[peak_index - 1 : peak_index + 2]
        denominator = left - 2.0 * center + right
        if abs(float(denominator)) > 1.0e-12:
            fractional = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    exclusion = max(count, sample_rate // 2)
    noise = np.concatenate(
        (
            score[: max(0, peak_index - exclusion)],
            score[min(len(score), peak_index + exclusion) :],
        )
    )
    reference = float(np.percentile(noise, 95)) if len(noise) else 0.0
    snr_db = 20.0 * np.log10((float(score[peak_index]) + 1.0e-12) / (reference + 1.0e-12))
    if snr_db < _MINIMUM_TONE_SNR_DB:
        raise ValueError(f"controlled sync tone {frequency_hz:g} Hz has SNR below 12 dB")
    return {
        "frequency_hz": frequency_hz,
        "video_seconds": (peak_index + fractional) / sample_rate,
        "snr_db": float(snr_db),
    }


def _fit_clock(
    cue_seconds: np.ndarray,
    video_seconds: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    design = np.column_stack((cue_seconds, np.ones(len(cue_seconds))))
    coefficients, *_ = np.linalg.lstsq(design, video_seconds, rcond=None)
    scale, offset = float(coefficients[0]), float(coefficients[1])
    residual_ms = (design @ coefficients - video_seconds) * 1000.0
    maximum = float(np.max(np.abs(residual_ms)))
    if abs(scale - 1.0) > _MAXIMUM_CLOCK_SCALE_ERROR:
        raise ValueError(f"controlled {label} clock scale differs by more than 0.5%")
    if maximum > _MAXIMUM_SYNC_RESIDUAL_MS:
        raise ValueError(f"controlled {label} cue residual exceeds 8 ms")
    return {
        "scale": scale,
        "offset_seconds": offset,
        "maximum_absolute_residual_ms": maximum,
        "rms_residual_ms": float(np.sqrt(np.mean(residual_ms**2))),
    }


def _events_for_direction(
    audio: np.ndarray,
    direction: Direction,
    *,
    sample_rate: int,
) -> list[dict[str, float]]:
    return [
        _detect_tone(audio, frequency_hz=frequency, sample_rate=sample_rate)
        for frequency in SYNC_FREQUENCIES[direction]
    ]


def detect_controlled_capture_cues(
    *,
    cue_manifest_path: Path,
    clockwise_path: Path,
    counter_clockwise_path: Path,
    evaluator_path: Path | None,
    output_path: Path,
) -> Path:
    """Recover native hold intervals and optional evaluator clock maps from cue audio."""
    paths = [
        cue_manifest_path,
        clockwise_path,
        counter_clockwise_path,
        output_path,
    ]
    if evaluator_path is not None:
        paths.append(evaluator_path)
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError(f"controlled cue detection is immutable: {output_path}")
    manifest = ControlledCaptureCueManifest.model_validate(read_json(cue_manifest_path))
    for record in manifest.directions:
        audio_path = Path(record.audio_path)
        if not audio_path.is_file() or sha256_file(audio_path) != record.audio_sha256:
            raise ValueError(f"controlled {record.direction} cue audio hash mismatch")
    source_paths = {
        "clockwise": clockwise_path,
        "counter_clockwise": counter_clockwise_path,
    }
    if evaluator_path is not None:
        source_paths["evaluator"] = evaluator_path
    initial_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    audio = {
        name: _decode_audio(path, sample_rate=DETECTION_SAMPLE_RATE)
        for name, path in source_paths.items()
    }
    records_by_direction = {record.direction: record for record in manifest.directions}

    def solve() -> dict[str, Any]:
        directions: list[dict[str, Any]] = []
        for direction in ("clockwise", "counter_clockwise"):
            cue_record = records_by_direction[direction]
            training_events = _events_for_direction(
                audio[direction], direction, sample_rate=DETECTION_SAMPLE_RATE
            )
            cue_times = np.asarray(
                [item.cue_seconds for item in cue_record.synchronization_events],
                dtype=np.float64,
            )
            training_times = np.asarray(
                [item["video_seconds"] for item in training_events], dtype=np.float64
            )
            training_clock = _fit_clock(cue_times, training_times, label=f"{direction} training")
            holds = [
                {
                    "angle_degrees": hold.angle_degrees,
                    "stable_start_seconds": (
                        training_clock["scale"] * hold.stable_start_cue_seconds
                        + training_clock["offset_seconds"]
                    ),
                    "stable_end_seconds": (
                        training_clock["scale"] * hold.stable_end_cue_seconds
                        + training_clock["offset_seconds"]
                    ),
                }
                for hold in cue_record.holds
            ]
            if (
                min(hold["stable_end_seconds"] - hold["stable_start_seconds"] for hold in holds)
                < 2.0
            ):
                raise ValueError("detected controlled hold duration fell below two seconds")
            direction_record: dict[str, Any] = {
                "direction": direction,
                "training_events": training_events,
                "training_clock_from_cue": training_clock,
                "holds": holds,
            }
            if evaluator_path is not None:
                evaluator_events = _events_for_direction(
                    audio["evaluator"], direction, sample_rate=DETECTION_SAMPLE_RATE
                )
                evaluator_times = np.asarray(
                    [item["video_seconds"] for item in evaluator_events], dtype=np.float64
                )
                evaluator_clock = _fit_clock(
                    cue_times, evaluator_times, label=f"{direction} evaluator"
                )
                evaluator_from_training = _fit_clock(
                    training_times,
                    evaluator_times,
                    label=f"{direction} evaluator-from-training",
                )
                direction_record.update(
                    {
                        "evaluator_events": evaluator_events,
                        "evaluator_clock_from_cue": evaluator_clock,
                        "evaluator_time_from_training_scale": evaluator_from_training["scale"],
                        "evaluator_time_from_training_offset_seconds": evaluator_from_training[
                            "offset_seconds"
                        ],
                        "synchronization_residual_ms": evaluator_from_training[
                            "maximum_absolute_residual_ms"
                        ],
                        "method": "audible_visual_sync_event",
                    }
                )
            else:
                direction_record.update(
                    {
                        "evaluator_events": None,
                        "evaluator_clock_from_cue": None,
                        "evaluator_time_from_training_scale": None,
                        "evaluator_time_from_training_offset_seconds": None,
                        "synchronization_residual_ms": None,
                        "method": "cue_to_native_training_audio_only",
                    }
                )
            directions.append(direction_record)
        return {"directions": directions}

    primary = solve()
    replay = solve()
    if primary != replay:
        raise RuntimeError("controlled cue detection did not replay exactly")
    final_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    if initial_hashes != final_hashes:
        raise RuntimeError("controlled cue source bytes changed during detection")
    payload = {
        "schema_version": "frayid_v3_controlled_capture_cue_detection.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "capture_mode": (
            "dual_camera_metric_evaluation"
            if evaluator_path is not None
            else "single_camera_evidence_consistent"
        ),
        "scientific_claim_ceiling": (
            "metric_accuracy_only_after_independent_evaluator_gates"
            if evaluator_path is not None
            else "evidence_consistent_mantle_reconstruction"
        ),
        "metric_accuracy_claim_allowed": False,
        "independent_evaluator_available": evaluator_path is not None,
        "metric_accuracy_blockers": (
            [] if evaluator_path is not None else ["independent_evaluator_camera_unavailable"]
        ),
        "evidence_scope": "native_audio_timing_only",
        "cue_manifest_path": str(cue_manifest_path),
        "cue_manifest_sha256": sha256_file(cue_manifest_path),
        "source_audio": {
            name: {"path": str(source_paths[name]), "sha256": initial_hashes[name]}
            for name in sorted(source_paths)
        },
        "detection_sample_rate_hz": DETECTION_SAMPLE_RATE,
        "directions": primary["directions"],
        "exact_same_input_replay": True,
        "training_video_frames_decoded": 0,
        "evaluator_video_frames_decoded": 0,
        "training_audio_streams_read": 2,
        "evaluator_audio_streams_read_for_sync_only": (1 if evaluator_path is not None else 0),
        "evaluator_fitting_access": False,
        "evaluator_parameter_selection_access": False,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_geometry_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "scientific_result_claimed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


__all__ = [
    "ControlledCaptureCueManifest",
    "CueEvent",
    "CueHold",
    "DirectionCue",
    "create_controlled_capture_cue_kit",
    "detect_controlled_capture_cues",
]
