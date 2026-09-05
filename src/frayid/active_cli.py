"""Small default CLI for the active FrayID workflow.

The historical research CLI remains available as ``frayid-legacy``. Keeping it
out of this module prevents routine commands from importing every retired V1,
V2, and V3 experiment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from frayid.capture_adapter import (
    build_adapter_interval,
    build_adapter_probe,
    export_selfrecon_dataset,
    qualify_probe_results,
)

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "docs" / "CURRENT.md"
HISTORY = ROOT / "docs" / "HISTORY.md"

app = typer.Typer(
    no_args_is_help=True,
    help="Lean entry point for the active unified-outer-surface workflow.",
)
capture_app = typer.Typer(no_args_is_help=True, help="Inspect local capture evidence.")
app.add_typer(capture_app, name="capture")


def _print_file(path: Path) -> None:
    if not path.is_file():
        raise typer.BadParameter(f"missing project document: {path}")
    typer.echo(path.read_text(encoding="utf-8"))


@app.command()
def status() -> None:
    """Print the only current project-state document."""
    _print_file(CURRENT)


@app.command()
def history() -> None:
    """Show how to recover retired plans from Git."""
    _print_file(HISTORY)


@capture_app.command("inspect")
def capture_inspect(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    count_frames: Annotated[
        bool,
        typer.Option("--count-frames", help="Decode the stream to obtain an exact frame count."),
    ] = False,
) -> None:
    """Read video metadata without modifying the capture."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise typer.BadParameter("ffprobe is required but was not found")

    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
    command.extend(
        [
            "-show_entries",
            "format=duration,size:stream=codec_name,codec_type,width,height,"
            "r_frame_rate,time_base,nb_read_frames",
            "-of",
            "json",
            str(video),
        ]
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    payload["path"] = str(video.resolve())
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@capture_app.command("prepare-probe")
def capture_prepare_probe(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    start_seconds: Annotated[
        float,
        typer.Option("--start", min=0.0, help="Start of the visually usable interval."),
    ],
    end_seconds: Annotated[
        float,
        typer.Option("--end", min=0.0, help="End of the visually usable interval."),
    ],
    samples: Annotated[
        int,
        typer.Option("--samples", min=1, max=64, help="Number of sparse qualification views."),
    ] = 12,
) -> None:
    """Build a private sparse-frame bundle for upstream adapter qualification."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise typer.BadParameter("ffmpeg and ffprobe are required but were not both found")
    try:
        manifest = build_adapter_probe(
            video,
            output_dir,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            samples=samples,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
    except (FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(manifest.resolve()))


@capture_app.command("prepare-interval")
def capture_prepare_interval(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    start_seconds: Annotated[
        float,
        typer.Option("--start", min=0.0, help="Start of the qualified usable interval."),
    ],
    end_seconds: Annotated[
        float,
        typer.Option("--end", min=0.0, help="End of the qualified usable interval."),
    ],
    frame_count: Annotated[
        int,
        typer.Option("--frames", min=2, max=360, help="Number of interval frames."),
    ] = 180,
    held_out_stride: Annotated[
        int,
        typer.Option("--held-out-stride", min=2, help="Interleaved evaluation-frame stride."),
    ] = 5,
) -> None:
    """Build a private full-interval bundle after a sparse probe qualifies."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise typer.BadParameter("ffmpeg and ffprobe are required but were not both found")
    try:
        manifest = build_adapter_interval(
            video,
            output_dir,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            frame_count=frame_count,
            held_out_stride=held_out_stride,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
    except (FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(manifest.resolve()))


@capture_app.command("qualify-probe")
def capture_qualify_probe(
    results_root: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
) -> None:
    """Qualify downloaded real CameraHMR and Sapiens2 sparse-probe outputs."""
    try:
        report = qualify_probe_results(results_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(report.resolve()))


@capture_app.command("export-selfrecon")
def capture_export_selfrecon(
    results_root: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
) -> None:
    """Export qualified full-interval evidence to SelfRecon's input interface."""
    try:
        manifest = export_selfrecon_dataset(results_root, output_dir)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(manifest.resolve()))


@app.command("legacy-help")
def legacy_help() -> None:
    """Explain how to access frozen research commands."""
    typer.echo("Use `frayid-legacy --help`; legacy commands are not active roadmap authority.")


if __name__ == "__main__":
    app()
