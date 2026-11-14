"""Small ffprobe helpers shared by the cat highlight scripts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )


def ffprobe_video(path: Path) -> dict:
    """Return width, height, fps, and duration for the first video stream."""
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout or "{}")
    streams = [s for s in payload.get("streams", []) if s.get("width")]
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    fps = _ratio(rate)
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"no duration for {path}")
    frames = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_sec": float(duration),
        "frame_count": int(frames) if frames not in (None, "N/A") else None,
    }


def has_audio(path: Path) -> bool:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return bool((proc.stdout or "").strip())


def _ratio(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(value)
