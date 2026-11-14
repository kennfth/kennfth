#!/usr/bin/env python3
"""Extract scene-change keyframes with FFmpeg and write their timestamps.

Single-shot cat clips often never cross a hard scene cut. This script still
uses FFmpeg's scene score: it keeps peaks above the threshold, and if a clip
has only a few weak changes it keeps the strongest remaining peaks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from media_info import ffprobe_video, run

FRAME_RE = re.compile(r"frame:\s*(\d+).*pts_time:([0-9.]+)")
SCORE_RE = re.compile(r"scene_score=([0-9.]+)")


def scene_scores(video: Path, score_file: Path) -> list[dict]:
    score_file.parent.mkdir(parents=True, exist_ok=True)
    if score_file.exists():
        score_file.unlink()
    proc = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select='gte(scene,0)',metadata=print:file={score_file}:direct=1",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scene detection failed: {proc.stderr.strip()}")
    if not score_file.exists():
        raise RuntimeError("FFmpeg did not write scene scores")
    return _parse_scores(score_file.read_text(encoding="utf-8", errors="replace"))


def _parse_scores(text: str) -> list[dict]:
    frame = None
    pts_time = None
    rows: list[dict] = []
    for line in text.splitlines():
        match = FRAME_RE.search(line)
        if match:
            frame = int(match.group(1))
            pts_time = float(match.group(2))
            continue
        score_match = SCORE_RE.search(line)
        if score_match and frame is not None and pts_time is not None:
            rows.append(
                {
                    "frame": frame,
                    "time": pts_time,
                    "score": float(score_match.group(1)),
                }
            )
    return rows


def local_peaks(rows: list[dict]) -> list[dict]:
    peaks: list[dict] = []
    for i, row in enumerate(rows):
        score = row["score"]
        prev_score = rows[i - 1]["score"] if i else -1.0
        next_score = rows[i + 1]["score"] if i + 1 < len(rows) else -1.0
        if score > 0 and score >= prev_score and score >= next_score and row["frame"] >= 2:
            peaks.append(row)
    return peaks


def select_keyframes(
    peaks: list[dict],
    *,
    threshold: float,
    min_count: int,
    max_count: int,
    min_gap: float,
) -> tuple[list[dict], str]:
    chosen: list[dict] = []

    def take(pool: list[dict], limit: int) -> None:
        ordered = sorted(pool, key=lambda row: row["score"], reverse=True)
        for row in ordered:
            if len(chosen) >= limit:
                return
            if any(abs(row["time"] - kept["time"]) < min_gap for kept in chosen):
                continue
            chosen.append(row)

    strong = [row for row in peaks if row["score"] >= threshold]
    take(strong, max_count)
    origin = "scene_threshold"
    if len(chosen) < min_count:
        weaker = [row for row in peaks if row["score"] >= 0.04]
        take(weaker, min_count)
        origin = "scene_peaks" if chosen else origin
    chosen.sort(key=lambda row: row["time"])
    return chosen, origin


def extract_images(video: Path, frames: list[dict], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("kf_*.jpg"):
        old.unlink()
    if not frames:
        return
    expr = "+".join(f"eq(n,{row['frame']})" for row in frames)
    scale = "scale='if(gte(iw,ih),480,-2)':'if(gte(iw,ih),-2,480)'"
    proc = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select='{expr}',{scale}",
            "-fps_mode",
            "vfr",
            "-q:v",
            "4",
            str(dest / "kf_%03d.jpg"),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"keyframe extract failed: {proc.stderr.strip()}")


def extract(video: Path, out_dir: Path, threshold: float) -> dict:
    info = ffprobe_video(video)
    score_file = out_dir / "scene_scores.txt"
    rows = scene_scores(video, score_file)
    peaks = local_peaks(rows)
    chosen, origin = select_keyframes(
        peaks,
        threshold=threshold,
        min_count=3,
        max_count=8,
        min_gap=0.5,
    )
    if not chosen and info["duration_sec"] > 0:
        origin = "sampled"
        for frac in (0.15, 0.5, 0.85):
            time_sec = info["duration_sec"] * frac
            frame = min(
                int(round(time_sec * info["fps"])) if info["fps"] else 0,
                max((info["frame_count"] or 1) - 1, 0),
            )
            chosen.append({"frame": frame, "time": round(time_sec, 3), "score": 0.0})

    key_dir = out_dir / "keyframes"
    extract_images(video, chosen, key_dir)
    images = sorted(key_dir.glob("kf_*.jpg"))
    if len(images) != len(chosen):
        raise RuntimeError(
            f"expected {len(chosen)} keyframes, FFmpeg wrote {len(images)}"
        )

    keyframes = []
    for image, row in zip(images, chosen):
        keyframes.append(
            {
                "file": f"keyframes/{image.name}",
                "frame": row["frame"],
                "time": round(float(row["time"]), 3),
                "scene_score": round(float(row["score"]), 4),
            }
        )
    payload = {
        "video": video.as_posix(),
        "width": info["width"],
        "height": info["height"],
        "fps": round(info["fps"], 3),
        "duration_sec": round(info["duration_sec"], 3),
        "threshold": threshold,
        "selection": origin,
        "scene_frames_scored": len(rows),
        "keyframes": keyframes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "keyframes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    score_file.unlink(missing_ok=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="FFmpeg scene keyframes for a cat video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--threshold", type=float, default=0.25, help="scene score cut")
    args = parser.parse_args()
    if not args.video.is_file():
        print(f"missing video: {args.video}", file=sys.stderr)
        return 1
    payload = extract(args.video, args.out, args.threshold)
    print(
        f"{args.video.name}: {len(payload['keyframes'])} keyframes "
        f"({payload['selection']}) -> {args.out / 'keyframes.json'}"
    )
    for row in payload["keyframes"]:
        print(f"  {row['time']:6.2f}s  score {row['scene_score']:.3f}  {row['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
