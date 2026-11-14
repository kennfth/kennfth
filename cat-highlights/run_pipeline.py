#!/usr/bin/env python3
"""Run keyframes, cat tracking, and highlight cuts for one or more videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Cat highlight pipeline")
    parser.add_argument("videos", nargs="*", type=Path, help="defaults to test_videos/orange_cat_*.mp4")
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    videos = args.videos
    if not videos:
        videos = sorted((ROOT / "test_videos").glob("orange_cat_*.mp4"))
    if not videos:
        print("no videos. Pass a file or run download_samples.py", file=sys.stderr)
        return 1
    py = sys.executable
    for video in videos:
        if not video.is_file():
            print(f"missing video: {video}", file=sys.stderr)
            return 1
        out = args.out / video.stem
        steps = [
            [py, str(ROOT / "extract_keyframes.py"), str(video), "--out", str(out), "--threshold", str(args.threshold)],
            [py, str(ROOT / "detect_cats.py"), str(video), "--out", str(out), "--stride", str(args.stride)],
            [py, str(ROOT / "make_highlights.py"), str(video), "--out", str(out)],
        ]
        for cmd in steps:
            print("+", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
