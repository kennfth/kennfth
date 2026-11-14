#!/usr/bin/env python3
"""Download the Pexels orange-cat samples and transcode them to a smaller mp4.

Source files are not committed. The script writes test_videos/orange_cat_XX.mp4
and test_videos/sources.json (attribution only).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from media_info import ffprobe_video, has_audio, run

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "test_videos"

SAMPLES = [
    {
        "file": "orange_cat_01.mp4",
        "title": "Orange stray cat in the public park",
        "author": "Zeki Binici",
        "page": "https://www.pexels.com/video/orange-stray-cat-in-the-public-park-19975462/",
        "source_url": "https://videos.pexels.com/video-files/19975462/19975462-hd_1280_720_25fps.mp4",
        "license": "Pexels License",
    },
    {
        "file": "orange_cat_02.mp4",
        "title": "An orange tabby cat playing with a stick",
        "author": "Nothing Ahead",
        "page": "https://www.pexels.com/video/an-orange-tabby-cat-playing-with-a-stick-15024233/",
        "source_url": "https://videos.pexels.com/video-files/15024233/15024233-hd_1280_720_30fps.mp4",
        "license": "Pexels License",
    },
    {
        "file": "orange_cat_03.mp4",
        "title": "An orange tabby kitten playing among garden plants",
        "author": "Marsel Sharipov",
        "page": "https://www.pexels.com/video/an-orange-tabby-kitten-playing-among-garden-plants-13825474/",
        "source_url": "https://videos.pexels.com/video-files/13825474/13825474-hd_720_1280_30fps.mp4",
        "license": "Pexels License",
    },
    {
        "file": "orange_cat_04.mp4",
        "title": "Angery orange cat with duck",
        "author": "Hehe Panda",
        "page": "https://www.pexels.com/video/angery-orange-cat-with-duck-20503094/",
        "source_url": "https://videos.pexels.com/video-files/20503094/20503094-hd_720_1280_60fps.mp4",
        "license": "Pexels License",
    },
    {
        "file": "orange_cat_05.mp4",
        "title": "A ginger cat looking around",
        "author": "cottonbro studio",
        "page": "https://www.pexels.com/video/a-ginger-cat-looking-around-6853904/",
        "source_url": "https://videos.pexels.com/video-files/6853904/6853904-uhd_2160_4096_25fps.mp4",
        "license": "Pexels License",
    },
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 100_000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "cat-highlights/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)


def transcode(src: Path, dest: Path) -> None:
    audio = has_audio(src)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        "scale='if(gte(iw,ih),960,-2)':'if(gte(iw,ih),-2,960)',fps=30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-movflags",
        "+faststart",
    ]
    if audio:
        cmd.extend(["-c:a", "aac", "-ac", "1", "-b:a", "96k"])
    else:
        cmd.append("-an")
    cmd.append(str(dest))
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache-src"
    cache.mkdir(exist_ok=True)
    manifest = []
    for sample in SAMPLES:
        raw = cache / sample["file"]
        print(f"fetch {sample['file']}", flush=True)
        fetch(sample["source_url"], raw)
        out = DEST / sample["file"]
        print(f"transcode {sample['file']}", flush=True)
        transcode(raw, out)
        info = ffprobe_video(out)
        if not 14.5 <= info["duration_sec"] <= 60.5:
            print(
                f"{sample['file']} is {info['duration_sec']:.2f}s, expected 15-60s",
                file=sys.stderr,
            )
            return 1
        manifest.append(
            {
                "file": sample["file"],
                "title": sample["title"],
                "author": sample["author"],
                "page": sample["page"],
                "source_url": sample["source_url"],
                "license": sample["license"],
                "duration_sec": round(info["duration_sec"], 3),
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 3),
            }
        )
        print(
            f"  {info['width']}x{info['height']} {info['duration_sec']:.2f}s {info['fps']:.2f}fps",
            flush=True,
        )
    (DEST / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST / 'sources.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
