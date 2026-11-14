#!/usr/bin/env python3
"""Turn cat tracks into jump, fail, exit, motion-spike, and multi-cat clips.

Rules use the boxes in events.json. FFmpeg cuts a short clip around each
kept moment. Host lines are templates unless a local Qwen-VL model is on
the machine; the pipeline does not wait on one.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from media_info import has_audio, run

TEMPLATES = {
    "jump": [
        "The orange one commits. Paws leave the floor.",
        "Hop's up. No countdown, no plan.",
    ],
    "fail": [
        "Airtime was the easy part. The landing files a complaint.",
        "That was a jump in theory and a drop in practice.",
    ],
    "exit": [
        "Exit, pursued by nothing. The cat is done with the shot.",
        "Offscreen. The orange one declines a second take.",
    ],
    "reentry": [
        "And we're back. The exit was a fake-out.",
        "Re-entry. The bit was not over.",
    ],
    "motion_spike": [
        "The orange one breaks the pose.",
        "That was not a still frame.",
    ],
    "multi_cat": [
        "Two cats enter. The plan does not survive contact.",
        "Multi-cat moment. Nobody is in charge.",
    ],
}

PAD = {
    "jump": (0.7, 1.2),
    "fail": (1.1, 0.9),
    "exit": (0.9, 0.7),
    "reentry": (0.4, 1.1),
    "motion_spike": (0.7, 1.1),
    "multi_cat": (0.8, 1.4),
}


def stamp(time_sec: float) -> str:
    minutes = int(time_sec // 60)
    seconds = time_sec - minutes * 60
    return f"{minutes}:{seconds:04.1f}"


def host_line(event_type: str, time_sec: float, index: int) -> str:
    options = TEMPLATES[event_type]
    return f"[{stamp(time_sec)}] {options[index % len(options)]}"


def vlm_status() -> str:
    """Return 'template' unless a local Qwen-VL model is actually installed."""
    ollama = shutil.which("ollama")
    if not ollama:
        return "template"
    try:
        proc = subprocess.run([ollama, "list"], capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return "template"
    text = f"{proc.stdout}\n{proc.stderr}".lower()
    if "qwen" in text and "vl" in text:
        return "qwen-vl"
    return "template"


def center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def group_tracks(detections: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in detections:
        grouped.setdefault(int(row["track_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["time"])
    return grouped


def find_jumps_and_fails(rows: list[dict], frame_h: float) -> list[dict]:
    if len(rows) < 3:
        return []
    heights = [row["bbox"][3] - row["bbox"][1] for row in rows]
    body = max(median(heights), 8.0)
    events: list[dict] = []
    jumps: list[dict] = []
    for i, start in enumerate(rows):
        for end in rows[i + 1 :]:
            dt = end["time"] - start["time"]
            if dt < 0.12:
                continue
            if dt > 0.45:
                break
            rise = center(start["bbox"])[1] - center(end["bbox"])[1]
            bottom_rise = start["bbox"][3] - end["bbox"][3]
            need = max(0.06 * frame_h, 0.55 * body)
            bottom_need = max(0.04 * frame_h, 0.35 * body)
            if rise < need or bottom_rise < bottom_need:
                continue
            if rise / dt < 0.8 * body:
                continue
            jumps.append(
                {
                    "type": "jump",
                    "time": round((start["time"] + end["time"]) / 2, 3),
                    "start_time": start["time"],
                    "end_time": end["time"],
                    "score": round(min(1.0, rise / max(frame_h, 1)), 3),
                    "track_id": start["track_id"],
                    "detail": f"center rose {rise:.0f}px in {dt:.2f}s",
                    "rise": rise,
                }
            )
    jumps = _suppress(jumps, 0.6)
    events.extend(jumps)

    for jump in jumps:
        window = [row for row in rows if jump["start_time"] - 0.05 <= row["time"] <= jump["end_time"] + 0.9]
        if len(window) < 2:
            continue
        apex = min(window, key=lambda row: center(row["bbox"])[1])
        after = [row for row in window if row["time"] >= apex["time"] + 0.08]
        if not after:
            continue
        landing = max(after, key=lambda row: center(row["bbox"])[1])
        drop = center(landing["bbox"])[1] - center(apex["bbox"])[1]
        dt = landing["time"] - apex["time"]
        if dt <= 0 or dt > 0.55:
            continue
        if drop < max(0.6 * jump["rise"], 0.05 * frame_h):
            continue
        events.append(
            {
                "type": "fail",
                "time": round((apex["time"] + landing["time"]) / 2, 3),
                "score": round(min(1.0, drop / max(frame_h, 1) + 0.05), 3),
                "track_id": jump["track_id"],
                "detail": f"dropped {drop:.0f}px in {dt:.2f}s after a rise of {jump['rise']:.0f}px",
            }
        )
    return events


def leaving_edge(box: list[float], width: float, height: float) -> str | None:
    x1, y1, x2, y2 = box
    cx, cy = center(box)
    if x1 <= 2 and cx < width * 0.18:
        return "left"
    if x2 >= width - 2 and cx > width * 0.82:
        return "right"
    if y1 <= 2 and cy < height * 0.18:
        return "top"
    if y2 >= height - 2 and cy > height * 0.82 and (y2 - y1) < height * 0.45:
        return "bottom"
    return None


def moving_toward(rows: list[dict], edge: str, width: float, height: float) -> bool:
    last = rows[-1]
    earlier = [row for row in rows if last["time"] - 1.2 <= row["time"] <= last["time"] - 0.15]
    if not earlier:
        return False
    prev = earlier[0]
    cx0, cy0 = center(prev["bbox"])
    cx1, cy1 = center(last["bbox"])
    if edge == "left":
        return cx1 < cx0 - 0.04 * width
    if edge == "right":
        return cx1 > cx0 + 0.04 * width
    if edge == "top":
        return cy1 < cy0 - 0.04 * height
    return cy1 > cy0 + 0.04 * height


def find_exits(tracks: dict[int, list[dict]], width: float, height: float, duration: float) -> list[dict]:
    events = []
    ordered = sorted(tracks.items(), key=lambda item: item[1][0]["time"])
    for track_id, rows in ordered:
        if rows[-1]["time"] - rows[0]["time"] < 0.35:
            continue
        edge = leaving_edge(rows[-1]["bbox"], width, height)
        if edge is None:
            continue
        ends_early = duration - rows[-1]["time"] >= 0.35
        if not (ends_early or moving_toward(rows, edge, width, height)):
            continue
        events.append(
            {
                "type": "exit",
                "time": round(rows[-1]["time"], 3),
                "score": 0.62,
                "track_id": track_id,
                "detail": f"left via the {edge} edge",
            }
        )
    exits = [event for event in events if event["type"] == "exit"]
    for track_id, rows in ordered:
        if rows[0]["time"] < 0.75:
            continue
        edge = leaving_edge(rows[0]["bbox"], width, height)
        if edge is None:
            continue
        prior = [
            event
            for event in exits
            if event["track_id"] != track_id and 0 <= rows[0]["time"] - event["time"] <= 4
        ]
        if not prior:
            continue
        events.append(
            {
                "type": "reentry",
                "time": round(rows[0]["time"], 3),
                "score": 0.58,
                "track_id": track_id,
                "detail": f"came back from the {edge} edge",
            }
        )
    return events


def track_speeds(rows: list[dict]) -> list[dict]:
    """Center speed over ~0.2s. A one-frame box twitch that reverses is dropped."""
    speeds = []
    for index, start in enumerate(rows):
        end = next((row for row in rows[index + 1 :] if row["time"] >= start["time"] + 0.18), None)
        if end is None:
            break
        dt = end["time"] - start["time"]
        if dt <= 0 or dt > 0.4:
            continue
        ax, ay = center(start["bbox"])
        bx, by = center(end["bbox"])
        later = next((row for row in rows if row["time"] >= end["time"] + 0.18), None)
        reversal = False
        if later is not None:
            cx, cy = center(later["bbox"])
            v1 = (bx - ax, by - ay)
            v2 = (cx - bx, cy - by)
            dist = math.hypot(*v1)
            dist2 = math.hypot(*v2)
            if dist > 1 and dist2 > 0.45 * dist and v1[0] * v2[0] + v1[1] * v2[1] < 0:
                reversal = True
        speeds.append(
            {
                "time": round((start["time"] + end["time"]) / 2, 3),
                "speed": math.hypot(bx - ax, by - ay) / dt,
                "track_id": start["track_id"],
                "reversal": reversal,
            }
        )
    return speeds


def find_motion_spikes(tracks: dict[int, list[dict]], frame_h: float) -> list[dict]:
    samples = []
    for rows in tracks.values():
        samples.extend(track_speeds(rows))
    if len(samples) < 4:
        return []
    speeds = [row["speed"] for row in samples]
    floor = 0.20 * frame_h
    cutoff = max(floor, percentile(speeds, 0.90))
    samples.sort(key=lambda row: row["time"])
    peaks = []
    for i, row in enumerate(samples):
        if row["reversal"] or row["speed"] < cutoff:
            continue
        prev_speed = samples[i - 1]["speed"] if i else 0
        next_speed = samples[i + 1]["speed"] if i + 1 < len(samples) else 0
        if row["speed"] >= prev_speed and row["speed"] >= next_speed:
            peaks.append(
                {
                    "type": "motion_spike",
                    "time": row["time"],
                    "score": round(min(0.9, row["speed"] / max(frame_h * 2.5, 1)), 3),
                    "track_id": row["track_id"],
                    "detail": f"track speed {row['speed']:.0f}px/s",
                }
            )
    return _suppress(peaks, 0.8)


def find_multi_cat(tracks: dict[int, list[dict]], frame_h: float) -> list[dict]:
    by_time: dict[int, list[dict]] = {}
    for rows in tracks.values():
        for row in rows:
            if row.get("conf", 1) < 0.6:
                continue
            by_time.setdefault(int(row["time"] * 5), []).append(row)
    chaotic = []
    for bucket, rows in sorted(by_time.items()):
        unique: list[dict] = []
        for row in rows:
            if any(row["track_id"] == kept["track_id"] for kept in unique):
                continue
            if any(_same_cat(row["bbox"], kept["bbox"]) for kept in unique):
                continue
            unique.append(row)
        if len(unique) < 2:
            continue
        speeds = []
        for row in unique:
            series = tracks[row["track_id"]]
            near = [item for item in track_speeds(series) if abs(item["time"] - row["time"]) <= 0.3]
            if near:
                speeds.append(max(item["speed"] for item in near))
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        if mean_speed < 0.2 * frame_h:
            continue
        chaotic.append(
            {
                "time": round(bucket / 5, 3),
                "cats": len(unique),
                "track_ids": [row["track_id"] for row in unique],
                "speed": mean_speed,
            }
        )
    if not chaotic:
        return []
    # Keep a run that lasts at least two buckets (~0.4s).
    runs: list[list[dict]] = []
    current: list[dict] = []
    for row in chaotic:
        if not current or row["time"] - current[-1]["time"] <= 0.35:
            current.append(row)
        else:
            runs.append(current)
            current = [row]
    if current:
        runs.append(current)
    events = []
    for run_rows in runs:
        if run_rows[-1]["time"] - run_rows[0]["time"] < 0.3 and len(run_rows) < 2:
            continue
        peak = max(run_rows, key=lambda row: (row["cats"], row["speed"]))
        events.append(
            {
                "type": "multi_cat",
                "time": peak["time"],
                "score": round(min(0.95, 0.55 + 0.1 * peak["cats"]), 3),
                "track_id": peak["track_ids"][0],
                "track_ids": peak["track_ids"],
                "detail": f"{peak['cats']} cats moving at once",
            }
        )
    return _suppress(events, 1.0)


def _split_body(a: list[float], b: list[float]) -> bool:
    """True when two boxes are touching halves of one cat, not two animals.

    A branch or a post often cuts one cat into a left box and a right box.
    Those halves share a vertical span and barely overlap horizontally.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw, bw = max(ax2 - ax1, 1), max(bx2 - bx1, 1)
    ah, bh = max(ay2 - ay1, 1), max(by2 - by1, 1)
    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    x_gap = max(ax1, bx1) - min(ax2, bx2)
    y_gap = max(ay1, by1) - min(ay2, by2)
    y_ratio = y_overlap / min(ah, bh)
    x_ratio = x_overlap / min(aw, bw)
    side_by_side = y_ratio > 0.55 and x_gap < 0.2 * max(aw, bw) and x_overlap < 0.3 * min(aw, bw)
    stacked = x_ratio > 0.55 and y_gap < 0.2 * max(ah, bh) and y_overlap < 0.3 * min(ah, bh)
    return side_by_side or stacked


def _same_cat(a: list[float], b: list[float]) -> bool:
    if _split_body(a, b):
        return True
    ax, ay = center(a)
    bx, by = center(b)
    width = max(a[2] - a[0], b[2] - b[0], 1)
    return math.hypot(ax - bx, ay - by) < 0.45 * width


def _suppress(events: list[dict], window: float) -> list[dict]:
    kept: list[dict] = []
    for event in sorted(events, key=lambda row: row["score"], reverse=True):
        if any(abs(event["time"] - prev["time"]) < window and event["type"] == prev["type"] for prev in kept):
            continue
        kept.append(event)
    return kept


def detect_events(payload: dict) -> list[dict]:
    width = float(payload["width"])
    height = float(payload["height"])
    duration = float(payload["duration_sec"])
    tracks = group_tracks(payload.get("detections") or [])
    events: list[dict] = []
    for rows in tracks.values():
        events.extend(find_jumps_and_fails(rows, height))
    events.extend(find_exits(tracks, width, height, duration))
    events.extend(find_motion_spikes(tracks, height))
    events.extend(find_multi_cat(tracks, height))
    anchors = [event["time"] for event in events if event["type"] in {"jump", "fail"}]
    events = [
        event
        for event in events
        if event["type"] != "motion_spike" or all(abs(event["time"] - anchor) > 0.5 for anchor in anchors)
    ]
    for event in events:
        event.pop("rise", None)
        event.pop("start_time", None)
        event.pop("end_time", None)
    events.sort(key=lambda row: row["time"])
    return events


def build_clips(events: list[dict], duration: float) -> list[dict]:
    if not events:
        return []
    ranked = sorted(events, key=lambda row: row["score"], reverse=True)
    chosen_events: list[dict] = []
    for event in ranked:
        pre, post = PAD[event["type"]]
        start = max(0.0, event["time"] - pre)
        end = min(duration, event["time"] + post)
        if any(not (end < kept["start"] - 0.2 or start > kept["end"] + 0.2) for kept in chosen_events):
            # Fold this label into the overlapping clip later.
            continue
        chosen_events.append({**event, "start": start, "end": end, "types": [event["type"]]})
        if len(chosen_events) == 4:
            break
    for event in events:
        for clip in chosen_events:
            if clip["start"] - 0.15 <= event["time"] <= clip["end"] + 0.15:
                if event["type"] not in clip["types"]:
                    clip["types"].append(event["type"])
                if event["score"] > clip["score"]:
                    clip["score"] = event["score"]
                    clip["type"] = event["type"]
                    clip["time"] = event["time"]
                    clip["detail"] = event["detail"]
    clips = []
    for clip in sorted(chosen_events, key=lambda row: row["start"]):
        start, end = _fit_window(clip["start"], clip["end"], clip["time"], duration)
        primary = _primary_type(clip["types"], clip["type"])
        clips.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "time": clip["time"],
                "primary_type": primary,
                "types": clip["types"],
                "score": clip["score"],
                "track_id": clip.get("track_id"),
                "detail": clip.get("detail", ""),
            }
        )
    return clips


def _primary_type(types: list[str], fallback: str) -> str:
    if "fail" in types and "jump" in types:
        return "fail"
    return fallback if fallback in types else types[0]


def _fit_window(start: float, end: float, anchor: float, duration: float) -> tuple[float, float]:
    if end - start < 1.8:
        extra = 1.8 - (end - start)
        start -= extra / 2
        end += extra / 2
    if end - start > 6.0:
        start = anchor - 2.2
        end = anchor + 2.4
    start = max(0.0, start)
    end = min(duration, end)
    if end - start < 1.2 and duration > 0:
        end = min(duration, start + 1.2)
        start = max(0.0, end - 1.2)
    return start, end


def cut_clip(video: Path, start: float, end: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.4, end - start)
    audio = has_audio(video)

    def command(crf: str, long_side: int, with_audio: bool) -> list[str]:
        scale = (
            f"scale='if(gte(iw,ih),{long_side},-2)':'if(gte(iw,ih),-2,{long_side})'"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-vf",
            scale,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            crf,
            "-movflags",
            "+faststart",
        ]
        if with_audio and audio:
            cmd.extend(["-c:a", "aac", "-ac", "1", "-b:a", "48k"])
        else:
            cmd.append("-an")
        cmd.append(str(dest))
        return cmd

    proc = run(command("30", 640, True))
    if proc.returncode != 0:
        proc = run(command("30", 640, False))
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {proc.stderr.strip()}")
    if dest.stat().st_size > 900_000:
        proc = run(command("33", 480, False))
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg recompress failed: {proc.stderr.strip()}")


def write_outputs(video: Path, events_path: Path, out_dir: Path) -> dict:
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    moments = detect_events(payload)
    source = vlm_status()
    for index, moment in enumerate(moments):
        moment["host_line"] = host_line(moment["type"], moment["time"], index)
        moment["commentary_source"] = source
    clips = build_clips(moments, float(payload["duration_sec"]))
    clip_dir = out_dir / "highlights"
    for old in clip_dir.glob("hl_*.mp4"):
        old.unlink()
    written = []
    for index, clip in enumerate(clips, start=1):
        name = f"hl_{index:02d}_{clip['primary_type']}_{clip['start']:.2f}-{clip['end']:.2f}.mp4"
        dest = clip_dir / name
        cut_clip(video, clip["start"], clip["end"], dest)
        line_type = clip["primary_type"]
        line = host_line(line_type, clip["time"], index)
        if "jump" in clip["types"] and "fail" in clip["types"]:
            line = f"[{stamp(clip['time'])}] {TEMPLATES['fail'][0]}"
        written.append(
            {
                **clip,
                "file": f"highlights/{name}",
                "bytes": dest.stat().st_size,
                "host_line": line,
                "commentary_source": source,
            }
        )
    result = {
        "video": video.as_posix(),
        "events_file": events_path.as_posix(),
        "duration_sec": payload["duration_sec"],
        "commentary_source": source,
        "events": moments,
        "clips": written,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "highlights.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {video.name}", f"commentary: {source}", ""]
    if not written:
        lines.append("No highlight rules fired on the cat tracks in this clip.")
    for clip in written:
        kinds = ", ".join(clip["types"])
        lines.append(
            f"{clip['host_line']}  ({kinds}, {clip['start']:.2f}-{clip['end']:.2f}s, {clip['file']})"
        )
    (out_dir / "commentary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _box(cx: float, cy: float, bw: float = 100, bh: float = 160) -> list[float]:
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def _det(time_sec: float, track_id: int, box: list[float], conf: float = 0.9) -> dict:
    return {
        "frame": int(time_sec * 30),
        "time": round(time_sec, 3),
        "track_id": track_id,
        "conf": conf,
        "bbox": box,
    }


def self_test() -> None:
    width, height = 1280, 720
    dets = []
    for step in range(0, 10):
        t = step * 0.1
        dets.append(_det(t, 1, _box(400, 500)))
    for step in range(1, 5):
        t = 1.0 + step * 0.075
        cy = 500 - step * 55
        dets.append(_det(t, 1, _box(400, cy)))
    for step in range(1, 5):
        t = 1.35 + step * 0.06
        cy = 280 + step * 60
        dets.append(_det(t, 1, _box(420, cy)))
    for step in range(0, 8):
        t = 3.0 + step * 0.1
        cx = 500 - step * 70
        box = _box(cx, 400)
        if step == 7:
            box = [0, 320, 70, 480]
        dets.append(_det(t, 1, box))
    # A second cat dashes back in from the left after the exit.
    for step in range(0, 6):
        t = 4.2 + step * 0.1
        box = [0, 300, 80, 460] if step == 0 else _box(120 + step * 40, 380)
        dets.append(_det(t, 2, box))
    payload = {
        "width": width,
        "height": height,
        "duration_sec": 8.0,
        "detections": dets,
        "motion": [],
    }
    events = detect_events(payload)
    types = {event["type"] for event in events}
    missing = {"jump", "fail", "exit", "reentry"} - types
    if missing:
        raise SystemExit(f"self-test missing {sorted(missing)} in {events}")
    clips = build_clips(events, 8.0)
    if not clips:
        raise SystemExit("self-test built no clips")

    dash = []
    for step in range(0, 16):
        t = step * 0.1
        if step < 4:
            cx = 200
        elif step < 10:
            cx = 200 + (step - 4) * 70
        else:
            cx = 200 + 6 * 70
        dash.append(_det(t, 3, _box(cx, 360)))
    dash_events = detect_events(
        {"width": width, "height": height, "duration_sec": 2.0, "detections": dash, "motion": []}
    )
    if not any(event["type"] == "motion_spike" for event in dash_events):
        raise SystemExit(f"self-test missing motion spike: {dash_events}")

    still = [_det(step * 0.1, 4, _box(400, 400)) for step in range(20)]
    still_events = detect_events(
        {"width": width, "height": height, "duration_sec": 2.0, "detections": still, "motion": []}
    )
    if any(event["type"] in {"jump", "fail", "motion_spike"} for event in still_events):
        raise SystemExit(f"still cat was called action: {still_events}")

    pair = []
    for step in range(10):
        t = 1 + step * 0.1
        pair.append(_det(t, 9, _box(150 + step * 25, 360)))
        pair.append(_det(t, 10, _box(900 - step * 20, 380)))
    pair_events = detect_events(
        {"width": width, "height": height, "duration_sec": 3.0, "detections": pair, "motion": []}
    )
    if not any(event["type"] == "multi_cat" for event in pair_events):
        raise SystemExit(f"self-test missing multi-cat: {pair_events}")

    split = []
    for step in range(10):
        t = 1 + step * 0.1
        dx = step * 12
        split.append(_det(t, 11, [180 + dx, 280, 340 + dx, 560]))
        split.append(_det(t, 12, [352 + dx, 300, 490 + dx, 540]))
    split_events = detect_events(
        {"width": width, "height": height, "duration_sec": 3.0, "detections": split, "motion": []}
    )
    if any(event["type"] == "multi_cat" for event in split_events):
        raise SystemExit(f"split cat was called multi-cat: {split_events}")

    duck = []
    for step in range(10):
        t = 1 + step * 0.1
        duck.append(_det(t, 13, _box(400 + step * 20, 400)))
        duck.append(_det(t, 14, _box(120, 420), conf=0.4))
    duck_events = detect_events(
        {"width": width, "height": height, "duration_sec": 3.0, "detections": duck, "motion": []}
    )
    if any(event["type"] == "multi_cat" for event in duck_events):
        raise SystemExit(f"low-conf second box was called multi-cat: {duck_events}")

    print("self-test ok", sorted(types | {"multi_cat"}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut cat highlight clips from events.json")
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.video is None or args.out is None:
        parser.error("video and --out are required")
    events_path = args.events or (args.out / "events.json")
    if not args.video.is_file():
        print(f"missing video: {args.video}", file=sys.stderr)
        return 1
    if not events_path.is_file():
        print(f"missing events: {events_path}", file=sys.stderr)
        return 1
    result = write_outputs(args.video, events_path, args.out)
    print(
        f"{args.video.name}: {len(result['events'])} events, "
        f"{len(result['clips'])} clips, commentary {result['commentary_source']}"
    )
    for clip in result["clips"]:
        print(
            f"  {clip['primary_type']:12} {clip['start']:.2f}-{clip['end']:.2f}s "
            f"{clip['bytes'] // 1024}KB  {clip['host_line']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
