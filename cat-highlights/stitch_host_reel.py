#!/usr/bin/env python3
"""Stitch short cat clips and burn text cues from the highlight pipeline.

The event layer is the existing pipeline (keyframes, events.json, highlights).
This script places three on-screen lines and does not add a voice track:

- ENTERTAINMENT when the motion score or the scene score rises
- COMMENT, the host line, at the event
- NEXT, an anticipation line written from that same event, 1.1s earlier

Source mp4s and the stitched reel stay out of git. Pass --work for scratch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from extract_keyframes import _parse_scores
from media_info import ffprobe_video, has_audio, run

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
CANVAS_W = 1280
CANVAS_H = 720
FPS = 30
BLACK_SEC = 0.4
LEAD_SEC = 1.1
NEXT_CLEAR = 0.12
COMMENT_HOLD = 1.85
ENT_HOLD = 1.45

ANTICIPATION = {
    "jump": "he's going for it",
    "fail": "landing won't stick",
    "exit": "he's about to bail",
    "reentry": "he's coming back",
    "motion_spike": "watch this",
    "multi_cat": "two of them now",
}

RANK = {
    "fail": 5,
    "jump": 4,
    "multi_cat": 4,
    "exit": 3,
    "reentry": 3,
    "motion_spike": 1,
}

STYLE = {
    "entertainment": {
        "fontsize": 30,
        "fontcolor": "0xFFE566",
        "boxcolor": "black@0.75",
        "x": "48",
        "y": "40",
    },
    "next": {
        "fontsize": 32,
        "fontcolor": "0x9BE7FF",
        "boxcolor": "0x062430@0.82",
        "x": "w-tw-48",
        "y": "40",
    },
    "comment": {
        "fontsize": 36,
        "fontcolor": "white",
        "boxcolor": "black@0.78",
        "x": "(w-tw)/2",
        "y": "h-th-48",
    },
}


def stamp(time_sec: float) -> str:
    minutes = int(time_sec // 60)
    seconds = time_sec - minutes * 60
    return f"{minutes}:{seconds:04.1f}"


def _ffmpeg(cmd: list[str]) -> None:
    proc = run(cmd)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = "\n".join(tail[-20:])
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {detail}")


def normalize_clip(src: Path, dest: Path) -> float:
    """Fit one clip on the 1280x720 canvas and give it a stereo track."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = f"{ffprobe_video(src)['duration_sec']:.3f}"
    video = (
        f"scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=decrease,"
        f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={FPS},format=yuv420p,setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-f",
        "lavfi",
        "-t",
        duration,
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
    ]
    if has_audio(src):
        cmd += [
            "-filter_complex",
            f"[0:v]{video}[v];[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
        ]
    else:
        cmd += [
            "-vf",
            video,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
        ]
    cmd += [
        "-t",
        duration,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    _ffmpeg(cmd)
    return ffprobe_video(dest)["duration_sec"]


def black_beat(dest: Path) -> float:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={CANVAS_W}x{CANVAS_H}:r={FPS}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            f"{BLACK_SEC:.3f}",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )
    return ffprobe_video(dest)["duration_sec"]


def concat_copy(parts: list[Path], dest: Path, list_file: Path) -> None:
    lines = [f"file '{path.resolve()}'" for path in parts]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )


def stitch(manifest: dict, work: Path, dest: Path, timeline_path: Path) -> dict:
    clips = manifest.get("clips") or []
    if not 3 <= len(clips) <= 5:
        raise SystemExit(f"need 3 to 5 clips, got {len(clips)}")
    parts_dir = work / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    black = parts_dir / "black.mp4"
    black_len = black_beat(black)
    parts: list[Path] = []
    segments = []
    beats = []
    cursor = 0.0
    for index, clip in enumerate(clips):
        src = Path(clip["file"])
        if not src.is_file():
            raise SystemExit(f"missing clip: {src}")
        part = parts_dir / f"seg_{index:02d}.mp4"
        duration = normalize_clip(src, part)
        parts.append(part)
        segments.append(
            {
                "index": index + 1,
                "id": clip["id"],
                "title": clip["title"],
                "url": clip["url"],
                "moment": clip.get("moment", ""),
                "start": round(cursor, 3),
                "end": round(cursor + duration, 3),
                "duration_sec": round(duration, 3),
            }
        )
        cursor += duration
        if index < len(clips) - 1:
            parts.append(black)
            beats.append({"start": round(cursor, 3), "end": round(cursor + black_len, 3)})
            cursor += black_len
    dest.parent.mkdir(parents=True, exist_ok=True)
    concat_copy(parts, dest, work / "concat.txt")
    probed = ffprobe_video(dest)
    timeline = {
        "video": str(dest),
        "width": probed["width"],
        "height": probed["height"],
        "fps": round(probed["fps"], 3),
        "duration_sec": round(probed["duration_sec"], 3),
        "black_sec": round(black_len, 3),
        "audio": "source audio kept; no voice track added",
        "segments": segments,
        "beats": beats,
    }
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    return timeline


def _segment_at(timeline: dict, time_sec: float) -> dict | None:
    for segment in timeline["segments"]:
        if segment["start"] - 0.02 <= time_sec < segment["end"] + 0.02:
            return segment
    return None


def _near_beat(timeline: dict, time_sec: float) -> bool:
    """True on a black beat and on the frames the cut itself shakes."""
    for beat in timeline.get("beats") or []:
        if beat["start"] - 0.55 <= time_sec <= beat["end"] + 0.30:
            return True
    return False


def _smooth(values: list[float], radius: int) -> list[float]:
    if not values:
        return []
    out = []
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        window = values[lo:hi]
        out.append(sum(window) / len(window))
    return out


def entertainment_cues(timeline: dict, motion: list[dict], scene_rows: list[dict]) -> list[dict]:
    cues: list[dict] = []
    if motion:
        times = [float(row["time"]) for row in motion]
        raw = [float(row["score"]) for row in motion]
        span = times[-1] - times[0] if len(times) > 1 else 1.0
        step = span / max(len(times) - 1, 1)
        radius = max(2, int(round(0.2 / max(step, 1e-3))))
        lookback = max(3, int(round(0.5 / max(step, 1e-3))))
        smooth = _smooth(raw, radius)
        rises: list[tuple[float, float, float]] = []
        for index in range(lookback, len(smooth) - 1):
            baseline = smooth[index - lookback]
            rise = smooth[index] - baseline
            if rise < 0.02 or smooth[index] < 0.025:
                continue
            if smooth[index] < smooth[index - 1] or smooth[index] < smooth[index + 1]:
                continue
            when = times[index]
            if _near_beat(timeline, when):
                continue
            rises.append((rise, when, smooth[index]))
        rises.sort(reverse=True)
        kept: list[tuple[float, float, float]] = []
        for rise, when, level in rises:
            if any(abs(when - prev[1]) < 2.6 for prev in kept):
                continue
            kept.append((rise, when, level))
            if len(kept) == 6:
                break
        for rise, when, level in kept:
            label = "it spikes" if rise >= 0.06 or level >= 0.12 else "motion up"
            cues.append(_ent_cue(timeline, when, label, round(rise, 4), "motion"))
    scene_kept = 0
    for row in sorted(scene_rows, key=lambda item: item["score"], reverse=True):
        when = float(row["time"])
        score = float(row["score"])
        if score < 0.16 or _near_beat(timeline, when):
            continue
        if any(abs(when - cue["start"]) < 2.4 for cue in cues):
            continue
        cues.append(_ent_cue(timeline, when, "scene shift", round(score, 4), "scene"))
        scene_kept += 1
        if scene_kept == 3:
            break
    cues.sort(key=lambda cue: cue["start"])
    return cues[:8]


def _ent_cue(timeline: dict, when: float, label: str, score: float, source: str) -> dict:
    duration = float(timeline["duration_sec"])
    start = max(0.0, when)
    end = min(duration, start + ENT_HOLD)
    segment = _segment_at(timeline, start)
    return {
        "kind": "entertainment",
        "start": round(start, 3),
        "end": round(end, 3),
        "text": f"ENTERTAINMENT  {label}",
        "score": score,
        "source": source,
        "segment_id": segment["id"] if segment else None,
    }


def on_screen_comment(host_line: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\s*", "", host_line).strip()
    parts = [part for part in re.split(r"(?<=[.!?])\s+", text) if part]
    if not parts:
        return "COMMENT  look at this"
    sentence = parts[0]
    if len(parts) > 1 and len(sentence) < 12:
        sentence = parts[1]
    elif len(parts) > 1 and len(sentence) < 22:
        combined = f"{sentence} {parts[1]}"
        if len(combined) <= 48:
            sentence = combined
        elif len(parts[1]) > len(sentence):
            sentence = parts[1]
    if len(sentence) > 48:
        sentence = sentence[:45].rstrip(" ,;:") + "..."
    return f"COMMENT  {sentence}"


def choose_comments(highlights: dict, timeline: dict) -> list[dict]:
    moments = list(highlights.get("events") or [])
    picked: list[dict] = []
    for segment in timeline["segments"]:
        inside = [
            moment
            for moment in moments
            if segment["start"] - 0.05 <= float(moment["time"]) < segment["end"] - 0.05
        ]
        ranked = sorted(
            inside,
            key=lambda moment: (RANK.get(moment["type"], 0), float(moment.get("score") or 0)),
            reverse=True,
        )
        got: list[dict] = []
        has_strong = any(moment["type"] != "motion_spike" for moment in ranked)
        for moment in ranked:
            if len(got) >= 2:
                break
            if any(abs(float(moment["time"]) - float(prev["time"])) < 2.2 for prev in got):
                continue
            if moment["type"] == "motion_spike":
                spikes_kept = sum(1 for prev in got if prev["type"] == "motion_spike")
                if has_strong and spikes_kept >= 1:
                    continue
                if spikes_kept >= 2:
                    continue
            got.append(moment)
        if not got and ranked:
            got.append(ranked[0])
        picked.extend(got)
    picked.sort(key=lambda moment: float(moment["time"]))
    return picked


def _host_line_for(moment: dict, highlights: dict) -> str:
    """Prefer the line make_highlights wrote for the matching highlight clip."""
    when = float(moment["time"])
    for clip in highlights.get("clips") or []:
        if abs(float(clip.get("time", -999)) - when) <= 0.45 and clip.get("host_line"):
            return str(clip["host_line"]).strip()
    return str(moment.get("host_line") or "").strip()


def comment_and_next(timeline: dict, highlights: dict) -> list[dict]:
    duration = float(timeline["duration_sec"])
    moments = choose_comments(highlights, timeline)
    comments: list[dict] = []
    for index, moment in enumerate(moments):
        when = float(moment["time"])
        nxt = float(moments[index + 1]["time"]) if index + 1 < len(moments) else None
        hold = COMMENT_HOLD
        if nxt is not None:
            hold = min(hold, max(1.05, nxt - when - 0.2))
        end = min(duration, when + hold)
        segment = _segment_at(timeline, when)
        host = _host_line_for(moment, highlights)
        comments.append(
            {
                "kind": "comment",
                "start": round(when, 3),
                "end": round(end, 3),
                "text": on_screen_comment(host) if host else f"COMMENT  {moment['type']}",
                "host_line": host,
                "event_type": moment["type"],
                "event_time": round(when, 3),
                "score": moment.get("score"),
                "detail": moment.get("detail", ""),
                "segment_id": segment["id"] if segment else None,
            }
        )
    nxt_cues: list[dict] = []
    for comment in comments:
        event_time = float(comment["event_time"])
        start = event_time - LEAD_SEC
        if start < 0.05:
            continue
        lead = event_time - start
        if lead < 0.8 or lead > 1.5:
            continue
        end = min(comment["start"] - NEXT_CLEAR, start + 1.15)
        if end - start < 0.55:
            continue
        phrase = ANTICIPATION.get(comment["event_type"], "here it comes")
        segment = _segment_at(timeline, start)
        nxt_cues.append(
            {
                "kind": "next",
                "start": round(start, 3),
                "end": round(end, 3),
                "text": f"NEXT  {phrase}",
                "phrase": phrase,
                "event_type": comment["event_type"],
                "event_time": comment["event_time"],
                "lead_sec": round(lead, 3),
                "segment_id": segment["id"] if segment else None,
                "event_segment_id": comment["segment_id"],
                "score": comment.get("score"),
            }
        )
    kept: list[dict] = []
    for cue in sorted(nxt_cues, key=lambda item: (RANK.get(item["event_type"], 0), item.get("score") or 0), reverse=True):
        if any(not (cue["end"] <= prev["start"] + 0.05 or cue["start"] >= prev["end"] - 0.05) for prev in kept):
            continue
        kept.append(cue)
    kept.sort(key=lambda cue: cue["start"])
    return comments + kept


def fill_entertainment_from_spikes(timeline: dict, cues: list[dict], highlights: dict) -> None:
    """If a segment's frame diff stays flat, use the track-speed spike instead."""
    for segment in timeline["segments"]:
        if any(cue["kind"] == "entertainment" and cue.get("segment_id") == segment["id"] for cue in cues):
            continue
        spikes = [
            moment
            for moment in highlights.get("events") or []
            if moment.get("type") == "motion_spike"
            and segment["start"] <= float(moment["time"]) < segment["end"]
            and not _near_beat(timeline, float(moment["time"]))
        ]
        if not spikes:
            continue
        best = max(float(moment.get("score") or 0) for moment in spikes)
        early = [moment for moment in spikes if float(moment.get("score") or 0) >= 0.8 * best]
        chosen = min(early, key=lambda moment: float(moment["time"]))
        when = float(chosen["time"])
        if any(cue["kind"] == "entertainment" and abs(cue["start"] - when) < 2.0 for cue in cues):
            continue
        label = "it spikes" if float(chosen.get("score") or 0) >= 0.25 else "motion up"
        cues.append(_ent_cue(timeline, when, label, round(float(chosen.get("score") or 0), 4), "track"))


def build_cues(timeline: dict, highlights: dict, events: dict, scene_rows: list[dict]) -> list[dict]:
    cues = entertainment_cues(timeline, events.get("motion") or [], scene_rows)
    fill_entertainment_from_spikes(timeline, cues, highlights)
    cues.extend(comment_and_next(timeline, highlights))
    cues.sort(key=lambda cue: (cue["start"], cue["kind"]))
    return cues


def _drawtext(cue: dict, text_file: Path) -> str:
    style = STYLE[cue["kind"]]
    text_file.write_text(cue["text"] + "\n", encoding="utf-8")
    start = f"{cue['start']:.3f}"
    end = f"{cue['end']:.3f}"
    return (
        "drawtext="
        f"fontfile={FONT}:"
        f"textfile={text_file.resolve()}:"
        f"enable=between(t\\,{start}\\,{end}):"
        f"x={style['x']}:y={style['y']}:"
        f"fontsize={style['fontsize']}:"
        f"fontcolor={style['fontcolor']}:"
        "box=1:"
        f"boxcolor={style['boxcolor']}:"
        "boxborderw=14"
    )


def burn(video: Path, cues: list[dict], dest: Path, work: Path) -> None:
    if not FONT.is_file():
        raise SystemExit(f"missing font {FONT}; install fonts-dejavu-core")
    if not cues:
        raise SystemExit("no cues to burn")
    text_dir = work / "cue-text"
    text_dir.mkdir(parents=True, exist_ok=True)
    filters = [_drawtext(cue, text_dir / f"cue_{index:02d}.txt") for index, cue in enumerate(cues)]
    script = work / "drawtext.txt"
    script.write_text(",\n".join(filters) + "\n", encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-filter_script:v",
            str(script),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )


def write_rundown(path: Path, timeline: dict, cues: list[dict], note: str) -> None:
    entertainment = [cue for cue in cues if cue["kind"] == "entertainment"]
    comments = [cue for cue in cues if cue["kind"] == "comment"]
    nxt = [cue for cue in cues if cue["kind"] == "next"]
    lines = [
        "# Stitched host reel",
        "",
        note.strip(),
        "",
        f"Duration {stamp(timeline['duration_sec'])} ({timeline['duration_sec']:.2f}s). "
        f"Canvas {timeline['width']}x{timeline['height']}. "
        f"Black beat {timeline['black_sec']:.2f}s between clips. "
        "Text only: no voice track was added. Source audio from the clips is still in the file.",
        "",
        "## Clips",
        "",
        "| # | On the reel | Moment | Post |",
        "| --- | --- | --- | --- |",
    ]
    for segment in timeline["segments"]:
        window = f"{stamp(segment['start'])}–{stamp(segment['end'])}"
        lines.append(
            f"| {segment['index']} | {window} | {segment['moment']} | [{segment['title']}]({segment['url']}) |"
        )
    lines += ["", "## Entertainment rises", "", "A rise in the pipeline motion score or scene score. The label is on screen at the rise.", ""]
    if not entertainment:
        lines.append("None. The motion and scene scores stayed flat outside the black beats.")
    for cue in entertainment:
        lines.append(
            f"- {stamp(cue['start'])}–{stamp(cue['end'])}  `{cue['text']}`  "
            f"({cue['source']} {cue['score']}, segment {cue.get('segment_id') or '—'})"
        )
    lines += ["", "## Host comments", "", "The pipeline host line, shown at the event.", ""]
    if not comments:
        lines.append("None. The highlight rules did not fire.")
    for cue in comments:
        lines.append(
            f"- {stamp(cue['start'])}–{stamp(cue['end'])}  `{cue['text']}`  "
            f"({cue['event_type']}, segment {cue.get('segment_id') or '—'})"
        )
        if cue.get("host_line"):
            lines.append(f"  - pipeline: {cue['host_line']}")
        if cue.get("detail"):
            lines.append(f"  - detail: {cue['detail']}")
    lines += [
        "",
        "## Anticipation",
        "",
        "Each line is written from an event the pipeline already found, and it comes up about 1.1s before that event.",
        "",
    ]
    if not nxt:
        lines.append("None. No event had a full 0.8s of lead inside the reel.")
    for cue in nxt:
        lines.append(
            f"- on screen {stamp(cue['start'])}–{stamp(cue['end'])}  `{cue['text']}`  "
            f"calls the {cue['event_type']} at {stamp(cue['event_time'])} "
            f"(lead {cue['lead_sec']:.2f}s, segment {cue.get('event_segment_id') or '—'})"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_scene_rows(analysis: Path) -> list[dict]:
    score_file = analysis / "scene_scores.txt"
    if not score_file.is_file():
        return []
    return _parse_scores(score_file.read_text(encoding="utf-8", errors="replace"))


def self_test() -> None:
    timeline = {
        "duration_sec": 20.0,
        "segments": [
            {"id": "door", "start": 0.0, "end": 8.0},
            {"id": "chase", "start": 8.4, "end": 20.0},
        ],
        "beats": [{"start": 8.0, "end": 8.4}],
    }
    motion = [{"time": round(index * 0.1, 3), "score": 0.01} for index in range(200)]
    for index, time_sec in enumerate([t * 0.1 for t in range(200)]):
        if 4.6 <= time_sec <= 5.4:
            motion[index]["score"] = 0.02 + (time_sec - 4.6) * 0.2
    scene_rows = [
        {"time": 5.0, "score": 0.04, "frame": 150},
        {"time": 8.05, "score": 0.8, "frame": 241},
        {"time": 12.0, "score": 0.4, "frame": 360},
    ]
    highlights = {
        "events": [
            {
                "type": "jump",
                "time": 6.2,
                "score": 0.4,
                "host_line": "[0:06.2] The orange one commits. Paws leave the floor.",
                "detail": "center rose 80px in 0.20s",
            },
            {
                "type": "motion_spike",
                "time": 6.4,
                "score": 0.2,
                "host_line": "[0:06.4] The orange one breaks the pose.",
                "detail": "track speed 200px/s",
            },
            {
                "type": "exit",
                "time": 14.0,
                "score": 0.3,
                "host_line": "[0:14.0] Offscreen. The orange one declines a second take.",
                "detail": "left through the right edge",
            },
        ]
    }
    cues = build_cues(timeline, highlights, {"motion": motion}, scene_rows)
    entertainment = [cue for cue in cues if cue["kind"] == "entertainment"]
    comments = [cue for cue in cues if cue["kind"] == "comment"]
    nxt = [cue for cue in cues if cue["kind"] == "next"]
    if any(_near_beat(timeline, cue["start"]) for cue in entertainment):
        raise SystemExit("self-test: entertainment landed on a black beat")
    if not any(4.2 <= cue["start"] <= 5.6 for cue in entertainment):
        raise SystemExit("self-test: missed the motion rise")
    if not any(abs(cue["start"] - 12.0) < 0.05 and cue["source"] == "scene" for cue in entertainment):
        raise SystemExit("self-test: missed the scene rise")
    jump = next(cue for cue in comments if cue["event_type"] == "jump")
    if "commits" not in jump["text"]:
        raise SystemExit(f"self-test: comment text lost the host line: {jump['text']}")
    jump_next = next(cue for cue in nxt if cue["event_type"] == "jump")
    if abs(jump_next["lead_sec"] - LEAD_SEC) > 0.02:
        raise SystemExit("self-test: anticipation lead is not 1.1s")
    if not (0.8 <= jump_next["lead_sec"] <= 1.5):
        raise SystemExit("self-test: lead outside 0.8-1.5s")
    if jump_next["end"] >= jump["start"]:
        raise SystemExit("self-test: anticipation overlaps its comment")
    if "going for it" not in jump_next["text"]:
        raise SystemExit("self-test: jump anticipation line missing")
    exit_comment = next(cue for cue in comments if cue["event_type"] == "exit")
    if "second take" not in exit_comment["text"]:
        raise SystemExit("self-test: short first sentence was not expanded")
    print(f"self-test ok: {len(entertainment)} entertainment, {len(comments)} comments, {len(nxt)} next")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stitch cat clips and burn host text")
    parser.add_argument("command", choices=["stitch", "overlay", "self-test"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--cues", type=Path)
    parser.add_argument("--rundown", type=Path)
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    if args.command == "stitch":
        if not args.manifest or not args.out or not args.timeline:
            raise SystemExit("stitch needs --manifest, --out, and --timeline")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        timeline = stitch(manifest, args.work, args.out, args.timeline)
        print(
            f"stitched {len(timeline['segments'])} clips, "
            f"{timeline['duration_sec']:.2f}s -> {args.out}"
        )
        for segment in timeline["segments"]:
            print(f"  {segment['id']:16} {segment['start']:7.2f}-{segment['end']:7.2f}  {segment['url']}")
        return 0
    if not args.video or not args.analysis or not args.timeline or not args.out:
        raise SystemExit("overlay needs --video, --analysis, --timeline, and --out")
    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    analysis = args.analysis
    highlights = json.loads((analysis / "highlights.json").read_text(encoding="utf-8"))
    events = json.loads((analysis / "events.json").read_text(encoding="utf-8"))
    cues = build_cues(timeline, highlights, events, load_scene_rows(analysis))
    cues_path = args.cues or (args.work / "cues.json")
    cues_path.parent.mkdir(parents=True, exist_ok=True)
    cues_path.write_text(json.dumps(cues, indent=2) + "\n", encoding="utf-8")
    print(f"{len(cues)} cues -> {cues_path}")
    for cue in cues:
        print(f"  {cue['kind']:14} {cue['start']:7.2f}-{cue['end']:7.2f}  {cue['text']}")
    burn(args.video, cues, args.out, args.work)
    if args.rundown:
        write_rundown(args.rundown, timeline, cues, args.note)
        print(f"rundown -> {args.rundown}")
    probed = ffprobe_video(args.out)
    print(f"reel {args.out} {probed['width']}x{probed['height']} {probed['duration_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
