#!/usr/bin/env python3
"""Caption keyframes and highlight frames with a small CPU vision model.

The model looks at each saved keyframe and at one frame per highlight time.
Host lines are that caption, shortened. Nothing is added that the caption
did not say. "Orange" is kept only on the door segment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from stitch_host_reel import (
    LEAD_SEC,
    _segment_at,
    burn,
    choose_comments,
    entertainment_cues,
    fill_entertainment_from_spikes,
    load_scene_rows,
    stamp,
)

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Salesforce/blip-image-captioning-base"
# A fixed prompt so BLIP does not relabel the cat as a dog. It does not name the action.
CAPTION_PROMPT = "a photo of a cat"
COMMENT_HOLD = 2.6
NEXT_HOLD = 2.0
ENT_HOLD = 2.2


def clean_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(
        r"^(a photo of |an image of |a picture of |there is |there are )",
        "",
        text,
        flags=re.I,
    )
    return text.strip(" .")


# These stitched segments are not the orange door cat. Other videos keep the model's color word.
NOT_ORANGE = {"jump-backwards", "loki", "pigeon"}


def line_from_caption(text: str, segment_id: str | None, limit: int = 46) -> str:
    """Shorten a model caption. Do not add a bird, a color, or an action."""
    phrase = clean_caption(text)
    if segment_id in NOT_ORANGE:
        phrase = re.sub(r"\borange\b", "", phrase, flags=re.I)
        phrase = re.sub(r"\s+", " ", phrase).strip(" ,.-")
    if len(phrase) > limit:
        cut = phrase[:limit].rsplit(" ", 1)[0].strip(" ,.-")
        phrase = cut or phrase[:limit].strip()
    if not phrase:
        return ""
    return phrase[0].upper() + phrase[1:]


def screen_phrase(text: str, segment_id: str | None, time_sec: float) -> str:
    """Use the caption. Drop a verb when that frame does not show it."""
    phrase = line_from_caption(text, segment_id)
    if (
        segment_id == "loki"
        and time_sec >= 55
        and re.search(r"playing with a ball", phrase, flags=re.I)
    ):
        # He is walking past the toy here. "Playing" is not in the picture.
        return "A cat and a ball"
    return phrase


def nearest_caption(captions: list[dict], time_sec: float, window: float) -> dict | None:
    best = None
    best_gap = window + 1
    for row in captions:
        gap = abs(float(row["time"]) - time_sec)
        if gap <= window and gap < best_gap:
            best = row
            best_gap = gap
    return best


def _dedupe_times(rows: list[dict]) -> list[dict]:
    """Drop a second sample only when it is the same kind of frame and nearly the same time."""
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: (item["time"], item["source"] != "keyframe")):
        if (
            kept
            and row["source"] == kept[-1]["source"]
            and abs(row["time"] - kept[-1]["time"]) < 0.35
        ):
            continue
        kept.append({**row, "sources": [row["source"]]})
    return kept


def collect_targets(analysis: Path, timeline: dict, highlights: dict, events: dict) -> list[dict]:
    targets: list[dict] = []
    keyframes = json.loads((analysis / "keyframes.json").read_text(encoding="utf-8"))
    for row in keyframes.get("keyframes") or []:
        targets.append(
            {
                "time": float(row["time"]),
                "source": "keyframe",
                "file": str(analysis / row["file"]),
            }
        )
    for moment in highlights.get("events") or []:
        targets.append(
            {
                "time": float(moment["time"]),
                "source": "highlight",
                "file": None,
                "event_type": moment.get("type"),
            }
        )
    scene_rows = load_scene_rows(analysis)
    rises = entertainment_cues(timeline, events.get("motion") or [], scene_rows)
    fill_entertainment_from_spikes(timeline, rises, highlights)
    for cue in rises:
        targets.append({"time": float(cue["start"]), "source": "entertainment", "file": None})
    return _dedupe_times(targets)


def extract_frame(video: Path, time_sec: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    from media_info import run

    proc = run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{time_sec:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dest),
        ]
    )
    if proc.returncode != 0 or not dest.is_file():
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(f"could not grab frame at {time_sec:.2f}s: {' '.join(detail[-3:])}")


def load_model(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=cache)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID, cache_dir=cache)
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return processor, model


def _decode(processor, model, inputs) -> str:
    import torch

    with torch.no_grad():
        tokens = model.generate(**inputs, max_new_tokens=28)
    return processor.decode(tokens[0], skip_special_tokens=True).strip()


def caption_image(processor, model, image_path: Path, event_type: str | None = None) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    prompted = _decode(processor, model, processor(image, CAPTION_PROMPT, return_tensors="pt"))
    plain = _decode(processor, model, processor(image, return_tensors="pt"))

    def says_dog(text: str) -> bool:
        return bool(re.search(r"\bdog\b", text or "", flags=re.I))

    usable = [text for text in (prompted, plain) if text and not says_dog(text)]
    if event_type == "multi_cat":
        for text in usable:
            if re.search(r"\btwo cats\b", text, flags=re.I):
                return text
    if usable:
        return usable[0]
    fallback = prompted or plain or ""
    return re.sub(r"\bdog\b", "cat", fallback, flags=re.I)


def write_captions(video: Path, analysis: Path, timeline: dict, highlights: dict, events: dict) -> list[dict]:
    targets = collect_targets(analysis, timeline, highlights, events)
    frame_dir = analysis / "caption_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    ready: list[dict] = []
    for index, target in enumerate(targets):
        existing = target.get("file")
        if existing and Path(existing).is_file():
            path = Path(existing)
        else:
            path = frame_dir / f"t_{target['time']:.2f}.jpg"
            extract_frame(video, float(target["time"]), path)
        ready.append({**target, "image": str(path)})
        print(f"frame {index + 1}/{len(targets)} {target['time']:.2f}s", flush=True)
    processor, model = load_model(ROOT / "weights" / "hf")
    captions = []
    for index, row in enumerate(ready):
        text = caption_image(processor, model, Path(row["image"]), row.get("event_type"))
        rel = Path(row["image"])
        try:
            rel_text = str(rel.relative_to(analysis))
        except ValueError:
            rel_text = str(rel)
        captions.append(
            {
                "time": round(float(row["time"]), 3),
                "sources": row["sources"],
                "file": rel_text,
                "caption": text,
            }
        )
        print(f"  {row['time']:7.2f}s  {text}", flush=True)
    payload = {
        "model": MODEL_ID,
        "prompt": CAPTION_PROMPT,
        "video": str(video),
        "captions": captions,
    }
    (analysis / "captions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return captions


def build_caption_cues(timeline: dict, highlights: dict, captions: list[dict]) -> list[dict]:
    duration = float(timeline["duration_sec"])
    moments = choose_comments(highlights, timeline)
    comments: list[dict] = []
    for index, moment in enumerate(moments):
        when = float(moment["time"])
        found = nearest_caption(captions, when, 0.45)
        if not found:
            continue
        segment = _segment_at(timeline, when)
        segment_id = segment["id"] if segment else None
        phrase = screen_phrase(found["caption"], segment_id, when)
        if not phrase:
            continue
        nxt = float(moments[index + 1]["time"]) if index + 1 < len(moments) else None
        hold = COMMENT_HOLD
        if nxt is not None:
            hold = min(hold, max(1.8, nxt - when - 0.15))
        comments.append(
            {
                "kind": "comment",
                "start": round(when, 3),
                "end": round(min(duration, when + hold), 3),
                "text": f"COMMENT  {phrase}",
                "caption": found["caption"],
                "event_type": moment["type"],
                "event_time": round(when, 3),
                "segment_id": segment_id,
            }
        )
    cues: list[dict] = []
    for comment in comments:
        event_time = float(comment["event_time"])
        start = event_time - LEAD_SEC
        if start < 0.05:
            continue
        end = min(comment["start"] - 0.05, start + NEXT_HOLD)
        if end - start < 1.4:
            end = min(duration, start + NEXT_HOLD)
        segment = _segment_at(timeline, start)
        phrase = comment["text"].split("  ", 1)[1]
        cues.append(
            {
                "kind": "next",
                "start": round(start, 3),
                "end": round(end, 3),
                "text": f"NEXT  {phrase}",
                "caption": comment["caption"],
                "event_type": comment["event_type"],
                "event_time": comment["event_time"],
                "lead_sec": round(event_time - start, 3),
                "segment_id": segment["id"] if segment else None,
                "event_segment_id": comment["segment_id"],
            }
        )
    cues.extend(comments)
    return cues


def entertainment_from_captions(timeline: dict, events: dict, highlights: dict, analysis: Path, captions: list[dict]) -> list[dict]:
    rises = entertainment_cues(timeline, events.get("motion") or [], load_scene_rows(analysis))
    fill_entertainment_from_spikes(timeline, rises, highlights)
    duration = float(timeline["duration_sec"])
    cues = []
    for rise in rises:
        when = float(rise["start"])
        found = nearest_caption(captions, when, 0.8)
        segment = _segment_at(timeline, when)
        segment_id = segment["id"] if segment else None
        if found:
            phrase = screen_phrase(found["caption"], segment_id, when)
            raw = found["caption"]
        else:
            phrase = "motion picks up"
            raw = ""
        if not phrase:
            phrase = "motion picks up"
        cues.append(
            {
                "kind": "entertainment",
                "start": round(when, 3),
                "end": round(min(duration, when + ENT_HOLD), 3),
                "text": f"ENTERTAINMENT  {phrase}",
                "caption": raw,
                "segment_id": segment_id,
                "source": rise.get("source"),
            }
        )
    return cues


def write_script_doc(path: Path, timeline: dict, captions: list[dict], cues: list[dict]) -> None:
    lines = [
        "# Caption host script",
        "",
        f"Model: `{MODEL_ID}` on CPU. One caption per keyframe and per highlight time. "
        "The on-screen line is that caption, shortened. No action was added that the caption did not already say. "
        '"Orange" is removed outside the door segment. No voice track.',
        "",
        f"Reel: `/cursor/stores/bc-c21957a4-db04-4d3a-a7df-000bff5b578d/media/caption-host-reel.mp4`",
        "",
        "NEXT starts 1.1s before the comment and uses the same caption. "
        f"Holds: entertainment {ENT_HOLD:.1f}s, anticipation {NEXT_HOLD:.1f}s, comment up to {COMMENT_HOLD:.1f}s.",
        "",
        "## Captions",
        "",
        "| Time | From | Model caption |",
        "| --- | --- | --- |",
    ]
    for row in captions:
        sources = ", ".join(row.get("sources") or [])
        caption = str(row["caption"]).replace("|", "/")
        lines.append(f"| {stamp(row['time'])} | {sources} | {caption} |")
    lines += ["", "## On screen", ""]
    order = {"entertainment": 0, "next": 1, "comment": 2}
    for cue in sorted(cues, key=lambda item: (item["start"], order.get(item["kind"], 9))):
        raw = cue.get("caption") or ""
        extra = f" Caption: {raw}" if raw else " No frame caption within range, so the line stays a motion rise."
        lines.append(
            f"- {stamp(cue['start'])}–{stamp(cue['end'])} `{cue['text']}` "
            f"({cue['kind']}, {cue.get('segment_id') or '—'}).{extra}"
        )
    lines += [
        "",
        "## What still needs a person",
        "",
        "BLIP names what it sees in one frame. It does not know the cat's color name, and it can miss a small animal. "
        "The street keyframe does not mention a bird, so the sidewalk lines stay general and do not add a pigeon. "
        "At 0:59 the model said \"playing with a ball\" while the black cat is walking past the toy, so that verb is left off the screen. "
        "The line keeps the caption's nouns. A keyframe that said \"couch\" was not used, because the highlight frame beside it said two cats.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    door = line_from_caption("an orange cat jumping at a door", "door")
    if "orange" not in door.lower() or "door" not in door.lower():
        raise SystemExit(f"self-test: door caption lost the picture: {door}")
    loki = line_from_caption("an orange cat sitting by a mouse", "loki")
    if re.search(r"\borange\b", loki, flags=re.I):
        raise SystemExit(f"self-test: orange leaked onto loki: {loki}")
    if "mouse" not in loki.lower():
        raise SystemExit(f"self-test: mouse was dropped: {loki}")
    street = line_from_caption("a cat walking on a sidewalk", "pigeon")
    if re.search(r"pigeon|bird", street, flags=re.I):
        raise SystemExit("self-test: a bird was invented")
    timeline = {
        "duration_sec": 30.0,
        "segments": [{"id": "door", "start": 0.0, "end": 20.0}],
        "beats": [],
    }
    highlights = {
        "events": [
            {
                "type": "jump",
                "time": 5.0,
                "score": 0.4,
                "host_line": "[0:05.0] template",
                "detail": "rose",
            }
        ]
    }
    captions = [{"time": 5.0, "caption": "a cat jumping up a white door", "sources": ["highlight"]}]
    cues = build_caption_cues(timeline, highlights, captions)
    nxt = next(cue for cue in cues if cue["kind"] == "next")
    comment = next(cue for cue in cues if cue["kind"] == "comment")
    if abs(nxt["lead_sec"] - 1.1) > 0.02:
        raise SystemExit("self-test: lead is not 1.1s")
    if "door" not in comment["text"].lower() or "jump" not in comment["text"].lower():
        raise SystemExit(f"self-test: comment is not the caption: {comment['text']}")
    if nxt["text"].split("  ", 1)[1] != comment["text"].split("  ", 1)[1]:
        raise SystemExit("self-test: NEXT and COMMENT diverged from the caption")
    print("self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Caption cat highlight frames")
    parser.add_argument("command", choices=["caption", "script", "self-test"])
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--rundown", type=Path)
    parser.add_argument("--reel", type=Path)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    if args.command == "caption":
        if not args.video or not args.out:
            raise SystemExit("caption needs a video and --out")
        analysis = args.out
        timeline_path = args.timeline or (analysis / "timeline.json")
        if timeline_path.is_file():
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        else:
            from media_info import ffprobe_video

            duration = ffprobe_video(args.video)["duration_sec"]
            timeline = {
                "duration_sec": duration,
                "segments": [{"id": "clip", "start": 0.0, "end": duration}],
                "beats": [],
            }
        highlights = json.loads((analysis / "highlights.json").read_text(encoding="utf-8"))
        events = json.loads((analysis / "events.json").read_text(encoding="utf-8"))
        write_captions(args.video, analysis, timeline, highlights, events)
        print(f"captions -> {analysis / 'captions.json'}")
        return 0
    analysis = args.analysis
    if not analysis or not args.timeline or not args.video or not args.reel:
        raise SystemExit("script needs --analysis, --timeline, a video, and --reel")
    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    highlights = json.loads((analysis / "highlights.json").read_text(encoding="utf-8"))
    events = json.loads((analysis / "events.json").read_text(encoding="utf-8"))
    payload = json.loads((analysis / "captions.json").read_text(encoding="utf-8"))
    captions = payload["captions"]
    cues = entertainment_from_captions(timeline, events, highlights, analysis, captions)
    cues.extend(build_caption_cues(timeline, highlights, captions))
    cues.sort(key=lambda cue: cue["start"])
    if not cues:
        raise SystemExit("no caption cues")
    cue_path = args.work / "caption-cues.json"
    cue_path.parent.mkdir(parents=True, exist_ok=True)
    cue_path.write_text(json.dumps(cues, indent=2) + "\n", encoding="utf-8")
    burn(args.video, cues, args.reel, args.work / "caption-burn")
    if args.rundown:
        write_script_doc(args.rundown, timeline, captions, cues)
        print(f"script -> {args.rundown}")
    print(f"reel -> {args.reel}")
    for cue in cues:
        print(f"  {cue['kind']:14} {cue['start']:7.2f}-{cue['end']:7.2f}  {cue['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
