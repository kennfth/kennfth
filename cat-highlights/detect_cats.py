#!/usr/bin/env python3
"""Detect and track cats with Ultralytics YOLO nano and ByteTrack.

COCO class "cat" is the only class kept. Boxes and track ids are written to
events.json, along with a cheap frame-difference motion score so later rules
can see when the picture actually moves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from media_info import ffprobe_video

ROOT = Path(__file__).resolve().parent


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


class IdFallback:
    """Assign stable ids when ByteTrack has not emitted one yet."""

    def __init__(self) -> None:
        self.next_id = 1
        self.prev: list[tuple[int, list[float]]] = []

    def assign(self, boxes: list[list[float]]) -> list[int]:
        assigned: list[int] = []
        used: set[int] = set()
        for box in boxes:
            best_iou = 0.3
            best_id = None
            for prev_id, prev_box in self.prev:
                if prev_id in used:
                    continue
                overlap = iou(box, prev_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = prev_id
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            assigned.append(best_id)
        self.prev = list(zip(assigned, boxes))
        return assigned


def load_model(weights: Path) -> YOLO:
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        cwd = Path.cwd()
        os.chdir(weights.parent)
        try:
            YOLO(weights.name)
        finally:
            os.chdir(cwd)
        stray = cwd / weights.name
        if stray.exists() and not weights.exists():
            stray.replace(weights)
    if not weights.exists():
        raise RuntimeError(f"could not download {weights.name}")
    return YOLO(str(weights))


def cat_class_id(model: YOLO) -> int:
    names = model.names
    if isinstance(names, dict):
        pairs = names.items()
    else:
        pairs = enumerate(names)
    for idx, name in pairs:
        if str(name).lower() == "cat":
            return int(idx)
    raise RuntimeError("this YOLO model has no COCO 'cat' class")


def _boxes_from_result(result) -> list[tuple[list[float], float, int | None]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    ids = None
    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
    found = []
    for i, box in enumerate(xyxy):
        track_id = int(ids[i]) if ids is not None else None
        found.append(([round(float(v), 1) for v in box], round(float(confs[i]), 3), track_id))
    return found


def _summarize(detections: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in detections:
        grouped.setdefault(row["track_id"], []).append(row)
    tracks = []
    for track_id, rows in sorted(grouped.items()):
        tracks.append(
            {
                "track_id": track_id,
                "start": rows[0]["time"],
                "end": rows[-1]["time"],
                "frames": len(rows),
                "max_conf": max(row["conf"] for row in rows),
            }
        )
    return tracks


def _draw_sample(frame: np.ndarray, rows: list[dict], dest: Path) -> None:
    canvas = frame.copy()
    for row in rows:
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 170, 255), 2)
        label = f"cat {row['track_id']} {row['conf']:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 170, 255),
            2,
            cv2.LINE_AA,
        )
    h, w = canvas.shape[:2]
    long_side = max(h, w)
    if long_side > 480:
        scale = 480 / long_side
        canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])


def detect(
    video: Path,
    out_dir: Path,
    *,
    weights: Path,
    conf: float,
    imgsz: int,
    stride: int,
) -> dict:
    info = ffprobe_video(video)
    model = load_model(weights)
    class_id = cat_class_id(model)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")

    fps = info["fps"] or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or info["width"]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or info["height"]
    fallback = IdFallback()
    detections: list[dict] = []
    motion: list[dict] = []
    samples: list[tuple[float, np.ndarray, list[dict]]] = []
    prev_small = None
    frame_index = 0
    used_conf = conf

    def pass_video(threshold: float) -> None:
        nonlocal prev_small, frame_index, fallback
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_small = None
        frame_index = 0
        fallback = IdFallback()
        detections.clear()
        motion.clear()
        samples.clear()
        model.predictor = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_sec = frame_index / fps if fps else 0.0
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
            if prev_small is None:
                score = 0.0
            else:
                score = float(cv2.absdiff(small, prev_small).mean()) / 255.0
            prev_small = small
            motion.append({"time": round(time_sec, 3), "score": round(score, 4)})

            if frame_index % stride == 0:
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[class_id],
                    conf=threshold,
                    imgsz=imgsz,
                    verbose=False,
                )[0]
                found = _boxes_from_result(result)
                missing = [box for box, _conf, track_id in found if track_id is None]
                filled = fallback.assign(missing) if missing else []
                missing_i = 0
                frame_rows = []
                for box, box_conf, track_id in found:
                    if track_id is None:
                        track_id = filled[missing_i]
                        missing_i += 1
                    row = {
                        "frame": frame_index,
                        "time": round(time_sec, 3),
                        "track_id": int(track_id),
                        "conf": box_conf,
                        "bbox": box,
                    }
                    detections.append(row)
                    frame_rows.append(row)
                if frame_rows:
                    samples.append((time_sec, frame.copy(), frame_rows))
            frame_index += 1
            if frame_index % int(max(fps, 1) * 5) == 0:
                print(f"  {time_sec:5.1f}s  cats so far {len(detections)}", flush=True)

    pass_video(conf)
    if not detections and conf > 0.15:
        used_conf = 0.15
        print("no cats at the first threshold; retrying at 0.15", flush=True)
        pass_video(used_conf)

    cap.release()
    annotated = []
    if samples:
        def shown(item):
            strong = [row for row in item[2] if row["conf"] >= 0.65]
            return strong or item[2]

        picks = []
        richest = max(samples, key=lambda item: (len(shown(item)), max(row["conf"] for row in shown(item))))
        sharpest = max(samples, key=lambda item: max(row["conf"] for row in shown(item)))
        for kind, item in (("most_cats", richest), ("highest_conf", sharpest)):
            if any(abs(item[0] - kept[0]) < 0.4 and kind != kept_kind for kept_kind, kept in picks):
                continue
            picks.append((kind, item))
        ann_dir = out_dir / "annotated"
        for kind, (time_sec, frame, _rows) in picks[:2]:
            rows = shown((time_sec, frame, _rows))
            name = f"{kind}_{time_sec:.2f}s.jpg".replace(".", "p", 1)
            dest = ann_dir / name
            _draw_sample(frame, rows, dest)
            annotated.append({"file": f"annotated/{dest.name}", "time": round(time_sec, 3), "cats": len(rows)})

    payload = {
        "video": video.as_posix(),
        "width": width,
        "height": height,
        "fps": round(float(fps), 3),
        "duration_sec": round(float(info["duration_sec"]), 3),
        "frame_count": frame_index,
        "model": weights.name,
        "tracker": "bytetrack",
        "class_name": "cat",
        "class_id": class_id,
        "conf_threshold": used_conf,
        "imgsz": imgsz,
        "vid_stride": stride,
        "detections": detections,
        "tracks": _summarize(detections),
        "motion": motion,
        "annotated": annotated,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack cat events")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if not args.video.is_file():
        print(f"missing video: {args.video}", file=sys.stderr)
        return 1
    print(f"tracking cats in {args.video.name}", flush=True)
    payload = detect(
        args.video,
        args.out,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        stride=args.stride,
    )
    print(
        f"{args.video.name}: {len(payload['detections'])} boxes, "
        f"{len(payload['tracks'])} tracks -> {args.out / 'events.json'}"
    )
    for track in payload["tracks"]:
        print(
            f"  track {track['track_id']}: {track['start']:.2f}s-{track['end']:.2f}s "
            f"frames {track['frames']} max_conf {track['max_conf']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
