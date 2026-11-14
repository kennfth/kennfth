# Cat highlights

Turn a cat video into scene keyframes, tracked cat boxes, and short highlight clips. The rules look for jumps, failed landings, exits and re-entries, motion spikes, and multi-cat chaos. Host lines are filled from those event types. If Qwen-VL is not installed locally, the scripts still write the lines from templates.

## Install

FFmpeg needs to be on `PATH` (`ffmpeg` and `ffprobe`).

```bash
cd cat-highlights
python3 -m venv .venv
source .venv/bin/activate
# CPU machines: install torch before ultralytics so pip does not fetch the CUDA wheel.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The first detection run downloads `yolo11n.pt` into `weights/`. Weights and the virtualenv are gitignored.

## Sample clips

Five Pexels clips, each 15–30 seconds, of an orange or ginger cat:

```bash
python download_samples.py
```

That writes `test_videos/orange_cat_01.mp4` through `orange_cat_05.mp4`. The mp4 files are gitignored. `test_videos/sources.json` records the author, page, and file URL.

| File | Clip | Author |
| --- | --- | --- |
| `orange_cat_01.mp4` | [Orange stray cat in the public park](https://www.pexels.com/video/orange-stray-cat-in-the-public-park-19975462/) (15s) | Zeki Binici |
| `orange_cat_02.mp4` | [An orange tabby cat playing with a stick](https://www.pexels.com/video/an-orange-tabby-cat-playing-with-a-stick-15024233/) (22s) | Nothing Ahead |
| `orange_cat_03.mp4` | [An orange tabby kitten playing among garden plants](https://www.pexels.com/video/an-orange-tabby-kitten-playing-among-garden-plants-13825474/) (30s) | Marsel Sharipov |
| `orange_cat_04.mp4` | [Angery orange cat with duck](https://www.pexels.com/video/angery-orange-cat-with-duck-20503094/) (15s) | Hehe Panda |
| `orange_cat_05.mp4` | [A ginger cat looking around](https://www.pexels.com/video/a-ginger-cat-looking-around-6853904/) (15s) | cottonbro studio |

License: [Pexels License](https://www.pexels.com/license/). Attribution is not required; it is listed here anyway.

## Run

```bash
python run_pipeline.py
```

One video:

```bash
python extract_keyframes.py test_videos/orange_cat_01.mp4 --out output/orange_cat_01
python detect_cats.py test_videos/orange_cat_01.mp4 --out output/orange_cat_01
python make_highlights.py test_videos/orange_cat_01.mp4 --out output/orange_cat_01
```

Check the highlight rules without a video:

```bash
python make_highlights.py --self-test
```

## Drop in a video

Put any cat clip where you like and point the scripts at it. A 15–60 second file is the intended size. Output goes to the directory you pass as `--out`.

```bash
python run_pipeline.py /path/to/my_cat.mp4 --out output
```

`detect_cats.py` uses the COCO cat class only (YOLO11 nano + ByteTrack). The first pass keeps boxes at confidence 0.25 and retries once at 0.15 if that finds nothing.

## What you get

For each video, under `output/<name>/`:

- `keyframes.json` and `keyframes/kf_*.jpg` — FFmpeg scene-score peaks and their timestamps
- `events.json` — cat boxes, track ids, and a per-frame motion score
- `annotated/*.jpg` — a couple of frames with the boxes drawn
- `highlights.json` — the rule hits and the clip list
- `highlights/hl_*.mp4` — re-encoded cuts, kept small
- `commentary.txt` — one host line per clip

Jumps are a fast rise of the whole box. A fail is that rise coming back down. An exit is a track that leaves through the edge of the frame; a re-entry is a new track that comes back in from an edge just after one. Motion spikes are peaks in track speed measured over about a fifth of a second, so a one-frame box twitch does not count. Multi-cat chaos needs two separated cats, each at confidence 0.6 or better. Touching halves of one cat (a branch can split a single animal) stay one cat, and a low-confidence box on something that is not a cat does not count.

Commentary uses a local Qwen-VL model only when `ollama list` already shows one. Otherwise `commentary_source` is `template`.

## Stitched host reel

`stitch_host_reel.py` joins 3–5 short clips with a 0.4s black beat, then burns three text lines onto the file the pipeline already analyzed. It does not add a voice track.

```bash
python stitch_host_reel.py stitch --manifest work/manifest.json --work work --out work/stitched-plain.mp4 --timeline work/timeline.json
python run_pipeline.py work/stitched-plain.mp4 --out work/output
python stitch_host_reel.py overlay --video work/stitched-plain.mp4 --analysis work/output/stitched-plain --timeline work/timeline.json --work work --out /path/to/reel.mp4
python stitch_host_reel.py self-test
```

ENTERTAINMENT marks a rise in the motion score or the scene score. COMMENT is the pipeline host line at the event. NEXT is a short anticipation line from that same event, placed 1.1s earlier. Source mp4s, the stitch, and the burned reel belong under `work/` or outside the repo. `work/` is gitignored.
