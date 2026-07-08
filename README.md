# Soccer Ball Detection & Tracking

A YOLOv8-based pipeline that detects a soccer ball in match video, tracks it frame-to-frame with ByteTrack, and renders an annotated clip with a bounding box, confidence score, and a fading motion trail.

## How it works

The pipeline runs in three stages, split across two environments (local for data/inference, Colab for GPU training):

```
Stage 1 (Local): src/prepare.py   →  data/processed/   →  upload to Google Drive
Stage 2 (Colab): notebooks/train.ipynb  →  models/best.pt  →  download locally
Stage 3 (Local): src/detect.py   →  src/utils/pipeline.py  →  annotated video
```

- **`src/prepare.py`** — filters a raw Roboflow-style dataset down to the ball class, subsamples it, and produces an 80/20 train/val split.
- **`notebooks/train.ipynb`** — fine-tunes YOLOv8s on the prepared dataset (run on Colab, GPU-accelerated).
- **`src/detect.py`** — CLI entry point that loads the trained weights and runs detection + tracking on an input video.
- **`src/utils/pipeline.py`** — `process_video()`, the shared core that opens the video, runs YOLO + ByteTrack per frame, and writes the annotated output.
- **`src/utils/annotate.py`** — `draw_annotations()`, draws the bounding box, confidence label, and fading trail onto a frame.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Prepare a dataset

Place a Roboflow-exported dataset in `data/raw/`, check its `data.yaml` for the ball class ID, then:

```bash
python src/prepare.py --input data/raw --output data/processed --ball-class-id <ID> --n 5000
```

### 2. Train

Upload `data/processed/` to Google Drive and run `notebooks/train.ipynb` in Colab. Download the resulting `best.pt` into `models/`.

### 3. Run inference on a video

```bash
python src/detect.py --input path/to/match.mp4 --output path/to/annotated.mp4
```

Optional flags: `--model` (default `models/best.pt`), `--conf` (default `0.4`), `--trail-len` (default `15` frames).

Input can be any container OpenCV/FFmpeg can decode (`.mp4`, `.mov`, etc.). The output is written as `.mp4` (H.264-style `mp4v` codec), so keep the `--output` extension as `.mp4`.

### 4. Evaluate on an independent test set

```bash
python -c "from ultralytics import YOLO; m = YOLO('models/best.pt'); m.val(data='data/test/data.yaml')"
```

## Project structure

```
src/
  prepare.py           # dataset filtering, subsampling, splitting
  detect.py            # CLI entry point for inference
  utils/
    annotate.py         # draw_annotations()
    pipeline.py         # process_video()
notebooks/
  train.ipynb          # Colab training notebook (YOLOv8s fine-tuning)
data/                  # raw/, processed/, test/ (gitignored — populate locally)
models/                # best.pt (gitignored — download after training)
runs/                  # training/validation artifacts (gitignored)
```

`data/`, `models/`, and `runs/` are gitignored due to size — they need to be populated locally (see Usage above) rather than pulled from this repo.
