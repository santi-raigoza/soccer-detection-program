import collections
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils.annotate import draw_annotations


def process_video(
    input_path: str,
    output_path: str,
    model_path: str,
    conf: float = 0.4,
    trail_len: int = 15,
) -> None:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    trail: collections.deque[tuple[int, tuple[int, int]]] = collections.deque()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=conf,
            verbose=False,
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            best = int(r.boxes.conf.argmax().item())
            x1, y1, x2, y2 = r.boxes.xyxy[best].cpu().numpy()
            trail.append((frame_idx, (int((x1 + x2) / 2), int((y1 + y2) / 2))))
            boxes_xyxy = r.boxes.xyxy[best : best + 1].cpu().numpy()
            confs = r.boxes.conf[best : best + 1].cpu().numpy()
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)

        # Age out points by elapsed frames, not detection count, so a run of
        # missed detections doesn't leave a stale point anchoring the trail.
        while trail and frame_idx - trail[0][0] >= trail_len:
            trail.popleft()

        annotated = draw_annotations(frame, boxes_xyxy, confs, [pt for _, pt in trail])
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
