import cv2
import numpy as np

_GREEN = (0, 255, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_annotations(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    trail_points: list[tuple[int, int]],
) -> np.ndarray:
    out = frame.copy()
    _draw_trail(out, trail_points)
    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), _GREEN, 2)
        label = f"ball {conf:.2f}"
        cv2.putText(out, label, (x1, y1 - 6), _FONT, 0.5, _GREEN, 1, cv2.LINE_AA)
    return out


def _draw_trail(frame: np.ndarray, trail_points: list[tuple[int, int]]) -> None:
    n = len(trail_points)
    for i, pt in enumerate(trail_points):
        alpha = int(255 * ((i + 1) / n) ** 3)
        color = (0, alpha, 0)
        cv2.circle(frame, pt, 3, color, -1)
    if n >= 2:
        for i in range(1, n):
            alpha = int(255 * ((i + 1) / n) ** 3)
            color = (0, alpha, 0)
            cv2.line(frame, trail_points[i - 1], trail_points[i], color, 2)
