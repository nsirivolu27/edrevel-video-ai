import math
from dataclasses import dataclass

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class Detection:
    frame: int
    class_name: str
    confidence: float
    bbox: BBox
    center: Point


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


def bbox_center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - overlap
    return 0.0 if union <= 0 else overlap / union


def normalized_distance(a: Point, b: Point, frame_width: int, frame_height: int) -> float:
    diag = math.hypot(frame_width, frame_height)
    return 0.0 if diag == 0 else math.dist(a, b) / diag


def expand_box(box: BBox, scale: float, frame_width: int, frame_height: int) -> BBox:
    if scale < 1:
        raise ValueError("scale should be >= 1")
    x1, y1, x2, y2 = box
    cx, cy = bbox_center(box)
    new_w = (x2 - x1) * scale
    new_h = (y2 - y1) * scale
    return (
        max(0.0, cx - new_w / 2),
        max(0.0, cy - new_h / 2),
        min(float(frame_width), cx + new_w / 2),
        min(float(frame_height), cy + new_h / 2),
    )


def distance_to_box(point: Point, box: BBox) -> float:
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def probe_video(path: str) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return VideoInfo(
        fps=round(fps, 3),
        frame_count=frames,
        duration_seconds=round(frames / fps, 3) if fps else 0,
        width=width,
        height=height,
    )


def draw_tracks(frame, objects, people) -> None:
    # simple visualization for debugging saved keyframes
    for track in people:
        _draw_one(frame, track, (80, 180, 255), f"person:{track.track_id}")
    for track in objects:
        _draw_one(frame, track, (80, 255, 120), f"{track.class_name}:{track.track_id}")


def _draw_one(frame, track, color, label: str) -> None:
    import cv2

    x1, y1, x2, y2 = [int(v) for v in track.last.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
