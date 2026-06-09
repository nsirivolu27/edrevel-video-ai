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


def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = int(frames) if frames and math.isfinite(frames) and frames > 0 else 0
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


def detect_installation_objects(
    frame,
    frame_num: int,
    existing: list[Detection] | None = None,
) -> list[Detection]:
    import cv2

    existing = existing or []
    height, width = frame.shape[:2]
    frame_area = max(1, width * height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # YOLO does not know cables, so this catches thin installation shapes.
    edges = cv2.Canny(gray, 55, 150)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals: list[Detection] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 6 or h < 6:
            continue

        bbox = _pad_box((x, y, x + w, y + h), 4, width, height)
        area_ratio = box_area(bbox) / frame_area
        if area_ratio < 0.00012:
            continue
        if _overlaps_existing(bbox, existing + proposals):
            continue

        label, confidence = _classify_contour(contour, bbox, width, height)
        if not label:
            continue
        if label == "installation_object" and area_ratio > 0.09:
            continue

        proposals.append(
            Detection(
                frame=frame_num,
                class_name=label,
                confidence=confidence,
                bbox=bbox,
                center=bbox_center(bbox),
            )
        )

    # Keep this useful instead of flooding the tracker with background texture.
    return sorted(proposals, key=lambda det: det.confidence, reverse=True)[:12]


def _classify_contour(contour, bbox: BBox, frame_width: int, frame_height: int):
    import cv2

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None, 0.0

    long_side = max(box_w, box_h)
    short_side = max(1.0, min(box_w, box_h))
    aspect = long_side / short_side
    contour_length = cv2.arcLength(contour, closed=False)
    diag = max(1.0, (frame_width**2 + frame_height**2) ** 0.5)

    rotated = cv2.minAreaRect(contour)
    rw, rh = rotated[1]
    rotated_aspect = max(rw, rh) / max(1.0, min(rw, rh))

    if rotated_aspect >= 4.0 and contour_length / diag > 0.035:
        confidence = min(0.78, 0.42 + min(rotated_aspect, 18) / 26 + contour_length / diag)
        return "cable", round(confidence, 3)

    # A softer bucket for brackets, connectors, plates, and small parts.
    extent = cv2.contourArea(contour) / max(1.0, box_w * box_h)
    if 0.22 <= extent <= 0.92 and 0.0006 <= (box_w * box_h) / max(1, frame_width * frame_height) <= 0.035:
        confidence = min(0.62, 0.31 + extent * 0.25 + min(aspect, 3.0) * 0.03)
        return "installation_object", round(confidence, 3)

    return None, 0.0


def _overlaps_existing(box: BBox, detections: list[Detection]) -> bool:
    for det in detections:
        if bbox_iou(box, det.bbox) > 0.28:
            return True
        cx, cy = bbox_center(box)
        x1, y1, x2, y2 = det.bbox
        if x1 <= cx <= x2 and y1 <= cy <= y2 and box_area(box) < box_area(det.bbox) * 0.65:
            return True
    return False


def _pad_box(box: BBox, pad: int, frame_width: int, frame_height: int) -> BBox:
    x1, y1, x2, y2 = box
    return (
        max(0.0, x1 - pad),
        max(0.0, y1 - pad),
        min(float(frame_width), x2 + pad),
        min(float(frame_height), y2 + pad),
    )
