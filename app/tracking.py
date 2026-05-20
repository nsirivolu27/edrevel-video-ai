from collections import deque
from dataclasses import dataclass, field

from app.utils import BBox, Detection, bbox_iou, normalized_distance


@dataclass
class TrackPoint:
    frame: int
    bbox: BBox
    center: tuple[float, float]
    confidence: float


@dataclass
class Track:
    track_id: int
    class_name: str
    points: list[TrackPoint] = field(default_factory=list)
    missed: int = 0

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    def add(self, detection: Detection) -> None:
        self.points.append(
            TrackPoint(detection.frame, detection.bbox, detection.center, detection.confidence)
        )
        self.missed = 0


class CentroidTracker:
    def __init__(self, frame_width: int, frame_height: int, max_missed: int = 6) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        scored_matches = []

        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                if det.class_name != track.class_name:
                    continue
                iou = bbox_iou(track.last.bbox, det.bbox)
                dist = normalized_distance(
                    track.last.center, det.center, self.frame_width, self.frame_height
                )
                # A mixed score is easier to explain than a black-box tracker.
                score = 0.65 * iou + 0.35 * max(0.0, 1.0 - dist * 6.0)
                if score >= 0.22:
                    scored_matches.append((score, track_id, det_idx))

        for _, track_id, det_idx in sorted(scored_matches, reverse=True):
            if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                continue
            self.tracks[track_id].add(detections[det_idx])
            unmatched_tracks.remove(track_id)
            unmatched_dets.remove(det_idx)

        for track_id in unmatched_tracks:
            self.tracks[track_id].missed += 1

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track = Track(self.next_id, det.class_name)
            track.add(det)
            self.tracks[self.next_id] = track
            self.next_id += 1

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.missed <= self.max_missed
        }
        return list(self.tracks.values())


def compress_states_to_ranges(frame_states: list[tuple[int, str]]) -> list[dict]:
    if not frame_states:
        return []

    ranges = []
    start_frame, state = frame_states[0]
    prev_frame = start_frame
    for frame, new_state in frame_states[1:]:
        if new_state != state:
            ranges.append({"frame_range": [start_frame, prev_frame], "state": state})
            start_frame, state = frame, new_state
        prev_frame = frame

    ranges.append({"frame_range": [start_frame, prev_frame], "state": state})
    return ranges


def classify_motion(
    points: list[TrackPoint],
    frame_width: int = 1920,
    frame_height: int = 1080,
    movement_threshold: float = 0.008,
) -> list[dict]:
    if not points:
        return []
    if len(points) == 1:
        return [{"frame_range": [points[0].frame, points[0].frame], "state": "stationary"}]

    # smooth noisy movement a bit; YOLO boxes wiggle even when objects don't
    recent = deque(maxlen=3)
    states = [(points[0].frame, "stationary")]
    for prev, cur in zip(points, points[1:]):
        recent.append(normalized_distance(prev.center, cur.center, frame_width, frame_height))
        avg_move = sum(recent) / len(recent)
        states.append((cur.frame, "moving" if avg_move >= movement_threshold else "stationary"))

    return compress_states_to_ranges(states)
