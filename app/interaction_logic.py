from dataclasses import dataclass, field

from app.tracking import Track
from app.utils import bbox_iou, distance_to_box, expand_box, normalized_distance


@dataclass
class PairState:
    confidence: float = 0.0
    hot_count: int = 0
    active_start: int | None = None
    peak: float = 0.0
    intervals: list[dict] = field(default_factory=list)
    timeline: list[tuple[int, float]] = field(default_factory=list)


def update_interaction_confidence(previous_score: float, current_evidence: float) -> float:
    current_evidence = max(0.0, min(1.0, current_evidence))
    if current_evidence > previous_score:
        score = previous_score * 0.55 + current_evidence * 0.45
    else:
        score = previous_score * 0.78 + current_evidence * 0.22
    return round(max(0.0, min(1.0, score)), 4)


def interaction_zone(person_box, frame_width: int, frame_height: int):
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    upper_body = (x1, y1, x2, y1 + h * 0.68)
    return expand_box(upper_body, 1.35, frame_width, frame_height)


def score_interaction(
    obj: Track,
    person: Track,
    frame_width: int,
    frame_height: int,
    just_started_moving: bool = False,
) -> float:
    zone = interaction_zone(person.last.bbox, frame_width, frame_height)
    dist_norm = distance_to_box(obj.last.center, zone) / max(
        1.0, (frame_width**2 + frame_height**2) ** 0.5
    )
    proximity = max(0.0, 1.0 - dist_norm * 9.0)
    overlap = min(1.0, bbox_iou(obj.last.bbox, zone) * 3.5)
    broad_nearness = max(
        0.0,
        1.0 - normalized_distance(obj.last.center, person.last.center, frame_width, frame_height) * 4,
    )

    # Hardcoded on purpose for this prototype; these are the knobs I'd tune with videos.
    bonus = 0.16 if just_started_moving else 0.0
    return round(min(1.0, 0.42 * proximity + 0.33 * overlap + 0.15 * broad_nearness + bonus), 4)


class InteractionScorer:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.threshold = 0.58
        self.min_frames = 2
        self.pairs: dict[tuple[int, int], PairState] = {}

    def update(
        self,
        frame: int,
        objects: list[Track],
        people: list[Track],
        moving_now: set[int],
    ) -> list[dict]:
        events = []
        seen = set()

        for obj in objects:
            for person in people:
                key = (person.track_id, obj.track_id)
                seen.add(key)
                pair = self.pairs.setdefault(key, PairState())
                evidence = score_interaction(
                    obj, person, self.frame_width, self.frame_height, obj.track_id in moving_now
                )
                pair.confidence = update_interaction_confidence(pair.confidence, evidence)
                pair.timeline.append((frame, pair.confidence))

                if pair.confidence >= self.threshold:
                    pair.hot_count += 1
                    pair.peak = max(pair.peak, pair.confidence)
                    if pair.hot_count >= self.min_frames and pair.active_start is None:
                        pair.active_start = frame
                    events.append(
                        {
                            "object_id": obj.track_id,
                            "person_id": person.track_id,
                            "frame": frame,
                            "confidence": pair.confidence,
                        }
                    )
                else:
                    self._maybe_close(pair, frame)

        for key, pair in self.pairs.items():
            if key in seen:
                continue
            pair.confidence = update_interaction_confidence(pair.confidence, 0.0)
            pair.timeline.append((frame, pair.confidence))
            if pair.confidence < self.threshold:
                self._maybe_close(pair, frame)

        return events

    def finish(self, final_frame: int) -> dict[tuple[int, int], PairState]:
        for pair in self.pairs.values():
            self._maybe_close(pair, final_frame, force=True)
        return self.pairs

    def _maybe_close(self, pair: PairState, frame: int, force: bool = False) -> None:
        if pair.active_start is None:
            pair.hot_count = 0
            pair.peak = 0.0
            return
        if not force and pair.confidence >= self.threshold:
            return
        # interaction probably ended here
        pair.intervals.append(
            {
                "frame_start": pair.active_start,
                "frame_end": frame,
                "confidence_peak": round(pair.peak, 3),
            }
        )
        pair.active_start = None
        pair.hot_count = 0
        pair.peak = 0.0
