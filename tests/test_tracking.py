from app.interaction_logic import InteractionScorer, score_interaction, update_interaction_confidence
from app.tracking import CentroidTracker, Track, TrackPoint, classify_motion, compress_states_to_ranges
from app.utils import Detection, bbox_iou, detect_installation_objects, expand_box


def test_basic_geometry_helpers():
    assert round(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 3) == 0.143
    assert expand_box((0, 0, 10, 10), 2.0, 100, 100) == (0, 0, 15, 15)


def test_installation_fallback_finds_cable_shape():
    import cv2
    import numpy as np

    frame = np.full((240, 320, 3), 245, dtype=np.uint8)
    cv2.line(frame, (35, 160), (285, 78), (20, 20, 20), 7)

    detections = detect_installation_objects(frame, frame_num=12)

    assert any(det.class_name == "cable" for det in detections)


def test_installation_fallback_skips_existing_yolo_box():
    import cv2
    import numpy as np

    frame = np.full((180, 260, 3), 245, dtype=np.uint8)
    cv2.rectangle(frame, (70, 60), (150, 120), (30, 30, 30), -1)
    existing = [Detection(0, "tv", 0.8, (60, 50, 160, 130), (110, 90))]

    detections = detect_installation_objects(frame, frame_num=0, existing=existing)

    assert detections == []


def test_tracker_keeps_same_id_for_nearby_box():
    tracker = CentroidTracker(640, 480)
    first = Detection(0, "drill", 0.9, (100, 100, 150, 150), (125, 125))
    second = Detection(5, "drill", 0.88, (105, 103, 155, 153), (130, 128))

    tracks = tracker.update([first])
    assert tracks[0].track_id == 0

    tracks = tracker.update([second])
    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    assert len(tracks[0].points) == 2


def test_motion_ranges_are_compact():
    assert compress_states_to_ranges(
        [(0, "stationary"), (5, "stationary"), (10, "moving"), (15, "moving")]
    ) == [
        {"frame_range": [0, 5], "state": "stationary"},
        {"frame_range": [10, 15], "state": "moving"},
    ]


def test_motion_uses_normalized_displacement():
    points = [
        TrackPoint(0, (95, 95, 105, 105), (100, 100), 0.9),
        TrackPoint(5, (97, 96, 107, 106), (102, 101), 0.9),
        TrackPoint(10, (145, 145, 155, 155), (150, 150), 0.9),
    ]

    motion = classify_motion(points, frame_width=200, frame_height=200, movement_threshold=0.02)
    assert motion[-1]["state"] == "moving"


def make_track(track_id, name, box):
    track = Track(track_id, name)
    track.points.append(
        TrackPoint(
            frame=0,
            bbox=box,
            center=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            confidence=0.9,
        )
    )
    return track


def test_interaction_confidence_and_persistence():
    person = make_track(0, "person", (100, 50, 220, 350))
    tool = make_track(1, "drill", (135, 95, 190, 160))
    far_tool = make_track(2, "drill", (500, 400, 560, 460))

    assert score_interaction(tool, person, 640, 480) > score_interaction(far_tool, person, 640, 480)

    high = update_interaction_confidence(0.2, 1.0)
    assert update_interaction_confidence(high, 0.0) < high

    scorer = InteractionScorer(640, 480)
    scorer.threshold = 0.4
    scorer.update(0, [tool], [person], set())
    scorer.update(5, [tool], [person], set())
    scorer.update(10, [tool], [person], set())
    assert scorer.pairs[(0, 1)].active_start == 10
