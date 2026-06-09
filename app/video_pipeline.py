import json
from functools import cache
from pathlib import Path

from app import database
from app.interaction_logic import InteractionScorer
from app.models import TaskStatus
from app.tracking import CentroidTracker, Track, classify_motion
from app.utils import (
    Detection,
    bbox_center,
    canonical_class_name,
    detect_installation_objects,
    draw_tracks,
    merge_overlapping_detections,
    probe_video,
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
YOLO_MODEL = "yolov8s-world.pt"  # open-vocab variant; plain yolov8n only knew the 80 COCO labels, which miss cables/lab gear
SAMPLE_EVERY = 5
MIN_CONFIDENCE = 0.05
MIN_BOX_AREA_RATIO = 0.0007
MAX_KEYFRAMES = 8
KEYFRAME_GAP = 24
MIN_TRACK_POINTS = 3
MAX_TRACKS_PER_CLASS = 3
SINGLETON_CLASSES = {
    "spectrophotometer",
    "microscope",
    "lab_bench",
    "test_tube_rack",
    "power_supply",
}
CONTEXT_ONLY_CLASSES = {"microscope", "lab_bench", "test_tube_rack"}

# COCO never contained install/lab items (cables, spectrophotometers, etc.), so we hand
# yolo-world an explicit vocabulary instead of trusting its built-in class list. An upload
# can override this per task (see process_task); this is just the sensible fallback.
DEFAULT_CLASSES = [
    "person",
    "spectrophotometer",
    "laboratory instrument",
    "data cable",
    "fiber optic cable",
    "power cable",
    "power supply",
    "power adapter",
    "front interface port",
    "connector",
    "laboratory bench",
    "microscope",
    "test tube rack",
]


@cache
def yolo_model():
    from ultralytics import YOLOWorld

    return YOLOWorld(YOLO_MODEL)


def process_task(task_id: str) -> None:
    task = database.get_task(task_id)
    if not task:
        return

    print(f"[task {task_id}] processing {task.filename}")
    database.update_task(task_id, TaskStatus.processing)
    try:
        # classes is optional and may be absent on older task records, so default it safely
        classes = getattr(task, "classes", None)
        result_path = process_video(task_id, task.filename, task.video_path, classes)
        database.update_task(task_id, TaskStatus.completed, result_path=str(result_path))
        print(f"[task {task_id}] done")
    except Exception as exc:
        database.update_task(task_id, TaskStatus.failed, error=str(exc))
        print(f"[task {task_id}] failed: {exc}")


def detect_frame(frame, frame_num: int) -> tuple[list[Detection], list[Detection]]:
    h, w = frame.shape[:2]
    result = yolo_model().predict(frame, conf=MIN_CONFIDENCE, verbose=False)[0]
    people, objects = [], []
    if result.boxes is None:
        return people, objects

    for box in result.boxes:
        conf = float(box.conf.item())
        class_id = int(box.cls.item())
        class_name = canonical_class_name(yolo_model().names.get(class_id, str(class_id)))
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        # skip weak detections and tiny background specks
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, w * h)
        if conf < MIN_CONFIDENCE or area_ratio < MIN_BOX_AREA_RATIO:
            continue

        det = Detection(frame_num, class_name, conf, (x1, y1, x2, y2), bbox_center((x1, y1, x2, y2)))
        if class_name == "person":
            people.append(det)
        else:
            objects.append(det)

    objects = merge_overlapping_detections(objects)
    objects.extend(detect_installation_objects(frame, frame_num, existing=people + objects))
    return merge_overlapping_detections(people), merge_overlapping_detections(objects)


def process_video(task_id: str, filename: str, video_path: str, classes: list[str] | None = None) -> Path:
    import cv2

    info = probe_video(video_path)
    out_dir = OUTPUT_DIR / task_id
    keyframe_dir = out_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    # Set the detection vocabulary for this video before reading any frames. yolo_model() is a
    # single cached instance, so we set classes per task here rather than per frame. Fine for the
    # local/sequential processing here; would need one model per worker if we ran tasks in parallel.
    detection_classes = list(classes or DEFAULT_CLASSES)
    if "person" not in detection_classes:
        detection_classes.append("person")  # keep it in or the people/objects split below goes empty
    yolo_model().set_classes(detection_classes)

    object_tracker = CentroidTracker(info.width, info.height)
    person_tracker = CentroidTracker(info.width, info.height)
    interactions = InteractionScorer(info.width, info.height)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    last_processed = 0
    all_objects: dict[int, Track] = {}
    motion_state: dict[int, str] = {}
    saved_keyframes: list[dict] = []
    previous_scene_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_num % SAMPLE_EVERY != 0:
            frame_num += 1
            continue

        people_dets, object_dets = detect_frame(frame, frame_num)
        people = person_tracker.update(people_dets)
        objects = object_tracker.update(object_dets)
        all_objects.update({track.track_id: track for track in objects})

        moving_now = _moving_transitions(objects, motion_state, info.width, info.height)
        actionable_objects = [
            track for track in objects if track.class_name not in CONTEXT_ONLY_CLASSES
        ]
        events = interactions.update(frame_num, actionable_objects, people, moving_now)

        if events and max(e["confidence"] for e in events) > 0.52:
            _save_keyframe(
                frame,
                frame_num,
                "interaction_peak",
                objects,
                people,
                keyframe_dir,
                saved_keyframes,
            )
        if _nearby_moving_objects(actionable_objects, people, moving_now, info.width, info.height):
            _save_keyframe(
                frame,
                frame_num,
                "stationary_to_moving_near_person",
                objects,
                people,
                keyframe_dir,
                saved_keyframes,
            )
        if previous_scene_frame is not None and _scene_change_score(previous_scene_frame, frame) > 0.32:
            _save_keyframe(
                frame,
                frame_num,
                "scene_change",
                objects,
                people,
                keyframe_dir,
                saved_keyframes,
            )
        previous_scene_frame = frame.copy()

        last_processed = frame_num
        frame_num += 1

    cap.release()
    pair_states = interactions.finish(last_processed)
    result = _format_result(filename, info, all_objects, pair_states, saved_keyframes)
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result_path


def _moving_transitions(
    objects: list[Track],
    old_states: dict[int, str],
    frame_width: int,
    frame_height: int,
) -> set[int]:
    moved = set()
    for track in objects:
        recent_motion = classify_motion(track.points[-4:], frame_width, frame_height)
        state = recent_motion[-1]["state"] if recent_motion else "stationary"
        if old_states.get(track.track_id, "stationary") == "stationary" and state == "moving":
            moved.add(track.track_id)
        old_states[track.track_id] = state
    return moved


def _save_keyframe(frame, frame_num, reason, objects, people, keyframe_dir, saved) -> None:
    import cv2
    if len(saved) >= MAX_KEYFRAMES:
        return
    if any(k["reason"] == reason and abs(k["frame"] - frame_num) < KEYFRAME_GAP for k in saved):
        return

    copy = frame.copy()
    draw_tracks(copy, objects, people)
    path = keyframe_dir / f"frame_{frame_num}_{reason}.jpg"
    cv2.imwrite(str(path), copy)
    saved.append({"frame": frame_num, "path": str(path), "reason": reason})


def _format_result(filename, info, objects, pair_states, keyframes) -> dict:
    candidates = []
    for object_id, track in sorted(objects.items()):
        object_interactions = []
        for (person_id, pair_object_id), state in pair_states.items():
            if pair_object_id != object_id:
                continue
            for interval in state.intervals:
                object_interactions.append(
                    {
                        "interacted_by_person": person_id,
                        "frame_start": interval["frame_start"],
                        "frame_end": interval["frame_end"],
                        "confidence_peak": interval["confidence_peak"],
                    }
                )

        if len(track.points) < MIN_TRACK_POINTS and not object_interactions:
            continue

        candidates.append(
            (
                bool(object_interactions),
                len(track.points),
                sum(point.confidence for point in track.points) / len(track.points),
                {
                    "object_id": object_id,
                    "class": track.class_name,
                    "motion_history": classify_motion(track.points, info.width, info.height),
                    "interactions": object_interactions,
                },
            )
        )

    final_objects = []
    class_counts = {}
    for _, _, _, item in sorted(candidates, key=lambda candidate: candidate[:3], reverse=True):
        class_name = item["class"]
        class_limit = 1 if class_name in SINGLETON_CLASSES else MAX_TRACKS_PER_CLASS
        if class_counts.get(class_name, 0) >= class_limit:
            continue
        final_objects.append(item)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        "videoMetadata": {
            "filename": filename,
            "duration_seconds": info.duration_seconds,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "resolution": {"width": info.width, "height": info.height},
        },
        "objectsDetected": final_objects,
        "keyFrames": keyframes,
    }


def _nearby_moving_objects(objects, people, moving_now, frame_width, frame_height) -> bool:
    from app.interaction_logic import score_interaction

    for obj in objects:
        if obj.track_id not in moving_now:
            continue
        for person in people:
            if score_interaction(obj, person, frame_width, frame_height, just_started_moving=True) >= 0.45:
                return True
    return False


def _scene_change_score(previous, current) -> float:
    import cv2

    previous_small = cv2.resize(previous, (160, 90))
    current_small = cv2.resize(current, (160, 90))
    return float(cv2.absdiff(previous_small, current_small).mean() / 255.0)
