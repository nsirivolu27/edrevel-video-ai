# Edrevel Video Object Interaction

Video object-interaction analysis project for Edrevel AI.

This repo started as my SDE Intern technical assessment for Edrevel AI and is now the working project my team and I are continuing during the internship. The core problem is still the same: given an installation-style video, identify the objects a person interacts with and explain when those interactions happen.

The current app is a FastAPI service that samples video frames, detects people and objects, tracks them over time, classifies object motion, estimates person-object interaction windows, and exports annotated keyframes.

## Current Focus

The first version used a base YOLO model, which worked for common COCO objects but missed domain-specific installation items like cables and small parts. The project is now moving from a general assessment prototype toward a more useful installation-video analysis tool.

Recent work adds a lightweight OpenCV fallback for objects YOLO does not reliably label:

- thin elongated shapes are proposed as `cable`
- smaller non-COCO parts are proposed as `installation_object`
- fallback detections are merged into the same tracking and interaction pipeline
- duplicate fallback boxes are skipped when YOLO already found the object

This is not meant to replace a fine-tuned detector. It is a practical bridge while we collect better data and decide which installation-specific labels matter most.

## What It Does

- Upload a video through `POST /tasks/upload`
- Extract video metadata: FPS, frame count, duration, and resolution
- Process every Nth frame while preserving original frame numbers
- Detect people separately from candidate objects
- Add installation-specific fallback detections for cables and parts
- Track people and objects with a custom centroid + IoU tracker
- Classify object motion as moving or stationary over time
- Estimate interaction windows using person proximity, object overlap, motion changes, and temporal confidence
- Save annotated keyframes under `outputs/{task_id}/keyframes/`
- Store task state and result paths in a local JSON file

## Project Layout

```text
app/
  main.py
  routes.py
  video_pipeline.py
  tracking.py
  interaction_logic.py
  utils.py
  models.py
  database.py

tests/
  test_api.py
  test_tracking.py

sample_result.json
requirements.txt
```

The structure is intentionally small. Most of the important logic lives in `video_pipeline.py`, `tracking.py`, `interaction_logic.py`, and `utils.py`.

## Pipeline

1. Probe the video for metadata.
2. Sample frames for speed.
3. Run YOLO and split detections into people and candidate objects.
4. Add OpenCV fallback detections for cable-like shapes and small installation parts.
5. Track people and objects separately with a centroid + IoU matcher.
6. Smooth object center movement and compress repeated states into frame ranges.
7. Build an upper-body interaction zone for each person.
8. Score nearby objects using distance, zone overlap, broad proximity, and recent motion changes.
9. Smooth interaction confidence over time so one noisy frame does not create a false interaction.
10. Export result JSON and annotated keyframes.

## Why the Custom Tracker?

ByteTrack or DeepSORT would likely track better in production, but this project keeps the first tracker simple and readable. The tracker matches detections using:

- class label
- bounding-box IoU
- normalized center distance
- missed-frame tolerance

That makes the behavior easier to debug and explain while the team is still validating the detection and interaction logic.

## Interaction Logic

The app does not use hand-pose estimation yet. Instead, it uses a practical heuristic:

- take the upper region of each person bounding box
- expand it slightly to represent a likely hand/upper-body interaction zone
- score objects based on distance and overlap with that zone
- add a small bonus when an object recently changes from stationary to moving
- require confidence to stay high across multiple sampled frames

This is a reasonable baseline for installation videos, but it is still a heuristic. Hand pose, better object labels, and more video examples are natural next steps.

## Setup

```bash
cd edrevel-video-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Useful URLs:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/docs`
- `GET http://127.0.0.1:8000/debug/paths`

The first real video run may download `yolov8n.pt`.

## API Examples

Upload:

```bash
curl -X POST "http://127.0.0.1:8000/tasks/upload" -F "file=@install_video.mp4"
```

Check status:

```bash
curl "http://127.0.0.1:8000/tasks/{task_id}/status"
```

Get result:

```bash
curl "http://127.0.0.1:8000/tasks/{task_id}/result"
```

List tasks:

```bash
curl "http://127.0.0.1:8000/tasks"
```

## Tests

```bash
pytest
```

The tests cover deterministic parts of the project: IoU, box expansion, ID stability, motion compression, confidence smoothing, API basics, and the installation-object fallback.

## Example Output

See `sample_result.json` for a fuller example.

```json
{
  "objectsDetected": [
    {
      "object_id": 1,
      "class": "drill",
      "motion_history": [
        { "frame_range": [0, 70], "state": "stationary" },
        { "frame_range": [75, 155], "state": "moving" }
      ],
      "interactions": [
        {
          "interacted_by_person": 0,
          "frame_start": 80,
          "frame_end": 160,
          "confidence_peak": 0.84
        }
      ]
    },
    {
      "object_id": 2,
      "class": "cable",
      "motion_history": [
        { "frame_range": [90, 160], "state": "moving" }
      ],
      "interactions": [
        {
          "interacted_by_person": 0,
          "frame_start": 95,
          "frame_end": 155,
          "confidence_peak": 0.72
        }
      ]
    }
  ]
}
```

## Assumptions and Tradeoffs

- Base YOLO labels are limited for installation videos.
- The OpenCV fallback helps with cables and small parts, but it can miss low-contrast or heavily occluded objects.
- Frame sampling improves speed but can miss very short interactions.
- The tracker is intentionally simple and may swap IDs when objects overlap heavily.
- The interaction zone is not true hand-contact detection.
- JSON persistence is fine for local development, but a database would be better for a deployed service.

## Roadmap

- Collect and label installation-specific video frames.
- Fine-tune YOLO on labels like `cable`, `connector`, `panel`, `bracket`, and common tools.
- Compare the current tracker with ByteTrack or DeepSORT.
- Add hand-pose or wrist/keypoint signals for better interaction timing.
- Save confidence timelines for debugging person-object pairs.
- Add batch inference and GPU-friendly processing for longer videos.
- Move task persistence from JSON to SQLite or Postgres.

## Project Status

This is an active internship project. The assessment version established the baseline service, and the current work is focused on making the detector more useful for real installation footage.
