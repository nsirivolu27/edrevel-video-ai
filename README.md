# Edrevel Video Object Interaction

Small FastAPI project for the Edrevel AI SDE Intern assessment.

The app takes an installation-style video, runs YOLO on sampled frames, tracks objects/people with a simple centroid + IoU tracker, and tries to estimate when a person is interacting with an object. It also exports a few annotated keyframes for the most interesting moments.

I tried to keep this as something I could reasonably build and explain in 1-2 days, so the ML part is intentionally practical instead of fancy.

## What it does

- Upload a video through `POST /tasks/upload`
- Process every Nth frame, while keeping original frame numbers
- Detect people separately from other objects
- Track objects with a small custom tracker
- Classify each object as moving/stationary over time
- Estimate person-object interactions using proximity, overlap, motion changes, and confidence smoothing
- Save annotated keyframes under `outputs/{task_id}/keyframes/`
- Store task state in a local JSON file

## Project layout

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

There are no separate repository/service/provider layers because this does not need them. Most of the interesting code is in `video_pipeline.py`, `tracking.py`, and `interaction_logic.py`.

## How the pipeline works

1. Probe the video first for FPS, frame count, duration, and resolution.
2. Sample every `SAMPLE_EVERY` frames to keep runtime reasonable.
3. Run YOLO and split detections into people and candidate objects.
4. Track people and objects separately with a centroid + IoU matcher.
5. Smooth object center movement and compress repeated states into frame ranges.
6. Build an upper-body interaction zone for each person.
7. Score nearby objects based on distance, zone overlap, broad proximity, and whether the object just started moving.
8. Increase/decrease pair confidence over time so one noisy frame does not count as an interaction.
9. Export result JSON and a few debug keyframes.

## Why a custom tracker?

ByteTrack or DeepSORT would probably track better, but I wanted the assignment logic to be easy to explain in an interview. This tracker matches only when:

- the class label is the same
- the new box overlaps the old box enough, or
- the center moved a reasonable distance

It is not production-grade tracking, but it is readable and good enough for showing the reasoning layer around detections.

## Interaction heuristic

I skipped hand-pose estimation on purpose. It would be more accurate, but it adds another model and more setup work. For a short assessment project, I used a practical approximation:

- take the upper part of the person box
- expand it a bit to represent likely arm/hand area
- score objects near or overlapping that zone
- give a small bonus when an object transitions from stationary to moving
- require confidence to stay high for more than one sampled frame

This will miss some edge cases, but it avoids the worst one-frame false positives.

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

## API examples

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

The tests cover the deterministic pieces: IoU, box expansion, ID stability, motion compression, confidence smoothing, and the small debug endpoints.

## Example output

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
    }
  ]
}
```

## Assumptions and tradeoffs

- YOLO COCO labels are limited, so some installation-specific items may show up with generic labels.
- Sampling frames improves speed but can miss short interactions.
- The tracker is intentionally simple and may swap IDs if objects overlap heavily.
- The interaction zone is a heuristic, not actual contact detection.
- JSON persistence is fine for a local assessment, but not ideal for many users at once.
- Some thresholds are hardcoded because they are easier to tune while watching real videos.

## Future improvements

- Add a hand pose model for better contact detection.
- Try ByteTrack or DeepSORT for more stable identities.
- Fine-tune labels for tools, cables, brackets, panels, and other installation objects.
- Batch YOLO inference for speed.
- Add optional GPU setup notes.
- Save pair confidence timelines for debugging.

## Notes

This is written as a prototype I would be comfortable walking through in an interview: the pieces are small, the matching/scoring logic is visible, and the TODOs are real next steps instead of pretending the project is complete production software.
