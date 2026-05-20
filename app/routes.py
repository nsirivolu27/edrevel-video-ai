import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app import database
from app.video_pipeline import UPLOAD_DIR, OUTPUT_DIR, process_task

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/debug/paths")
def debug_paths():
    # Kept intentionally small. Helpful while running the assessment locally.
    return {
        "uploads": str(UPLOAD_DIR.resolve()),
        "outputs": str(OUTPUT_DIR.resolve()),
        "db": str(database.DB_PATH.resolve()),
    }


@router.post("/tasks/upload")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    database.init_db()
    task_id = str(uuid.uuid4())
    safe_name = Path(file.filename or "video.mp4").name
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    video_path = task_dir / safe_name

    with video_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    task = database.create_task(task_id, safe_name, str(video_path))
    background_tasks.add_task(process_task, task_id)
    return {"task_id": task.task_id, "status": task.status}


@router.get("/tasks")
def tasks():
    return {"tasks": database.list_tasks()}


@router.get("/tasks/{task_id}/status")
def task_status(task_id: str):
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/tasks/{task_id}/result")
def task_result(task_id: str):
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not task.result_path:
        raise HTTPException(status_code=409, detail=f"Task is still {task.status}")

    path = Path(task.result_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    return json.loads(path.read_text(encoding="utf-8"))
