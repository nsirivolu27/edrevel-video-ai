import json
from dataclasses import replace
from pathlib import Path
from threading import Lock

from app.models import Task, TaskStatus, now_utc

DB_PATH = Path("data/tasks.json")
_lock = Lock()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text("[]", encoding="utf-8")


def create_task(
    task_id: str,
    filename: str,
    video_path: str,
    classes: list[str] | None = None,
) -> Task:
    task = Task(
        task_id=task_id,
        filename=filename,
        video_path=video_path,
        status=TaskStatus.queued,
        created_at=now_utc(),
        updated_at=now_utc(),
        classes=classes,  # optional override; None lets the pipeline fall back to DEFAULT_CLASSES
    )
    with _lock:
        tasks = _read()
        tasks.append(task)
        _write(tasks)
    return task


def get_task(task_id: str) -> Task | None:
    with _lock:
        return next((task for task in _read() if task.task_id == task_id), None)


def list_tasks() -> list[Task]:
    with _lock:
        return sorted(_read(), key=lambda task: task.created_at, reverse=True)


def update_task(
    task_id: str,
    status: TaskStatus,
    error: str | None = None,
    result_path: str | None = None,
) -> Task:
    with _lock:
        tasks = _read()
        for idx, task in enumerate(tasks):
            if task.task_id != task_id:
                continue
            tasks[idx] = replace(
                task,
                status=status,
                error=error,
                result_path=result_path or task.result_path,
                updated_at=now_utc(),
            )
            _write(tasks)
            return tasks[idx]
    raise KeyError(task_id)


def _read() -> list[Task]:
    init_db()
    raw = json.loads(DB_PATH.read_text(encoding="utf-8"))
    return [Task.from_dict(item) for item in raw]


def _write(tasks: list[Task]) -> None:
    DB_PATH.write_text(
        json.dumps([task.to_dict() for task in tasks], indent=2),
        encoding="utf-8",
    )
