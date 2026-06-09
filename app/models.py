from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum


class TaskStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


@dataclass
class Task:
    task_id: str
    filename: str
    status: TaskStatus
    video_path: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result_path: str | None = None
    classes: list[str] | None = None  # per-upload detection vocabulary; None means the pipeline falls back to DEFAULT_CLASSES

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            task_id=data["task_id"],
            filename=data["filename"],
            status=TaskStatus(data["status"]),
            video_path=data["video_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            error=data.get("error"),
            result_path=data.get("result_path"),
            classes=data.get("classes"),  # optional; older saved tasks won't have it, hence .get
        )


def now_utc() -> datetime:
    return datetime.now(timezone.utc)