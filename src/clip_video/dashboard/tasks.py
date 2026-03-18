from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    task_type: str
    target: str
    description: str
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0
    error: str | None = None
    created_at: str = ""
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "target": self.target,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class TaskQueue:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def submit(self, task_type: str, target: str, description: str, **metadata: Any) -> str:
        task_id = uuid.uuid4().hex[:12]
        self._tasks[task_id] = Task(
            task_id=task_id,
            task_type=task_type,
            target=target,
            description=description,
            created_at=datetime.now().isoformat(),
            metadata=metadata,
        )
        return task_id

    def get(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def update(
        self,
        task_id: str,
        status: TaskStatus | None = None,
        progress: float | None = None,
        error: str | None = None,
    ) -> None:
        task = self._tasks[task_id]
        if status is not None:
            task.status = status
        if progress is not None:
            task.progress = progress
        if error is not None:
            task.error = error
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            task.completed_at = datetime.now().isoformat()

    def list_tasks(self) -> list[Task]:
        return list(self._tasks.values())
