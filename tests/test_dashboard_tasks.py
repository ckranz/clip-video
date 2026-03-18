import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from clip_video.dashboard.tasks import TaskQueue, TaskStatus
from clip_video.dashboard.server import create_app


class TestTaskQueue:
    def test_submit_returns_task_id(self):
        queue = TaskQueue()
        task_id = queue.submit("reprocess", target="talk-a.mp4", description="Re-crop talk-a")
        assert task_id is not None

    def test_get_task(self):
        queue = TaskQueue()
        task_id = queue.submit("reprocess", target="talk-a.mp4", description="Re-crop")
        task = queue.get(task_id)
        assert task.status == TaskStatus.QUEUED
        assert task.task_type == "reprocess"
        assert task.target == "talk-a.mp4"

    def test_list_tasks(self):
        queue = TaskQueue()
        queue.submit("reprocess", target="a.mp4", description="A")
        queue.submit("reprocess", target="b.mp4", description="B")
        assert len(queue.list_tasks()) == 2

    def test_update_progress(self):
        queue = TaskQueue()
        task_id = queue.submit("reprocess", target="a.mp4", description="A")
        queue.update(task_id, status=TaskStatus.RUNNING, progress=50.0)
        task = queue.get(task_id)
        assert task.status == TaskStatus.RUNNING
        assert task.progress == 50.0

    def test_complete_task(self):
        queue = TaskQueue()
        task_id = queue.submit("reprocess", target="a.mp4", description="A")
        queue.update(task_id, status=TaskStatus.COMPLETED, progress=100.0)
        task = queue.get(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_fail_task(self):
        queue = TaskQueue()
        task_id = queue.submit("reprocess", target="a.mp4", description="A")
        queue.update(task_id, status=TaskStatus.FAILED, error="FFmpeg crash")
        task = queue.get(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.error == "FFmpeg crash"

    def test_task_serialization(self):
        queue = TaskQueue()
        task_id = queue.submit("reprocess", target="a.mp4", description="A")
        d = queue.get(task_id).to_dict()
        assert d["task_type"] == "reprocess"
        assert d["status"] == "queued"


class TestTasksAPI:
    def test_tasks_endpoint(self, tmp_path):
        brand_path = tmp_path / "testbrand"
        brand_path.mkdir()
        (brand_path / "config.json").write_text(json.dumps({"name": "testbrand"}))

        app = create_app(brand_name="testbrand", brands_root=tmp_path)
        app.state.task_queue.submit("reprocess", target="a.mp4", description="A")

        client = TestClient(app)
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["task_type"] == "reprocess"
        assert data[0]["status"] == "queued"

    # SSE stream endpoint (/api/tasks/stream) is tested manually —
    # TestClient hangs on streaming responses that never complete.
