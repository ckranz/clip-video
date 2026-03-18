import pytest
from fastapi.testclient import TestClient


class TestServerStartup:
    def test_health_endpoint(self, tmp_path):
        from clip_video.dashboard.server import create_app

        app = create_app(brand_name="test", brands_root=tmp_path)
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["brand"] == "test"

    def test_static_index(self, tmp_path):
        from clip_video.dashboard.server import create_app

        app = create_app(brand_name="test", brands_root=tmp_path)
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
