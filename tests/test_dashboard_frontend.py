from pathlib import Path

import pytest
from fastapi.testclient import TestClient

STATIC_DIR = Path(__file__).parent.parent / "src" / "clip_video" / "dashboard" / "static"


class TestStaticFilesExist:
    def test_index_html_exists(self):
        assert (STATIC_DIR / "index.html").is_file()

    def test_app_js_exists(self):
        assert (STATIC_DIR / "app.js").is_file()

    def test_style_css_exists(self):
        assert (STATIC_DIR / "style.css").is_file()


class TestIndexHtmlStructure:
    @pytest.fixture()
    def html(self):
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def test_loads_vue(self, html):
        assert "vue@3" in html

    def test_loads_tailwind(self, html):
        assert "tailwindcss" in html

    def test_loads_app_js(self, html):
        assert "/static/app.js" in html

    def test_loads_style_css(self, html):
        assert "/static/style.css" in html

    def test_has_app_mount_point(self, html):
        assert 'id="app"' in html

    def test_has_tab_navigation(self, html):
        assert "currentTab" in html

    def test_has_brand_header(self, html):
        assert "brand.name" in html
        assert "brand.logo_url" in html

    def test_has_three_views(self, html):
        assert "review-view" in html
        assert "schedule-view" in html
        assert "tasks-view" in html


class TestAppJsStructure:
    @pytest.fixture()
    def js(self):
        return (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    def test_creates_vue_app(self, js):
        assert "createApp" in js

    def test_mounts_app(self, js):
        assert "app.mount" in js

    def test_defines_review_view(self, js):
        assert "ReviewView" in js

    def test_defines_schedule_view(self, js):
        assert "ScheduleView" in js

    def test_defines_tasks_view(self, js):
        assert "TasksView" in js

    def test_fetches_brand_info(self, js):
        assert "/api/brand" in js

    def test_has_tab_state(self, js):
        assert "currentTab" in js

    def test_default_tab_is_review(self, js):
        assert "ref('review')" in js


class TestStyleCss:
    @pytest.fixture()
    def css(self):
        return (STATIC_DIR / "style.css").read_text(encoding="utf-8")

    def test_has_scrollbar_styles(self, css):
        assert "scrollbar" in css

    def test_has_video_styles(self, css):
        assert "video" in css


class TestStaticFilesServed:
    def test_app_js_served(self, tmp_path):
        from clip_video.dashboard.server import create_app

        app = create_app(brand_name="test", brands_root=tmp_path)
        client = TestClient(app)
        resp = client.get("/static/app.js")
        assert resp.status_code == 200
        assert "createApp" in resp.text

    def test_style_css_served(self, tmp_path):
        from clip_video.dashboard.server import create_app

        app = create_app(brand_name="test", brands_root=tmp_path)
        client = TestClient(app)
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
        assert "scrollbar" in resp.text

    def test_index_contains_vue_app(self, tmp_path):
        from clip_video.dashboard.server import create_app

        app = create_app(brand_name="test", brands_root=tmp_path)
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "vue@3" in resp.text
