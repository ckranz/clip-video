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


class TestReviewViewImplementation:
    @pytest.fixture()
    def js(self):
        return (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    def test_fetches_videos_on_mount(self, js):
        assert "/api/videos" in js

    def test_has_status_filter(self, js):
        assert "statusFilter" in js

    def test_has_all_status_filter_options(self, js):
        for status in ["all", "new", "selected", "skipped", "scheduled", "posted"]:
            assert f"'{status}'" in js

    def test_has_clip_version_toggle(self, js):
        for version in ["raw", "portrait", "final"]:
            assert f"'{version}'" in js

    def test_has_video_player(self, js):
        assert "<video" in js
        assert "controls" in js

    def test_has_status_badge_styles(self, js):
        assert "bg-gray-600" in js  # new
        assert "bg-blue-600" in js  # selected
        assert "bg-amber-600" in js  # scheduled
        assert "bg-green-600" in js  # posted

    def test_has_speaker_input(self, js):
        assert "onSpeakerBlur" in js

    def test_has_position_dropdown(self, js):
        assert "onPositionChange" in js
        assert "speaker_position" in js

    def test_has_youtube_url_input(self, js):
        assert "onYoutubeBlur" in js
        assert "youtube_url" in js

    def test_has_reprocess_button(self, js):
        assert "reprocessVideo" in js
        assert "/reprocess" in js

    def test_has_select_and_skip_buttons(self, js):
        assert "setClipStatus" in js
        assert "'selected'" in js
        assert "'skipped'" in js

    def test_optimistic_update_with_rollback(self, js):
        assert "oldStatus" in js

    def test_has_social_copy_section(self, js):
        assert "copyToClipboard" in js
        assert "navigator.clipboard" in js

    def test_has_toast_notifications(self, js):
        assert "showToast" in js
        assert "ToastContainer" in js

    def test_has_quality_score_display(self, js):
        assert "quality_score" in js

    def test_has_duration_formatter(self, js):
        assert "formatDuration" in js

    def test_has_topics_display(self, js):
        assert "clip.topics" in js

    def test_has_hook_text_display(self, js):
        assert "clip.hook_text" in js

    def test_has_summary_display(self, js):
        assert "clip.summary" in js

    def test_has_responsive_grid(self, js):
        assert "grid-cols-1" in js
        assert "md:grid-cols-2" in js
        assert "xl:grid-cols-3" in js

    def test_media_url_helper(self, js):
        assert "mediaUrl" in js

    def test_has_dark_theme_classes(self, js):
        assert "bg-gray-900" in js
        assert "border-gray-800" in js

    def test_has_filter_counts(self, js):
        assert "statusCounts" in js


class TestIndexHtmlHasToastContainer:
    def test_toast_container_in_html(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "toast-container" in html


class TestStyleCss:
    @pytest.fixture()
    def css(self):
        return (STATIC_DIR / "style.css").read_text(encoding="utf-8")

    def test_has_scrollbar_styles(self, css):
        assert "scrollbar" in css

    def test_has_video_styles(self, css):
        assert "video" in css


class TestScheduleViewImplementation:
    @pytest.fixture()
    def js(self):
        return (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    def test_fetches_schedule_on_mount(self, js):
        assert "/api/schedule" in js

    def test_fetches_videos_for_clip_picker(self, js):
        assert "selectedClips" in js or "availableClips" in js

    def test_has_month_navigation(self, js):
        assert "prevMonth" in js
        assert "nextMonth" in js

    def test_has_calendar_grid(self, js):
        assert "grid-cols-7" in js

    def test_has_day_of_week_headers(self, js):
        assert "Mon" in js
        assert "Tue" in js
        assert "Sun" in js

    def test_has_today_highlight(self, js):
        assert "ring-blue-500" in js

    def test_has_platform_colors(self, js):
        assert "linkedin" in js
        assert "youtube" in js

    def test_has_schedule_modal(self, js):
        assert "showScheduleModal" in js or "scheduleModal" in js

    def test_posts_schedule_to_api(self, js):
        assert "/schedule" in js
        assert "POST" in js

    def test_deletes_schedule_entry(self, js):
        assert "DELETE" in js

    def test_has_remove_button(self, js):
        assert "removeSchedule" in js or "deleteSchedule" in js

    def test_has_dimmed_empty_days(self, js):
        assert "opacity" in js or "dimmed" in js or "text-gray-600" in js

    def test_has_outside_month_dimming(self, js):
        assert "outside" in js or "other-month" in js or "text-gray-700" in js

    def test_has_responsive_list_fallback(self, js):
        assert "sm:hidden" in js or "md:grid" in js or "lg:grid" in js

    def test_has_detail_view_for_scheduled_clip(self, js):
        assert "hook_text" in js
        assert "summary" in js

    def test_computes_calendar_days(self, js):
        assert "calendarDays" in js or "daysInMonth" in js

    def test_has_dark_theme_calendar_cells(self, js):
        assert "bg-gray-900" in js
        assert "border-gray-800" in js


class TestTasksViewImplementation:
    @pytest.fixture()
    def js(self):
        return (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    def test_emits_task_count_event(self, js):
        assert "task-count" in js

    def test_uses_event_source_for_sse(self, js):
        assert "EventSource" in js
        assert "/api/tasks/stream" in js

    def test_has_polling_fallback(self, js):
        assert "/api/tasks" in js
        assert "setInterval" in js or "polling" in js.lower()

    def test_has_empty_state_message(self, js):
        assert "No background tasks" in js

    def test_has_status_badges(self, js):
        assert "queued" in js
        assert "running" in js
        assert "completed" in js
        assert "failed" in js

    def test_has_progress_bar(self, js):
        assert "bg-blue-500" in js
        assert "bg-gray-800" in js

    def test_has_relative_time_formatting(self, js):
        assert "ago" in js

    def test_has_error_display(self, js):
        assert "error" in js.lower()

    def test_cleans_up_on_unmount(self, js):
        assert "onUnmounted" in js

    def test_has_task_card_styling(self, js):
        assert "task_type" in js or "task.type" in js or "task.task_type" in js

    def test_sorts_tasks(self, js):
        assert "sort" in js

    def test_computes_active_count(self, js):
        assert "activeCount" in js or "active" in js.lower()


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
