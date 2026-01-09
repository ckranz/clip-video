"""Tests for search and selection functionality."""

import pytest
from pathlib import Path
from datetime import datetime

from clip_video.search import (
    SearchResult,
    SearchResults,
    BrandSearcher,
    MultiQuerySearcher,
)
from clip_video.selection import (
    Selection,
    SelectionTracker,
    SelectionManager,
)
from clip_video.transcript.index import (
    TranscriptIndex,
    PhraseMatch,
    WordOccurrence,
)


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_from_phrase_match(self):
        """Test creating SearchResult from PhraseMatch."""
        word_occ = WordOccurrence(
            word="hello",
            start=1.0,
            end=1.5,
            confidence=0.95,
            project_name="project1",
            video_id="video1",
            segment_index=0,
            word_index=0,
        )

        match = PhraseMatch(
            phrase="hello",
            words=[word_occ],
            start=1.0,
            end=1.5,
            project_name="project1",
            video_id="video1",
        )

        result = SearchResult.from_phrase_match(match)

        assert result.phrase == "hello"
        assert result.start == 1.0
        assert result.end == 1.5
        assert result.confidence == 0.95
        assert result.video_id == "video1"

    def test_duration(self):
        """Test duration calculation."""
        result = SearchResult(
            phrase="test",
            start=1.0,
            end=2.5,
            project_name="p",
            video_id="v",
        )

        assert result.duration == 1.5

    def test_source_key(self):
        """Test source key generation."""
        result = SearchResult(
            phrase="test",
            start=1.0,
            end=2.0,
            project_name="project1",
            video_id="video1",
        )

        assert result.source_key == "project1:video1"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        result = SearchResult(
            phrase="hello world",
            start=1.0,
            end=2.0,
            project_name="project1",
            video_id="video1",
            confidence=0.9,
            rank_score=1.5,
        )

        data = result.to_dict()
        loaded = SearchResult.from_dict(data)

        assert loaded.phrase == result.phrase
        assert loaded.start == result.start
        assert loaded.end == result.end
        assert loaded.confidence == result.confidence
        assert loaded.rank_score == result.rank_score


class TestSearchResults:
    """Tests for SearchResults class."""

    def test_unique_videos(self):
        """Test getting unique video sources."""
        results = SearchResults(
            query="test",
            brand_name="brand1",
            results=[
                SearchResult(phrase="a", start=1, end=2, project_name="p", video_id="v1"),
                SearchResult(phrase="b", start=2, end=3, project_name="p", video_id="v1"),
                SearchResult(phrase="c", start=1, end=2, project_name="p", video_id="v2"),
            ],
        )

        assert len(results.unique_videos) == 2

    def test_by_video(self):
        """Test grouping results by video."""
        results = SearchResults(
            query="test",
            brand_name="brand1",
            results=[
                SearchResult(phrase="a", start=1, end=2, project_name="p", video_id="v1"),
                SearchResult(phrase="b", start=2, end=3, project_name="p", video_id="v1"),
                SearchResult(phrase="c", start=1, end=2, project_name="p", video_id="v2"),
            ],
        )

        by_video = results.by_video
        assert len(by_video["p:v1"]) == 2
        assert len(by_video["p:v2"]) == 1

    def test_top_results(self):
        """Test getting top results by rank."""
        results = SearchResults(
            query="test",
            brand_name="brand1",
            results=[
                SearchResult(phrase="a", start=1, end=2, project_name="p", video_id="v1", rank_score=0.5),
                SearchResult(phrase="b", start=2, end=3, project_name="p", video_id="v2", rank_score=0.9),
                SearchResult(phrase="c", start=1, end=2, project_name="p", video_id="v3", rank_score=0.7),
            ],
        )

        top = results.top_results(2)
        assert len(top) == 2
        assert top[0].phrase == "b"  # Highest rank
        assert top[1].phrase == "c"  # Second highest

    def test_save_and_load(self, tmp_path):
        """Test saving and loading search results."""
        results = SearchResults(
            query="hello world",
            brand_name="test_brand",
            results=[
                SearchResult(phrase="hello", start=1, end=2, project_name="p", video_id="v"),
            ],
            total_count=1,
        )

        path = tmp_path / "results.json"
        results.save(path)
        loaded = SearchResults.load(path)

        assert loaded.query == results.query
        assert loaded.brand_name == results.brand_name
        assert len(loaded.results) == 1


class TestSelectionTracker:
    """Tests for SelectionTracker class."""

    def test_is_available_no_selections(self):
        """Test availability with no prior selections."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        result = SearchResult(
            phrase="hello",
            start=1.0,
            end=2.0,
            project_name="p",
            video_id="v1",
        )

        assert tracker.is_available(result) is True

    def test_is_available_excluded_video(self):
        """Test availability with excluded video."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        tracker.exclude_video("v1")

        result = SearchResult(
            phrase="hello",
            start=1.0,
            end=2.0,
            project_name="p",
            video_id="v1",
        )

        assert tracker.is_available(result) is False

    def test_is_available_overlap(self):
        """Test availability with overlapping selection."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")

        # Make a selection
        result1 = SearchResult(
            phrase="hello",
            start=1.0,
            end=2.0,
            project_name="p",
            video_id="v1",
        )
        tracker.select("hello", result1, line_number=1)

        # Try overlapping result
        result2 = SearchResult(
            phrase="world",
            start=1.5,
            end=2.5,
            project_name="p",
            video_id="v1",
        )

        assert tracker.is_available(result2) is False

    def test_is_available_too_close(self):
        """Test availability with too-close selection."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        tracker.MIN_GAP_SECONDS = 5.0

        # Make a selection
        result1 = SearchResult(
            phrase="hello",
            start=1.0,
            end=2.0,
            project_name="p",
            video_id="v1",
        )
        tracker.select("hello", result1, line_number=1)

        # Try result that's too close
        result2 = SearchResult(
            phrase="world",
            start=4.0,  # Only 2 seconds after result1 ends
            end=5.0,
            project_name="p",
            video_id="v1",
        )

        assert tracker.is_available(result2) is False

        # But far enough should be ok
        result3 = SearchResult(
            phrase="far",
            start=10.0,
            end=11.0,
            project_name="p",
            video_id="v1",
        )

        assert tracker.is_available(result3) is True

    def test_select(self):
        """Test making a selection."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        result = SearchResult(
            phrase="hello",
            start=1.0,
            end=2.0,
            project_name="p",
            video_id="v1",
        )

        selection = tracker.select("hello", result, line_number=1)

        assert selection.target_text == "hello"
        assert selection.result == result
        assert selection.line_number == 1
        assert len(tracker.selections) == 1

    def test_select_unavailable_raises(self):
        """Test that selecting unavailable raises error."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        tracker.exclude_video("v1")

        result = SearchResult(
            phrase="hello",
            start=1.0,
            end=2.0,
            project_name="p",
            video_id="v1",
        )

        with pytest.raises(ValueError):
            tracker.select("hello", result, line_number=1)

    def test_auto_select(self):
        """Test automatic selection."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        results = [
            SearchResult(phrase="hello", start=1, end=2, project_name="p", video_id="v1", rank_score=0.9),
            SearchResult(phrase="hello", start=3, end=4, project_name="p", video_id="v2", rank_score=0.7),
        ]

        selection = tracker.auto_select("hello", results, line_number=1)

        assert selection is not None
        assert selection.result == results[0]  # Highest ranked

    def test_auto_select_diversity(self):
        """Test auto select preferring diversity."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")

        # First selection from v1
        result1 = SearchResult(phrase="first", start=1, end=2, project_name="p", video_id="v1")
        tracker.select("first", result1, line_number=1)

        # Now auto-select should prefer v2 even if v1 has higher rank
        results = [
            SearchResult(phrase="second", start=10, end=11, project_name="p", video_id="v1", rank_score=0.9),
            SearchResult(phrase="second", start=1, end=2, project_name="p", video_id="v2", rank_score=0.7),
        ]

        selection = tracker.auto_select("second", results, line_number=2, prefer_diversity=True)

        assert selection is not None
        assert selection.result.video_id == "v2"  # Prefers unused video

    def test_unselect(self):
        """Test removing a selection."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        result = SearchResult(phrase="hello", start=1, end=2, project_name="p", video_id="v1")
        selection = tracker.select("hello", result, line_number=1)

        assert len(tracker.selections) == 1

        tracker.unselect(selection)

        assert len(tracker.selections) == 0

    def test_get_selections_for_line(self):
        """Test filtering selections by line."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        tracker.select(
            "a",
            SearchResult(phrase="a", start=1, end=2, project_name="p", video_id="v1"),
            line_number=1,
        )
        tracker.select(
            "b",
            SearchResult(phrase="b", start=3, end=4, project_name="p", video_id="v2"),
            line_number=2,
        )
        tracker.select(
            "c",
            SearchResult(phrase="c", start=5, end=6, project_name="p", video_id="v3"),
            line_number=1,
        )

        line1_selections = tracker.get_selections_for_line(1)
        assert len(line1_selections) == 2

    def test_statistics(self):
        """Test getting statistics."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        tracker.select(
            "a",
            SearchResult(phrase="a", start=1, end=2, project_name="p", video_id="v1"),
            line_number=1,
        )
        tracker.select(
            "b",
            SearchResult(phrase="b", start=3, end=4, project_name="p", video_id="v2"),
            line_number=2,
        )

        stats = tracker.get_statistics()

        assert stats["total_selections"] == 2
        assert stats["unique_videos_used"] == 2
        assert stats["lines_with_selections"] == 2

    def test_save_and_load(self, tmp_path):
        """Test saving and loading tracker."""
        tracker = SelectionTracker(project_name="test", brand_name="brand")
        tracker.select(
            "hello",
            SearchResult(phrase="hello", start=1, end=2, project_name="p", video_id="v1"),
            line_number=1,
        )
        tracker.exclude_video("v2")
        tracker.exclude_range("v3", 10.0, 20.0)

        path = tmp_path / "tracker.json"
        tracker.save(path)
        loaded = SelectionTracker.load(path)

        assert loaded.project_name == tracker.project_name
        assert len(loaded.selections) == 1
        assert "v2" in loaded.excluded_videos
        assert "v3" in loaded.excluded_ranges


class TestSelection:
    """Tests for Selection class."""

    def test_source_key(self):
        """Test source key generation."""
        selection = Selection(
            target_text="hello",
            result=SearchResult(
                phrase="hello",
                start=1.5,
                end=2.5,
                project_name="proj",
                video_id="vid",
            ),
            line_number=1,
        )

        assert selection.source_key == "proj:vid:1.50-2.50"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        selection = Selection(
            target_text="hello",
            result=SearchResult(
                phrase="hello",
                start=1.0,
                end=2.0,
                project_name="p",
                video_id="v",
            ),
            line_number=1,
            notes="test note",
        )

        data = selection.to_dict()
        loaded = Selection.from_dict(data)

        assert loaded.target_text == selection.target_text
        assert loaded.line_number == selection.line_number
        assert loaded.notes == selection.notes


class TestSelectionManager:
    """Tests for SelectionManager class."""

    def test_get_creates_new(self, tmp_path):
        """Test that get creates a new tracker if none exists."""
        manager = SelectionManager(brands_root=tmp_path)
        tracker = manager.get("brand1", "project1")

        assert tracker.brand_name == "brand1"
        assert tracker.project_name == "project1"
        assert len(tracker.selections) == 0

    def test_save_and_get(self, tmp_path):
        """Test saving and loading via manager."""
        manager = SelectionManager(brands_root=tmp_path)

        # Create and save
        tracker = manager.get("brand1", "project1")
        tracker.select(
            "hello",
            SearchResult(phrase="hello", start=1, end=2, project_name="p", video_id="v"),
            line_number=1,
        )
        manager.save(tracker)

        # Load again
        loaded = manager.get("brand1", "project1")

        assert len(loaded.selections) == 1

    def test_exists(self, tmp_path):
        """Test checking if tracker exists."""
        manager = SelectionManager(brands_root=tmp_path)

        assert manager.exists("brand1", "project1") is False

        tracker = manager.get("brand1", "project1")
        manager.save(tracker)

        assert manager.exists("brand1", "project1") is True

    def test_delete(self, tmp_path):
        """Test deleting a tracker."""
        manager = SelectionManager(brands_root=tmp_path)

        tracker = manager.get("brand1", "project1")
        manager.save(tracker)

        assert manager.exists("brand1", "project1") is True

        result = manager.delete("brand1", "project1")

        assert result is True
        assert manager.exists("brand1", "project1") is False
