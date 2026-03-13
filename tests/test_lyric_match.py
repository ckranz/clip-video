"""Tests for lyric match mode."""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from clip_video.modes.lyric_match import (
    LyricMatchConfig,
    LineClipSet,
    LyricMatchProject,
    LyricMatchProcessor,
)
from clip_video.lyrics.phrases import ExtractionTarget, ExtractionList
from clip_video.search import SearchResults


class TestLyricMatchConfig:
    """Tests for LyricMatchConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = LyricMatchConfig()

        assert config.max_candidates_per_target == 10  # Changed for variety
        assert config.prefer_diversity is True
        assert config.extract_words is True
        assert config.extract_phrases is True
        assert config.output_format == "mp4"
        assert config.min_word_length == 1  # Capture all words
        assert config.use_stop_words is False  # Don't filter stop words
        assert config.min_phrase_words == 2
        assert config.max_phrase_words == 5
        assert config.shuffle_candidates is True  # For variety across videos
        assert config.fuzzy_matching is True

    def test_default_fuzzy_matching_enabled(self):
        config = LyricMatchConfig()
        assert config.fuzzy_matching is True

    def test_fuzzy_matching_can_be_disabled(self):
        config = LyricMatchConfig(fuzzy_matching=False)
        assert config.fuzzy_matching is False


class TestLineClipSet:
    """Tests for LineClipSet class."""

    def test_coverage_no_targets(self):
        """Test coverage with no targets."""
        lcs = LineClipSet(line_number=1, line_text="Test line")
        assert lcs.coverage == 100.0

    def test_coverage_no_clips(self):
        """Test coverage with targets but no clips."""
        lcs = LineClipSet(
            line_number=1,
            line_text="Test line",
            targets=[
                ExtractionTarget(text="hello", source_line=1, source_text=""),
                ExtractionTarget(text="world", source_line=1, source_text=""),
            ],
        )
        assert lcs.coverage == 0.0

    def test_coverage_partial(self):
        """Test partial coverage."""
        lcs = LineClipSet(
            line_number=1,
            line_text="Test line",
            targets=[
                ExtractionTarget(text="hello", source_line=1, source_text=""),
                ExtractionTarget(text="world", source_line=1, source_text=""),
            ],
            clips={
                "hello": [Path("/clips/hello.mp4")],
            },
        )
        assert lcs.coverage == 50.0

    def test_coverage_full(self):
        """Test full coverage."""
        lcs = LineClipSet(
            line_number=1,
            line_text="Test line",
            targets=[
                ExtractionTarget(text="hello", source_line=1, source_text=""),
                ExtractionTarget(text="world", source_line=1, source_text=""),
            ],
            clips={
                "hello": [Path("/clips/hello.mp4")],
                "world": [Path("/clips/world.mp4"), Path("/clips/world2.mp4")],
            },
        )
        assert lcs.coverage == 100.0

    def test_total_clips(self):
        """Test total clips calculation."""
        lcs = LineClipSet(
            line_number=1,
            line_text="Test line",
            clips={
                "hello": [Path("/clips/hello1.mp4"), Path("/clips/hello2.mp4")],
                "world": [Path("/clips/world.mp4")],
            },
        )
        assert lcs.total_clips == 3


class TestLyricMatchProject:
    """Tests for LyricMatchProject class."""

    def test_create_project(self):
        """Test creating a project."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
        )

        assert project.name == "test_project"
        assert project.brand_name == "test_brand"
        assert project.created_at  # Should have timestamp

    def test_project_root(self):
        """Test project root path."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
        )

        expected = Path.cwd() / "brands" / "test_brand" / "projects" / "test_project"
        assert project.project_root == expected

    def test_clips_dir(self):
        """Test clips directory path."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
        )

        assert project.clips_dir == project.project_root / "clips"

    def test_total_targets(self):
        """Test total targets count."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
            extraction_list=ExtractionList(
                lyrics_file=None,
                title=None,
                artist=None,
                targets_by_line={
                    1: [ExtractionTarget(text="a", source_line=1, source_text="")],
                    2: [
                        ExtractionTarget(text="b", source_line=2, source_text=""),
                        ExtractionTarget(text="c", source_line=2, source_text=""),
                    ],
                },
            ),
        )

        assert project.total_targets == 3

    def test_coverage_percent(self):
        """Test coverage percentage calculation."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
            line_clip_sets=[
                LineClipSet(
                    line_number=1,
                    line_text="First line",
                    targets=[ExtractionTarget(text="first", source_line=1, source_text="")],
                    clips={"first": [Path("/clips/first.mp4")]},
                ),
                LineClipSet(
                    line_number=2,
                    line_text="Second line",
                    targets=[ExtractionTarget(text="second", source_line=2, source_text="")],
                    clips={},  # No clips
                ),
            ],
        )

        assert project.coverage_percent == 50.0

    def test_get_coverage_gaps(self):
        """Test getting coverage gaps."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
            line_clip_sets=[
                LineClipSet(
                    line_number=1,
                    line_text="First line",
                    targets=[
                        ExtractionTarget(text="first", source_line=1, source_text=""),
                        ExtractionTarget(text="missing", source_line=1, source_text=""),
                    ],
                    clips={"first": [Path("/clips/first.mp4")]},
                ),
            ],
        )

        gaps = project.get_coverage_gaps()
        assert len(gaps) == 1
        assert gaps[0]["target_text"] == "missing"

    def test_get_summary(self):
        """Test getting project summary."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
            line_clip_sets=[
                LineClipSet(
                    line_number=1,
                    line_text="Test line",
                    targets=[ExtractionTarget(text="test", source_line=1, source_text="")],
                    clips={"test": [Path("/clips/test.mp4")]},
                ),
            ],
        )

        summary = project.get_summary()

        assert summary["project_name"] == "test_project"
        assert summary["brand_name"] == "test_brand"
        assert summary["total_lines"] == 1
        assert summary["coverage_percent"] == 100.0

    def test_serialization(self, tmp_path):
        """Test saving and loading project."""
        project = LyricMatchProject(
            name="test_project",
            brand_name="test_brand",
            lyrics_file=Path("/path/to/lyrics.txt"),
            line_clip_sets=[
                LineClipSet(
                    line_number=1,
                    line_text="Test line",
                    targets=[ExtractionTarget(text="test", source_line=1, source_text="")],
                    clips={"test": [Path("/clips/test.mp4")]},
                ),
            ],
        )

        # Save
        path = tmp_path / "project.json"
        project.save(path)

        # Load
        loaded = LyricMatchProject.load(path)

        assert loaded.name == project.name
        assert loaded.brand_name == project.brand_name
        assert len(loaded.line_clip_sets) == 1
        assert loaded.line_clip_sets[0].line_text == "Test line"


class TestLyricMatchProcessor:
    """Tests for LyricMatchProcessor class."""

    def test_init(self, tmp_path):
        """Test processor initialization."""
        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
        )

        assert processor.brand_name == "test_brand"
        assert processor.brands_root == tmp_path

    def test_create_project(self, tmp_path):
        """Test creating a new project."""
        # Create brands directory structure
        brand_path = tmp_path / "test_brand"
        brand_path.mkdir(parents=True)

        # Create lyrics file
        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text("""First line of the song
Second line here
Third line too""")

        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
        )

        project = processor.create_project("my_song", lyrics_file)

        assert project.name == "my_song"
        assert project.brand_name == "test_brand"
        assert project.extraction_list is not None
        assert len(project.line_clip_sets) == 3  # 3 lines

    def test_load_project(self, tmp_path):
        """Test loading an existing project."""
        # Create a project
        brand_path = tmp_path / "test_brand"
        brand_path.mkdir(parents=True)

        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text("Hello world")

        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
        )

        # Create and save
        project = processor.create_project("my_song", lyrics_file)
        project.save()

        # Load
        loaded = processor.load_project("my_song")

        assert loaded is not None
        assert loaded.name == "my_song"

    def test_load_nonexistent_project(self, tmp_path):
        """Test loading a project that doesn't exist."""
        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
        )

        result = processor.load_project("nonexistent")
        assert result is None

    def test_generate_report(self, tmp_path):
        """Test generating a project report."""
        brand_path = tmp_path / "test_brand"
        brand_path.mkdir(parents=True)

        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text("Hello world")

        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
        )

        project = processor.create_project("my_song", lyrics_file)
        report = processor.generate_report(project)

        assert "Lyric Match Project: my_song" in report
        assert "Brand: test_brand" in report
        assert "Word coverage:" in report

    def test_get_video_paths(self, tmp_path):
        """Test getting video file paths."""
        # Create brand with videos
        brand_path = tmp_path / "test_brand"
        videos_dir = brand_path / "videos"
        videos_dir.mkdir(parents=True)

        # Create fake video files
        (videos_dir / "video1.mp4").touch()
        (videos_dir / "video2.mov").touch()
        (videos_dir / "notavideo.txt").touch()  # Should be ignored

        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
        )

        project = LyricMatchProject(
            name="test",
            brand_name="test_brand",
            lyrics_file=Path("/fake/path.txt"),
        )

        paths = processor.get_video_paths(project)

        assert len(paths) == 2
        assert "video1" in paths
        assert "video2" in paths


class TestIntegration:
    """Integration tests for the full lyric match workflow."""

    def test_full_workflow_no_videos(self, tmp_path):
        """Test the full workflow without actual videos."""
        # Setup
        brand_path = tmp_path / "test_brand"
        brand_path.mkdir(parents=True)

        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text("""[Verse 1]
Hello beautiful world
How are you today

[Chorus]
We are singing along""")

        # Create processor and project
        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
        )

        project = processor.create_project("test_song", lyrics_file)

        # Verify project structure
        assert project.name == "test_song"
        assert len(project.line_clip_sets) == 3  # 3 content lines

        # Search (will find nothing without real transcripts)
        results = processor.search_all(project)

        # Should have searched for targets
        assert isinstance(results, dict)

        # Generate report
        report = processor.generate_report(project)
        assert "test_song" in report

        # Save report
        report_path = processor.save_report(project)
        assert report_path.exists()


def _make_project_with_targets(tmp_path, targets, brand_name="test_brand"):
    """Helper to create a project with specific extraction targets."""
    brand_path = tmp_path / brand_name
    brand_path.mkdir(parents=True, exist_ok=True)

    extraction_list = ExtractionList(
        lyrics_file=None,
        title=None,
        artist=None,
        targets_by_line={
            1: targets,
        },
    )

    project = LyricMatchProject(
        name="test_project",
        brand_name=brand_name,
        lyrics_file=Path("/fake/lyrics.txt"),
        extraction_list=extraction_list,
        line_clip_sets=[
            LineClipSet(
                line_number=1,
                line_text="test line",
                targets=list(targets),
            )
        ],
    )
    project._project_root = tmp_path / brand_name / "projects" / "test_project"
    return project


class TestSearchAllFuzzyMatching:
    """Tests for fuzzy matching integration in search_all."""

    @patch("clip_video.lyrics.fuzzy.FuzzyWordExpander")
    def test_fuzzy_expands_missing_words(self, MockExpander, tmp_path):
        expander_instance = MockExpander.return_value
        expander_instance.expand.return_value = {"hello": ["hallo"], "world": ["whirled"]}

        targets = [
            ExtractionTarget(text="hello", source_line=1, source_text="hello world"),
            ExtractionTarget(text="world", source_line=1, source_text="hello world"),
        ]
        project = _make_project_with_targets(tmp_path, targets)

        config = LyricMatchConfig(fuzzy_matching=True)
        processor = LyricMatchProcessor("test_brand", brands_root=tmp_path, config=config)

        processor.search_all(project)

        expander_instance.expand.assert_called_once()
        called_words = expander_instance.expand.call_args[0][0]
        assert "hello" in called_words
        assert "world" in called_words

    @patch("clip_video.lyrics.fuzzy.FuzzyWordExpander")
    def test_fuzzy_skipped_when_disabled(self, MockExpander, tmp_path):
        targets = [
            ExtractionTarget(text="hello", source_line=1, source_text="hello"),
        ]
        project = _make_project_with_targets(tmp_path, targets)

        config = LyricMatchConfig(fuzzy_matching=False)
        processor = LyricMatchProcessor("test_brand", brands_root=tmp_path, config=config)

        processor.search_all(project)

        MockExpander.assert_not_called()

    @patch("clip_video.lyrics.fuzzy.FuzzyWordExpander")
    def test_fuzzy_adds_alternatives_to_targets(self, MockExpander, tmp_path):
        expander_instance = MockExpander.return_value
        expander_instance.expand.return_value = {"hello": ["hallo", "hullo"]}

        targets = [
            ExtractionTarget(text="hello", source_line=1, source_text="hello"),
        ]
        project = _make_project_with_targets(tmp_path, targets)

        config = LyricMatchConfig(fuzzy_matching=True)
        processor = LyricMatchProcessor("test_brand", brands_root=tmp_path, config=config)

        processor.search_all(project)

        # Check alternatives were added to the target in the project
        target = project.line_clip_sets[0].targets[0]
        assert "hallo" in target.alternatives
        assert "hullo" in target.alternatives

    @patch("clip_video.lyrics.fuzzy.FuzzyWordExpander")
    def test_fuzzy_skipped_when_no_missing_words(self, MockExpander, tmp_path):
        """When all words have search results, fuzzy expansion is skipped."""
        targets = [
            ExtractionTarget(text="hello", source_line=1, source_text="hello"),
        ]
        project = _make_project_with_targets(tmp_path, targets)
        # Pre-populate clips so the word is "found"
        project.line_clip_sets[0].clips["hello"] = [Path("/clips/hello.mp4")]

        config = LyricMatchConfig(fuzzy_matching=True)
        processor = LyricMatchProcessor("test_brand", brands_root=tmp_path, config=config)

        # Mock the searcher to return results for "hello"
        mock_results = MagicMock(spec=SearchResults)
        mock_results.results = [MagicMock()]
        processor.searcher.search = MagicMock(return_value=mock_results)

        processor.search_all(project)

        MockExpander.assert_not_called()

    @patch("clip_video.lyrics.fuzzy.FuzzyWordExpander")
    def test_fuzzy_skips_words_with_existing_alternatives(self, MockExpander, tmp_path):
        expander_instance = MockExpander.return_value
        expander_instance.expand.return_value = {"world": ["whirled"]}

        targets = [
            ExtractionTarget(text="hello", source_line=1, source_text="hello world", alternatives=["hallo"]),
            ExtractionTarget(text="world", source_line=1, source_text="hello world"),
        ]
        project = _make_project_with_targets(tmp_path, targets)

        config = LyricMatchConfig(fuzzy_matching=True)
        processor = LyricMatchProcessor("test_brand", brands_root=tmp_path, config=config)

        processor.search_all(project)

        called_words = expander_instance.expand.call_args[0][0]
        assert "hello" not in called_words
        assert "world" in called_words
